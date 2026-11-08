import json
import tempfile
import time
from pathlib import Path

import gradio as gr

from ml.model import load_model, predict_image

ROOT = Path(__file__).resolve().parent

CHECKPOINT = (
    ROOT / "checkpoints" / "experiments" / "leakage_reduced"
    / "condition_balanced_best_v1.pth"
)

with open(ROOT / "audit/species_classes.json", encoding="utf-8") as f:
    species_mapping = json.load(f)

with open(ROOT / "audit/condition_classes.json", encoding="utf-8") as f:
    condition_mapping = json.load(f)

species_names = [
    name for name, _ in sorted(species_mapping.items(), key=lambda x: x[1])
]
condition_names = [
    name for name, _ in sorted(condition_mapping.items(), key=lambda x: x[1])
]

model, device = load_model(CHECKPOINT)

print("\nSmart Plant Doctor loaded.")
print("Device:", device)
print("Species classes:", len(species_names))
print("Condition classes:", len(condition_names))


# ============================================================
# IMAGE PREDICTION
# ============================================================

def predict_pil(image):
    """Run the existing trained model on a PIL image."""

    if image is None:
        raise ValueError("No image was supplied.")

    image_path = None

    try:
        image = image.convert("RGB")

        with tempfile.NamedTemporaryFile(
            suffix=".jpg", delete=False
        ) as temp:
            image_path = Path(temp.name)

        image.save(image_path, format="JPEG", quality=95)

        return predict_image(
            model=model,
            image_path=image_path,
            device=device,
            species_names=species_names,
            condition_names=condition_names,
            top_k=3,
        )

    finally:
        if image_path is not None:
            image_path.unlink(missing_ok=True)


def build_summary(result, humidity=None, starch_status="Not tested"):
    """Create a detailed report without claiming a confirmed diagnosis."""

    lines = [
        "## Plant Health Report",
        "",
        "### Species candidates",
    ]

    for item in result["species_candidates"]:
        lines.append(
            f"- **{item['label']}** — "
            f"{item['score'] * 100:.2f}% model score"
        )

    lines.extend(["", "### Condition candidates"])

    for item in result["condition_candidates"]:
        lines.append(
            f"- **{item['label']}** — "
            f"{item['score'] * 100:.2f}% model score"
        )

    lines.extend(["", "### Environmental context"])

    if humidity is not None:
        try:
            value = float(humidity)
            if 0 <= value <= 100:
                lines.append(
                    f"- User-entered relative humidity: **{value:.1f}%**"
                )
                lines.append(
                    "- This is not a camera measurement. Confirm it "
                    "with a humidity sensor."
                )
            else:
                lines.append("- Humidity is outside the valid 0–100% range.")
        except (TypeError, ValueError):
            lines.append("- Humidity is unavailable.")
    else:
        lines.append("- Humidity was not provided.")

    lines.extend(["", "### Starch assessment"])

    if starch_status == "Not tested":
        lines.append("- Unknown: no starch test result was provided.")
    else:
        lines.append(f"- User-reported assessment: **{starch_status}**")
        lines.append(
            "- This is not a starch measurement made by the AI."
        )

    lines.extend([
        "",
        "### Limitations",
        "- Model scores are not calibrated probabilities.",
        "- A high score does not guarantee a correct prediction.",
        "- Small crops can omit useful leaf context.",
        "- Similar visible symptoms can have different causes.",
        "- Confirm suspected disease or nutrient deficiency before treatment.",
        "",
        f"**Device:** `{device}`",
        f"**Model notice:** {result.get('notice', 'Preliminary screening only.')}",
    ])

    return "\n".join(lines)


def analyze_image(image, humidity=None, starch_status="Not tested"):
    if image is None:
        return "Please capture or upload an image first."

    try:
        result = predict_pil(image)
        return build_summary(result, humidity, starch_status)
    except Exception as error:
        print("Image analysis error:", error)
        return "Analysis failed. Check the terminal for details."


# ============================================================
# REGION CROPPING
# ============================================================

def get_region_image(editor_value):
    """
    Gradio ImageEditor usually returns a dictionary containing
    background, layers, and composite. Use the edited composite
    so that the user's crop is analyzed.
    """

    if editor_value is None:
        return None

    if hasattr(editor_value, "convert"):
        return editor_value

    if isinstance(editor_value, dict):
        for key in ("composite", "background"):
            image = editor_value.get(key)
            if image is not None and hasattr(image, "convert"):
                return image

    return None


def analyze_region(full_leaf, editor_value, humidity, starch_status):
    region = get_region_image(editor_value)

    if region is None:
        return (
            "## Region image missing\n\n"
            "Upload an image in the region editor, then use its crop "
            "tool to select the portion you want to inspect."
        )

    try:
        region_result = predict_pil(region)

        lines = [
            "# Leaf Region Analysis",
            "",
            "The predictions below are generated from the edited "
            "region image supplied by the crop editor.",
            "",
            "## Selected-region predictions",
            "",
            build_summary(region_result, humidity, starch_status),
        ]

        if full_leaf is not None:
            full_result = predict_pil(full_leaf)

            lines.extend([
                "",
                "---",
                "",
                "## Whole-leaf predictions",
                "",
                build_summary(full_result, humidity, starch_status),
                "",
                "## Comparison",
                "",
            ])

            full_condition = (
                full_result["condition_candidates"][0]["label"]
                if full_result["condition_candidates"] else "Unavailable"
            )
            region_condition = (
                region_result["condition_candidates"][0]["label"]
                if region_result["condition_candidates"] else "Unavailable"
            )

            lines.append(f"- Whole-leaf top condition: **{full_condition}**")
            lines.append(f"- Region top condition: **{region_condition}**")

            if full_condition == region_condition:
                lines.append(
                    "- Both inputs produced the same top condition label."
                )
            else:
                lines.append(
                    "- The top labels differ. A small crop may change "
                    "the prediction because it removes context."
                )

            lines.append(
                "\nThis comparison does not prove that the selected "
                "region has a separate disease or deficiency."
            )

        return "\n".join(lines)

    except Exception as error:
        print("Region analysis error:", error)
        return (
            "Region analysis failed. Check the terminal for details "
            "and try a clear image."
        )


# ============================================================
# CONTINUOUS CAMERA SCANNING
# ============================================================

last_scan_time = 0.0
MIN_SCAN_INTERVAL = 1.5


def continuous_scan(image, enabled, humidity, starch_status):
    global last_scan_time

    if not enabled:
        return gr.skip()

    if image is None:
        return "Waiting for a webcam frame..."

    now = time.monotonic()

    if now - last_scan_time < MIN_SCAN_INTERVAL:
        return gr.skip()

    last_scan_time = now
    return analyze_image(image, humidity, starch_status)


# ============================================================
# INTERFACE
# ============================================================

with gr.Blocks(title="Smart Plant Doctor") as demo:

    gr.Markdown(
        """
        # Smart Plant Doctor
        ### Live plant scanner and leaf-region analysis

        Scan a live leaf, upload a photograph, or crop a specific
        part of a leaf to compare its predictions with the whole leaf.
        """
    )

    with gr.Row():

        with gr.Column():

            gr.Markdown("## Live Scanner")

            camera = gr.Image(
                sources=["webcam"],
                type="pil",
                label="Live Webcam",
                streaming=True,
            )

            scan_button = gr.Button(
                "Scan Current Frame",
                variant="primary",
            )

            continuous_enabled = gr.Checkbox(
                label="Enable continuous scanning",
                value=False,
            )

            gr.Markdown("## Upload an Image")

            upload = gr.Image(
                sources=["upload"],
                type="pil",
                label="Whole Leaf Image",
            )

            upload_button = gr.Button("Analyze Uploaded Image")

        with gr.Column():

            gr.Markdown("## Environmental Information")

            humidity = gr.Number(
                label="Relative humidity (%)",
                value=None,
                minimum=0,
                maximum=100,
                precision=1,
            )

            starch_status = gr.Dropdown(
                choices=[
                    "Not tested",
                    "Starch detected in qualitative test",
                    "Little or no starch detected in qualitative test",
                    "Quantitative test performed — value not entered",
                ],
                value="Not tested",
                label="Starch test assessment",
            )

    gr.Markdown("---")
    gr.Markdown("## Leaf Region Scanner")

    gr.Markdown(
        """
        1. Upload a clear leaf photograph into the editor.
        2. Use the editor's crop tool to select the spot or region.
        3. Keep the crop focused on the area you want to inspect.
        4. Optionally provide the original whole-leaf image below
           so its predictions can be compared with the crop.
        """
    )

    region_editor = gr.ImageEditor(
        type="pil",
        sources=["upload"],
        label="Upload and crop a leaf region",
    )

    full_leaf_comparison = gr.Image(
        sources=["upload"],
        type="pil",
        label="Optional: original whole-leaf image for comparison",
    )

    region_button = gr.Button(
        "Analyze Selected Region",
        variant="primary",
    )

    gr.Markdown("## AI Results")

    analysis = gr.Markdown(
        "Scan a leaf or upload an image to begin."
    )

    scan_button.click(
        fn=analyze_image,
        inputs=[camera, humidity, starch_status],
        outputs=analysis,
        concurrency_limit=1,
    )

    upload_button.click(
        fn=analyze_image,
        inputs=[upload, humidity, starch_status],
        outputs=analysis,
        concurrency_limit=1,
    )

    region_button.click(
        fn=analyze_region,
        inputs=[
            full_leaf_comparison,
            region_editor,
            humidity,
            starch_status,
        ],
        outputs=analysis,
        concurrency_limit=1,
    )

    camera.stream(
        fn=continuous_scan,
        inputs=[
            camera,
            continuous_enabled,
            humidity,
            starch_status,
        ],
        outputs=analysis,
        stream_every=1.5,
        time_limit=3600,
        concurrency_limit=1,
        trigger_mode="always_last",
    )

    gr.Markdown(
        """
        ---
        **Scientific note:** Model predictions are preliminary.
        Humidity must be measured by a sensor, and starch status
        must come from an appropriate test. Cropping does not
        create new model capabilities or confirm a diagnosis.
        """
    )


if __name__ == "__main__":
    demo.launch()
