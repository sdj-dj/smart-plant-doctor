import json
import argparse
from pathlib import Path

from ml.model import load_model, predict_image

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_class_names(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    names = [""] * len(mapping)

    for name, class_id in mapping.items():
        names[int(class_id)] = name

    if any(not name for name in names):
        raise ValueError(f"Missing class IDs in {file_path}")

    return names


def main():
    parser = argparse.ArgumentParser(
        description="Smart Plant Doctor prediction"
    )

    parser.add_argument("--image", required=True)
    parser.add_argument("--top-k", type=int, default=3)

    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be at least 1")

    image_path = Path(args.image).expanduser()

    if not image_path.is_absolute():
        image_path = PROJECT_ROOT / image_path

    checkpoint_path = (
        PROJECT_ROOT
        / "checkpoints"
        / "resnet18_agricultural_ai_step1.pth"
    )

    species_names = load_class_names(
        PROJECT_ROOT / "audit" / "species_classes.json"
    )

    condition_names = load_class_names(
        PROJECT_ROOT / "audit" / "condition_classes.json"
    )

    print("Loading Smart Plant Doctor...", flush=True)

    model, device = load_model(checkpoint_path)

    print(f"Device: {device}", flush=True)
    print(f"Image: {image_path.name}", flush=True)
    print("Running prediction...", flush=True)

    results = predict_image(
        model=model,
        image_path=image_path,
        device=device,
        species_names=species_names,
        condition_names=condition_names,
        top_k=args.top_k,
    )

    print("\n" + "=" * 45)
    print("       SMART PLANT DOCTOR")
    print("=" * 45)

    print("\nPLANT SPECIES CANDIDATES")

    for i, item in enumerate(
        results["species_candidates"], 1
    ):
        print(
            f"{i}. {item['label']} "
            f"({item['score']:.2%} score)"
        )

    print("\nCONDITION CANDIDATES")

    for i, item in enumerate(
        results["condition_candidates"], 1
    ):
        print(
            f"{i}. {item['label']} "
            f"({item['score']:.2%} score)"
        )

    print("\nIMPORTANT")
    print(results["notice"])
    print("=" * 45)


if __name__ == "__main__":
    main()