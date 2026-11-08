from pathlib import Path
from collections import defaultdict

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = ROOT / "audit" / "training_manifest.csv"
VERIFICATION_PATH = ROOT / "results" / "visual_duplicate_verification.csv"

OUTPUT_PATH = ROOT / "results" / "visual_duplicate_group_analysis.csv"
SUMMARY_PATH = ROOT / "results" / "visual_duplicate_group_summary.txt"

SPLITS = {"train", "validation", "test"}


def resolve_manifest_path(value):
    path = Path(str(value))

    if path.is_absolute():
        return path.resolve()

    parts = list(path.parts)

    while parts and parts[0] in {"data", "Image Data base"}:
        parts.pop(0)

    return (
        ROOT / "data" / "Image Data base" / Path(*parts)
    ).resolve()


def resolve_report_path(value):
    path = Path(str(value))

    if path.is_absolute():
        return path.resolve()

    return (ROOT / path).resolve()


def main():
    manifest = pd.read_csv(MANIFEST_PATH)
    pairs = pd.read_csv(VERIFICATION_PATH)

    required_manifest = {
        "archive_path",
        "group_id",
        "split",
        "class_name",
        "plant_species",
        "condition",
    }

    required_pairs = {
        "hash",
        "path_a",
        "path_b",
        "ssim",
        "pixel_mae",
        "status",
    }

    missing_manifest = required_manifest - set(manifest.columns)
    missing_pairs = required_pairs - set(pairs.columns)

    if missing_manifest:
        raise ValueError(
            f"Manifest is missing columns: {sorted(missing_manifest)}"
        )

    if missing_pairs:
        raise ValueError(
            f"Verification report is missing columns: {sorted(missing_pairs)}"
        )

    manifest["path_key"] = manifest["archive_path"].map(
        resolve_manifest_path
    )
    manifest["split"] = (
        manifest["split"].astype(str).str.strip().str.lower()
    )

    # Do not silently choose between duplicate manifest entries.
    duplicate_paths = manifest[
        manifest["path_key"].duplicated(keep=False)
    ]

    if not duplicate_paths.empty:
        raise ValueError(
            f"{len(duplicate_paths)} manifest rows share image paths. "
            "Inspect them before proceeding."
        )

    lookup = manifest.set_index("path_key", drop=False)

    records = []
    unmatched_paths = set()
    missing_image_paths = set()

    for row in pairs.itertuples(index=False):
        path_a = resolve_report_path(row.path_a)
        path_b = resolve_report_path(row.path_b)

        if path_a not in lookup.index:
            unmatched_paths.add(str(path_a))
            continue

        if path_b not in lookup.index:
            unmatched_paths.add(str(path_b))
            continue

        if not path_a.is_file():
            missing_image_paths.add(str(path_a))

        if not path_b.is_file():
            missing_image_paths.add(str(path_b))

        a = lookup.loc[path_a]
        b = lookup.loc[path_b]

        species_conflict = (
            pd.notna(a["plant_species"])
            and pd.notna(b["plant_species"])
            and str(a["plant_species"]) != str(b["plant_species"])
        )

        condition_conflict = (
            pd.notna(a["condition"])
            and pd.notna(b["condition"])
            and str(a["condition"]) != str(b["condition"])
        )

        records.append({
            "hash": row.hash,
            "path_a": str(path_a),
            "path_b": str(path_b),
            "ssim": float(row.ssim),
            "pixel_mae": float(row.pixel_mae),
            "visual_status": row.status,
            "split_a": a["split"],
            "split_b": b["split"],
            "group_id_a": a["group_id"],
            "group_id_b": b["group_id"],
            "class_a": a["class_name"],
            "class_b": b["class_name"],
            "species_a": a["plant_species"],
            "species_b": b["plant_species"],
            "condition_a": a["condition"],
            "condition_b": b["condition"],
            "same_existing_group": (
                pd.notna(a["group_id"])
                and pd.notna(b["group_id"])
                and str(a["group_id"]).strip() != ""
                and str(a["group_id"]) == str(b["group_id"])
            ),
            "species_conflict": species_conflict,
            "condition_conflict": condition_conflict,
            "cross_split": a["split"] != b["split"],
        })

    report = pd.DataFrame(records)
    report.to_csv(OUTPUT_PATH, index=False)

    summary = [
        "VISUAL DUPLICATE / EXISTING GROUP ANALYSIS",
        "=" * 55,
        f"Verification pairs: {len(pairs):,}",
        f"Pairs matched to manifest: {len(report):,}",
        f"Unmatched image paths: {len(unmatched_paths):,}",
        f"Missing image files: {len(missing_image_paths):,}",
    ]

    if not report.empty:
        summary.extend([
            f"Pairs already in the same group: {int(report['same_existing_group'].sum()):,}",
            f"Pairs with different existing groups: {int((~report['same_existing_group']).sum()):,}",
            f"Pairs crossing splits: {int(report['cross_split'].sum()):,}",
            f"Pairs with species conflicts: {int(report['species_conflict'].sum()):,}",
            f"Pairs with condition conflicts: {int(report['condition_conflict'].sum()):,}",
        ])

        summary.append("\nVisual status counts:")
        summary.extend(
            f"  {status}: {count:,}"
            for status, count in report["visual_status"].value_counts().items()
        )

        summary.append("\nHighest-priority review cases:")
        priority = report[
            report["species_conflict"]
            | report["condition_conflict"]
            | (
                report["visual_status"].eq("near_pixel_identical")
                & (
                    report["class_a"].astype(str)
                    != report["class_b"].astype(str)
                )
            )
        ]

        if priority.empty:
            summary.append("  None found by these checks.")
        else:
            summary.append(
                priority[
                    [
                        "path_a",
                        "path_b",
                        "ssim",
                        "visual_status",
                        "split_a",
                        "split_b",
                        "class_a",
                        "class_b",
                        "same_existing_group",
                    ]
                ].head(30).to_string(index=False)
            )

    summary.extend([
        "",
        "No images, manifest rows, splits, or checkpoints were modified.",
        "Visual similarity is a screening signal, not proof of duplicate identity.",
    ])

    text = "\n".join(summary)
    SUMMARY_PATH.write_text(text, encoding="utf-8")

    print(text)
    print(f"\nDetailed report: {OUTPUT_PATH}")
    print(f"Summary report:  {SUMMARY_PATH}")

    if unmatched_paths:
        unmatched_output = ROOT / "results" / "unmatched_visual_duplicate_paths.txt"
        unmatched_output.write_text(
            "\n".join(sorted(unmatched_paths)),
            encoding="utf-8",
        )
        print(f"Unmatched paths: {unmatched_output}")

    if missing_image_paths:
        missing_output = ROOT / "results" / "missing_visual_duplicate_images.txt"
        missing_output.write_text(
            "\n".join(sorted(missing_image_paths)),
            encoding="utf-8",
        )
        print(f"Missing files: {missing_output}")


if __name__ == "__main__":
    main()
