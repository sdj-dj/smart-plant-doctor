from pathlib import Path
from collections import Counter
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = ROOT / "audit" / "training_manifest.csv"
PAIR_PATH = ROOT / "results" / "visual_duplicate_group_analysis.csv"

COMPONENT_PATH = ROOT / "results" / "visual_duplicate_components.csv"
SUMMARY_PATH = ROOT / "results" / "visual_duplicate_component_summary.txt"

# Only automatically connect the strongest category from our current audit.
# The other categories remain available for manual review.
ACCEPTED_STATUS = "near_pixel_identical"


def resolve_manifest_path(value):
    path = Path(str(value))

    if path.is_absolute():
        return str(path.resolve())

    parts = list(path.parts)

    while parts and parts[0] in {"data", "Image Data base"}:
        parts.pop(0)

    return str(
        (ROOT / "data" / "Image Data base" / Path(*parts)).resolve()
    )


class UnionFind:
    def __init__(self):
        self.parent = {}
        self.size = {}

    def add(self, item):
        if item not in self.parent:
            self.parent[item] = item
            self.size[item] = 1

    def find(self, item):
        parent = self.parent[item]

        if parent != item:
            self.parent[item] = self.find(parent)

        return self.parent[item]

    def union(self, a, b):
        self.add(a)
        self.add(b)

        root_a = self.find(a)
        root_b = self.find(b)

        if root_a == root_b:
            return

        # Union by size.
        if self.size[root_a] < self.size[root_b]:
            root_a, root_b = root_b, root_a

        self.parent[root_b] = root_a
        self.size[root_a] += self.size[root_b]


def main():
    manifest = pd.read_csv(MANIFEST_PATH)
    pairs = pd.read_csv(PAIR_PATH)

    required_manifest = {
        "archive_path",
        "split",
        "group_id",
        "class_name",
        "plant_species",
        "condition",
    }

    required_pairs = {
        "path_a",
        "path_b",
        "visual_status",
        "ssim",
        "pixel_mae",
        "class_a",
        "class_b",
        "split_a",
        "split_b",
    }

    if missing := required_manifest - set(manifest.columns):
        raise ValueError(f"Missing manifest columns: {sorted(missing)}")

    if missing := required_pairs - set(pairs.columns):
        raise ValueError(f"Missing pair columns: {sorted(missing)}")

    # Reuse the same path-resolution convention as the previous audit.
    manifest["path_key"] = manifest["archive_path"].map(
        resolve_manifest_path
    )

    if manifest["path_key"].duplicated().any():
        raise ValueError("Duplicate image paths exist in the manifest.")

    lookup = manifest.set_index("path_key", drop=False)

    accepted = pairs[
        pairs["visual_status"] == ACCEPTED_STATUS
    ].copy()

    uf = UnionFind()

    # Include every manifest image so singleton components are retained.
    for path in manifest["path_key"]:
        uf.add(path)

    for row in accepted.itertuples(index=False):
        a = str(row.path_a)
        b = str(row.path_b)

        if a not in lookup.index or b not in lookup.index:
            raise ValueError(f"Pair path not found in manifest: {a} | {b}")

        uf.union(a, b)

    members = {}

    for path in manifest["path_key"]:
        root = uf.find(path)
        members.setdefault(root, []).append(path)

    rows = []

    for component_number, (root, paths) in enumerate(
        sorted(members.items(), key=lambda item: min(item[1])),
        start=1,
    ):
        group = lookup.loc[paths]

        splits = sorted(group["split"].astype(str).unique())
        classes = sorted(group["class_name"].dropna().astype(str).unique())
        species = sorted(group["plant_species"].dropna().astype(str).unique())
        conditions = sorted(group["condition"].dropna().astype(str).unique())
        existing_groups = sorted(
            group["group_id"].dropna().astype(str).unique()
        )

        rows.append({
            "candidate_component_id": f"visual_component_{component_number:07d}",
            "image_count": len(paths),
            "split_count": len(splits),
            "splits": " | ".join(splits),
            "class_count": len(classes),
            "classes": " | ".join(classes),
            "species_count": len(species),
            "species": " | ".join(species),
            "condition_count": len(conditions),
            "conditions": " | ".join(conditions),
            "existing_group_count": len(existing_groups),
            "paths": " | ".join(paths),
        })

    components = pd.DataFrame(rows)
    components.to_csv(COMPONENT_PATH, index=False)

    sizes = components["image_count"]
    crossing = components[components["split_count"] > 1]
    multi_image = components[components["image_count"] > 1]

    summary = [
        "VISUAL DUPLICATE CONNECTED-COMPONENT AUDIT",
        "=" * 58,
        f"Manifest images: {len(manifest):,}",
        f"All candidate components: {len(components):,}",
        f"Accepted pair edges: {len(accepted):,}",
        f"Components containing multiple images: {len(multi_image):,}",
        f"Components spanning multiple splits: {len(crossing):,}",
        f"Largest component size: {int(sizes.max()):,}",
        f"Components with conflicting classes: {int((components['class_count'] > 1).sum()):,}",
        f"Components with conflicting species: {int((components['species_count'] > 1).sum()):,}",
        f"Components with conflicting conditions: {int((components['condition_count'] > 1).sum()):,}",
        "",
        "Component size distribution:",
        sizes.describe().to_string(),
        "",
        "Largest 20 components:",
        components.sort_values(
            ["image_count", "candidate_component_id"],
            ascending=[False, True],
        ).head(20)[
            [
                "candidate_component_id",
                "image_count",
                "split_count",
                "splits",
                "class_count",
                "classes",
                "condition_count",
                "conditions",
            ]
        ].to_string(index=False),
        "",
        "No manifest, dataset image, split, or checkpoint was modified.",
        "Connected components are candidates for review, not automatic proof",
        "that all linked images depict the same original photograph.",
    ]

    text = "\n".join(summary)
    SUMMARY_PATH.write_text(text, encoding="utf-8")

    print(text)
    print(f"\nComponent report: {COMPONENT_PATH}")
    print(f"Summary report:   {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
