from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit


ROOT = Path(__file__).resolve().parents[1]

MANIFEST = ROOT / "audit" / "training_manifest.csv"
COMPONENTS = ROOT / "results" / "visual_duplicate_components.csv"

OUT_MANIFEST = ROOT / "results" / "leakage_reduced_manifest.csv"
OUT_GROUPS = ROOT / "results" / "leakage_reduced_groups.csv"
OUT_SUMMARY = ROOT / "results" / "leakage_reduced_split_summary.txt"


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b:
            return
        if self.size[a] < self.size[b]:
            a, b = b, a
        self.parent[b] = a
        self.size[a] += self.size[b]


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


def main():
    for path in [MANIFEST, COMPONENTS]:
        if not path.is_file():
            raise FileNotFoundError(f"Required input not found: {path}")

    outputs = [OUT_MANIFEST, OUT_GROUPS, OUT_SUMMARY]
    existing = [str(p) for p in outputs if p.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing output files:\n"
            + "\n".join(existing)
        )

    df = pd.read_csv(MANIFEST, low_memory=False)
    comps = pd.read_csv(COMPONENTS, low_memory=False)

    required = [
        "archive_path", "group_id", "class_name",
        "plant_species", "condition", "split"
    ]
    for col in required:
        if col not in df.columns or df[col].isna().any():
            raise ValueError(f"Missing column or values in manifest: {col}")

    if not {"image_count", "paths"}.issubset(comps.columns):
        raise ValueError("Visual-component report has unexpected columns.")

    df = df.reset_index(drop=True)
    df["_resolved_path"] = df["archive_path"].map(resolve_manifest_path)

    if df["_resolved_path"].duplicated().any():
        raise ValueError("Duplicate resolved image paths in the manifest.")

    path_to_row = {
        path: i for i, path in enumerate(df["_resolved_path"])
    }

    uf = UnionFind(len(df))

    # A. Keep every existing group_id together.
    for indices in df.groupby("group_id", sort=False).indices.values():
        indices = list(indices)
        for i in indices[1:]:
            uf.union(int(indices[0]), int(i))

    # B. Keep accepted multi-image visual components together.
    # The report only contains accepted components; manual-review pairs
    # are not added here.
    merged = 0
    missing_paths = []

    for component_index, row in comps.iterrows():
        if int(row["image_count"]) <= 1:
            continue

        paths = [
            str(Path(p.strip()).expanduser().resolve())
            for p in str(row["paths"]).split(" | ")
            if p.strip()
        ]

        if len(paths) != int(row["image_count"]):
            raise ValueError(
                f"Bad path count in visual component row {component_index}."
            )

        indices = []
        for path in paths:
            idx = path_to_row.get(path)
            if idx is None:
                missing_paths.append(path)
            else:
                indices.append(idx)

        if len(indices) != len(paths):
            continue

        # Check label consistency before merging.
        sub = df.iloc[indices]
        label_cols = [
            c for c in [
                "class_name", "plant_species", "condition",
                "species_id", "condition_id"
            ] if c in sub.columns
        ]
        conflicts = [
            c for c in label_cols
            if sub[c].nunique(dropna=False) > 1
        ]
        if conflicts:
            raise ValueError(
                f"Label conflict in visual component {component_index}: "
                f"{conflicts}. No output was written."
            )

        for idx in indices[1:]:
            uf.union(indices[0], idx)
        merged += 1

    if missing_paths:
        raise ValueError(
            f"{len(missing_paths)} visual-component paths were not found "
            f"in the manifest. Example: {missing_paths[:3]}"
        )

    roots = [uf.find(i) for i in range(len(df))]
    _, inverse = np.unique(roots, return_inverse=True)
    df["leakage_group_id"] = [
        f"LG{x:07d}" for x in inverse
    ]

    # Ensure combined groups have consistent labels.
    for col in [
        "class_name", "plant_species", "condition",
        "species_id", "condition_id"
    ]:
        if col not in df.columns:
            continue
        counts = df.groupby("leakage_group_id")[col].nunique(dropna=False)
        if (counts > 1).any():
            raise ValueError(
                f"Combined groups contain conflicting {col} labels. "
                "No output was written."
            )

    groups = (
        df.groupby("leakage_group_id", sort=False)
        .agg(
            class_name=("class_name", "first"),
            image_count=("archive_path", "size"),
        )
        .reset_index()
    )

    # One row per indivisible group. Split proportions are approximate.
    labels = groups["class_name"].astype(str).to_numpy()
    group_ids = groups["leakage_group_id"].to_numpy()
    dummy_x = np.zeros((len(groups), 1))
    assignment = {}

    can_stratify = (
        len(groups) >= 10
        and groups["class_name"].value_counts().min() >= 10
    )

    try:
        if not can_stratify:
            raise ValueError("Not enough groups per class for 10-fold stratification.")

        test_splitter = StratifiedGroupKFold(
            n_splits=10, shuffle=True, random_state=42
        )
        trainval_idx, test_idx = next(
            test_splitter.split(dummy_x, labels, groups=group_ids)
        )

        trainval = groups.iloc[trainval_idx].reset_index(drop=True)
        test_groups = groups.iloc[test_idx]["leakage_group_id"]

        for gid in test_groups:
            assignment[gid] = "test"

        if trainval["class_name"].value_counts().min() < 9:
            raise ValueError("Not enough groups per class for validation stratification.")

        val_splitter = StratifiedGroupKFold(
            n_splits=9, shuffle=True, random_state=43
        )
        train_idx, val_idx = next(
            val_splitter.split(
                np.zeros((len(trainval), 1)),
                trainval["class_name"].astype(str).to_numpy(),
                groups=trainval["leakage_group_id"].to_numpy(),
            )
        )

        for gid in trainval.iloc[train_idx]["leakage_group_id"]:
            assignment[gid] = "train"
        for gid in trainval.iloc[val_idx]["leakage_group_id"]:
            assignment[gid] = "validation"

        method = "StratifiedGroupKFold"

    except ValueError as exc:
        print("Stratified assignment unavailable:", exc)
        print("Using group-aware random splitting as fallback.")

        assignment.clear()

        test_splitter = GroupShuffleSplit(
            n_splits=1, test_size=0.10, random_state=42
        )
        trainval_idx, test_idx = next(
            test_splitter.split(groups, groups=group_ids)
        )

        trainval = groups.iloc[trainval_idx].reset_index(drop=True)
        test_groups = groups.iloc[test_idx]["leakage_group_id"]

        for gid in test_groups:
            assignment[gid] = "test"

        val_splitter = GroupShuffleSplit(
            n_splits=1, test_size=1 / 9, random_state=43
        )
        train_idx, val_idx = next(
            val_splitter.split(
                trainval, groups=trainval["leakage_group_id"].to_numpy()
            )
        )

        for gid in trainval.iloc[train_idx]["leakage_group_id"]:
            assignment[gid] = "train"
        for gid in trainval.iloc[val_idx]["leakage_group_id"]:
            assignment[gid] = "validation"

        method = "GroupShuffleSplit fallback"

    if set(assignment) != set(groups["leakage_group_id"]):
        raise RuntimeError("Not every connected group received a split.")

    df["new_split"] = df["leakage_group_id"].map(assignment)

    if df["new_split"].isna().any():
        raise RuntimeError("Some rows have no new split.")

    crossing = df.groupby("leakage_group_id")["new_split"].nunique()
    if (crossing > 1).any():
        raise RuntimeError("A connected group crosses new splits.")

    # Preserve original split under an explicit name.
    df = df.rename(columns={"split": "original_split"})
    df = df.drop(columns=["_resolved_path"])

    group_report = (
        df.groupby("leakage_group_id", sort=False)
        .agg(
            image_count=("archive_path", "size"),
            new_split=("new_split", "first"),
            classes=("class_name", lambda s: " | ".join(map(str, s.unique()))),
            species=("plant_species", lambda s: " | ".join(map(str, s.unique()))),
            conditions=("condition", lambda s: " | ".join(map(str, s.unique()))),
            original_group_count=("group_id", "nunique"),
            original_splits=("original_split", lambda s: " | ".join(map(str, s.unique()))),
        )
        .reset_index()
    )

    # Write only new files.
    df.to_csv(OUT_MANIFEST, index=False)
    group_report.to_csv(OUT_GROUPS, index=False)

    counts = df["new_split"].value_counts()
    class_table = pd.crosstab(df["new_split"], df["class_name"])

    with OUT_SUMMARY.open("w", encoding="utf-8") as f:
        f.write("LEAKAGE-REDUCED SPLIT SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"Images: {len(df):,}\n")
        f.write(f"Connected groups: {df['leakage_group_id'].nunique():,}\n")
        f.write(f"Accepted visual components merged: {merged:,}\n")
        f.write(f"Split method: {method}\n\n")
        f.write("Image counts and proportions:\n")
        for name in ["train", "validation", "test"]:
            n = int(counts.get(name, 0))
            f.write(f"{name}: {n:,} ({n / len(df):.2%})\n")
        f.write("\nClass counts by split:\n")
        f.write(class_table.to_string())
        f.write("\n\nValidation:\n")
        f.write("Connected groups crossing splits: 0\n")
        f.write("Original manifest and checkpoints were not modified.\n")
        f.write(
            "Manual-review/high-similarity pairs were not automatically merged.\n"
        )

    print("\nSUCCESS: New manifest created.")
    print(f"Manifest: {OUT_MANIFEST}")
    print(f"Groups:   {OUT_GROUPS}")
    print(f"Summary:  {OUT_SUMMARY}")
    print("\nSplit counts:")
    print(counts.to_string())
    print("\nCross-split connected groups:", int((crossing > 1).sum()))


if __name__ == "__main__":
    main()
