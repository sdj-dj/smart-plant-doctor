from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


PROJECT_DIR = Path(__file__).resolve().parent.parent

# Use the leakage-reduced manifest.
MANIFEST_PATH = (
    PROJECT_DIR / "results" / "leakage_reduced_manifest.csv"
)

IMAGE_ROOT = PROJECT_DIR / "data" / "Image Data base"

SPLIT_COLUMN = "new_split"

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


class PlantDataset(Dataset):
    def __init__(self, split="train", transform=None):
        if split not in {"train", "validation", "test"}:
            raise ValueError(
                "split must be train, validation, or test"
            )

        if not MANIFEST_PATH.is_file():
            raise FileNotFoundError(
                f"Leakage-reduced manifest not found: {MANIFEST_PATH}"
            )

        manifest = pd.read_csv(MANIFEST_PATH, low_memory=False)

        required_columns = {
            "archive_path",
            "species_id",
            "condition_id",
            "species_mask",
            "condition_mask",
            SPLIT_COLUMN,
            "leakage_group_id",
        }

        missing = required_columns - set(manifest.columns)

        if missing:
            raise ValueError(
                f"Manifest is missing required columns: {sorted(missing)}"
            )

        self.data = (
            manifest.loc[manifest[SPLIT_COLUMN] == split]
            .reset_index(drop=True)
        )

        self.transform = transform or IMAGE_TRANSFORM

        if self.data.empty:
            raise ValueError(f"No images found for split: {split}")

        # Validate labels and masks before training.
        for label_column, mask_column, class_count in [
            ("species_id", "species_mask", 19),
            ("condition_id", "condition_mask", 36),
        ]:
            labels = pd.to_numeric(
                self.data[label_column], errors="coerce"
            )
            masks = self.data[mask_column].astype("boolean")

            if masks.isna().any():
                raise ValueError(
                    f"Missing mask values in {mask_column}"
                )

            valid_mask = masks.fillna(False).astype(bool)

            invalid = valid_mask & (
                labels.isna()
                | (labels < 0)
                | (labels >= class_count)
                | (labels % 1 != 0)
            )

            if invalid.any():
                raise ValueError(
                    f"{invalid.sum()} invalid labels marked valid in "
                    f"{label_column}"
                )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        relative_path = Path(str(row["archive_path"]))

        # Handle paths with or without dataset-folder prefixes.
        parts = list(relative_path.parts)

        while parts and parts[0] in {"data", "Image Data base"}:
            parts.pop(0)

        image_path = IMAGE_ROOT / Path(*parts)

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image not found: {image_path}\n"
                f"Manifest path: {row['archive_path']}"
            )

        with Image.open(image_path) as image:
            image = image.convert("RGB")

            if self.transform:
                image = self.transform(image)

        return {
            "image": image,
            "species": int(row["species_id"]),
            "condition": int(row["condition_id"]),
            "species_mask": bool(row["species_mask"]),
            "condition_mask": bool(row["condition_mask"]),
        }


if __name__ == "__main__":
    for split in ("train", "validation", "test"):
        dataset = PlantDataset(split=split)

        print(f"{split.capitalize()} images: {len(dataset):,}")

    train_dataset = PlantDataset(split="train")
    sample = train_dataset[0]

    print("Image tensor shape:", sample["image"].shape)
    print("Species label:", sample["species"])
    print("Condition label:", sample["condition"])