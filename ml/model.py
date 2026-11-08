"""
Smart Plant Doctor — Universal Plant-Care AI
Multi-task ResNet18 model.

Outputs:
    1. Plant species: 19 classes
    2. Plant condition: 36 classes

Note:
Predictions are for preliminary image-based screening,
not confirmed disease diagnoses.
"""

from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# ============================================================
# 1. MODEL CONFIGURATION
# ============================================================

NUM_SPECIES = 19
NUM_CONDITIONS = 36
IMAGE_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ============================================================
# 2. DEVICE SELECTION
# ============================================================

def get_device():
    """Select an available acceleration device."""

    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


# ============================================================
# 3. RESNET18 MULTI-HEAD MODEL
# ============================================================

class ResNetAgriculturalAI(nn.Module):
    """
    Shared ResNet18 feature extractor with two prediction heads.
    """

    def __init__(
        self,
        num_species=NUM_SPECIES,
        num_conditions=NUM_CONDITIONS,
        pretrained=False,
    ):
        super().__init__()

        # Build ResNet18.
        # pretrained=False avoids downloading another checkpoint.
        weights = (
            models.ResNet18_Weights.DEFAULT
            if pretrained
            else None
        )

        self.backbone = models.resnet18(weights=weights)

        # ResNet18 normally outputs 1000 ImageNet classes.
        # Replace the final classifier with an identity layer.
        self.backbone.fc = nn.Identity()

        feature_size = 512

        # Head 1: plant species.
        self.species_head = nn.Sequential(
            nn.Linear(feature_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_species),
        )

        # Head 2: plant condition.
        self.condition_head = nn.Sequential(
            nn.Linear(feature_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_conditions),
        )

    def forward(self, images):
        features = self.backbone(images)

        species_logits = self.species_head(features)
        condition_logits = self.condition_head(features)

        return species_logits, condition_logits


# ============================================================
# 4. IMAGE PREPROCESSING
# ============================================================

def get_image_transform():
    """Prepare an image for the model."""

    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ])


# ============================================================
# 5. CHECKPOINT LOADING
# ============================================================

def load_model(checkpoint_path):
    """
    Load an existing model checkpoint.

    Supports:
        - A direct PyTorch state dictionary.
        - A dictionary containing 'model_state_dict'.
        - A dictionary containing 'state_dict'.
    """

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Check the path in your project configuration."
        )

    device = get_device()

    # Construct the exact architecture expected by the checkpoint.
    model = ResNetAgriculturalAI(
        num_species=NUM_SPECIES,
        num_conditions=NUM_CONDITIONS,
        pretrained=False,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        raise ValueError("Unexpected checkpoint format.")

    # Handle checkpoints saved using DataParallel.
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }

    model.load_state_dict(state_dict, strict=True)

    model.to(device)
    model.eval()

    return model, device


# ============================================================
# 6. IMAGE PREDICTION
# ============================================================

def predict_image(
    model,
    image_path,
    device,
    species_names,
    condition_names,
    top_k=3,
):
    """
    Return the top candidate species and conditions.

    Confidence values are softmax scores. They should not be
    interpreted as calibrated probabilities of disease.
    """

    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    if len(species_names) != NUM_SPECIES:
        raise ValueError(
            f"Expected {NUM_SPECIES} species names."
        )

    if len(condition_names) != NUM_CONDITIONS:
        raise ValueError(
            f"Expected {NUM_CONDITIONS} condition names."
        )

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image_tensor = get_image_transform()(image)

    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.inference_mode():
        species_logits, condition_logits = model(image_tensor)

        species_probabilities = torch.softmax(
            species_logits.float(), dim=1
        )[0]

        condition_probabilities = torch.softmax(
            condition_logits.float(), dim=1
        )[0]

    species_k = min(top_k, len(species_names))
    condition_k = min(top_k, len(condition_names))

    species_scores, species_indices = torch.topk(
        species_probabilities, species_k
    )

    condition_scores, condition_indices = torch.topk(
        condition_probabilities, condition_k
    )

    species_results = [
        {
            "label": species_names[index.item()],
            "score": round(score.item(), 4),
        }
        for score, index in zip(
            species_scores, species_indices
        )
    ]

    condition_results = [
        {
            "label": condition_names[index.item()],
            "score": round(score.item(), 4),
        }
        for score, index in zip(
            condition_scores, condition_indices
        )
    ]

    return {
        "species_candidates": species_results,
        "condition_candidates": condition_results,
        "notice": (
            "Preliminary image-based screening only. "
            "Results are not confirmed diagnoses."
        ),
    }