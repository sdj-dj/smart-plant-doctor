from pathlib import Path

from ml.model import (
    ResNetAgriculturalAI,
    load_model,
    NUM_SPECIES,
    NUM_CONDITIONS,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "resnet18_agricultural_ai_step1.pth"
)


def main():
    print("Checking Smart Plant Doctor model...")

    model, device = load_model(CHECKPOINT)

    print("Device:", device)
    print("Architecture:", type(model).__name__)
    print(
        "Species outputs:",
        model.species_head[-1].out_features,
    )
    print(
        "Condition outputs:",
        model.condition_head[-1].out_features,
    )

    assert isinstance(model, ResNetAgriculturalAI)
    assert model.species_head[-1].out_features == NUM_SPECIES
    assert model.condition_head[-1].out_features == NUM_CONDITIONS

    print("\nRunning a dummy-image inference test...")

    import torch

    dummy_image = torch.randn(
        1, 3, 224, 224, device=device
    )

    with torch.inference_mode():
        species_logits, condition_logits = model(dummy_image)

    assert species_logits.shape == (1, NUM_SPECIES)
    assert condition_logits.shape == (1, NUM_CONDITIONS)

    print("Species output shape:", tuple(species_logits.shape))
    print("Condition output shape:", tuple(condition_logits.shape))
    print("\nModel smoke test passed.")


if __name__ == "__main__":
    main()