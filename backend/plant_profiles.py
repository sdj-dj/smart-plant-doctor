"""Plant-specific monitoring profiles and cautious care guidance.

Thresholds are illustrative starting points, not universal diagnoses.
Calibrate them for the plant variety, growing stage, and environment.
"""

PLANT_PROFILES = {
    "tomato": {
        "display_name": "Tomato",
        "soil_moisture": {
            "unit": "%",
            "low": 30,
            "high": 75,
            "note": "Calibrate the sensor against the actual growing medium.",
        },
        "temperature_c": {"low": 18, "high": 30},
        "humidity_pct": {"low": 40, "high": 80},
        "care_checks": [
            "Check soil moisture before watering.",
            "Ensure the pot or bed drains freely.",
            "Inspect both sides of leaves for changing symptoms.",
            "Record whether symptoms affect older or newer leaves.",
        ],
    },
    "tea": {
        "display_name": "Tea",
        "soil_moisture": {
            "unit": "%",
            "low": 35,
            "high": 75,
            "note": "These provisional values require local calibration.",
        },
        "temperature_c": {"low": 18, "high": 30},
        "humidity_pct": {"low": 50, "high": 90},
        "care_checks": [
            "Check soil moisture and drainage.",
            "Inspect affected leaves and nearby healthy leaves.",
            "Record whether spots are spreading over time.",
            "Compare symptoms with reliable tea-specific plant-health guidance.",
        ],
    },
    "pepper": {
        "display_name": "Bell pepper",
        "soil_moisture": {
            "unit": "%",
            "low": 30,
            "high": 75,
            "note": "Calibrate the sensor for your growing medium.",
        },
        "temperature_c": {"low": 18, "high": 30},
        "humidity_pct": {"low": 40, "high": 80},
        "care_checks": [
            "Check soil moisture before watering.",
            "Ensure drainage is adequate.",
            "Inspect new growth and both sides of leaves.",
            "Record changes in symptoms and growing conditions.",
        ],
    },
}


def get_plant_profile(plant: str) -> dict:
    """Return a known plant profile or raise a clear error."""
    key = plant.strip().lower()

    if key not in PLANT_PROFILES:
        supported = ", ".join(sorted(PLANT_PROFILES))
        raise ValueError(
            f"Unknown plant '{plant}'. Supported profiles: {supported}"
        )

    return PLANT_PROFILES[key]
