"""Explanations for environmental measurements in Smart Plant Doctor.

These descriptions explain what a measurement represents. They do not
define universal thresholds or diagnose plant diseases.
"""

ENVIRONMENTAL_KNOWLEDGE = {
    "temperature_c": {
        "name": "Temperature",
        "unit": "°C",
        "meaning": (
            "Temperature describes how warm or cool the measured "
            "environment was at the recorded time."
        ),
        "interpretation": (
            "Suitability depends on plant species, growth stage, "
            "measurement location, and the duration of exposure."
        ),
        "limitation": (
            "A single reading does not establish long-term temperature "
            "conditions or explain the cause of a leaf symptom."
        ),
    },
    "humidity_pct": {
        "name": "Relative humidity",
        "unit": "%",
        "meaning": (
            "Relative humidity describes how much water vapour is in "
            "the air relative to the maximum possible at that temperature."
        ),
        "interpretation": (
            "Humidity can provide environmental context when reviewed "
            "alongside temperature, plant species, and observation history."
        ),
        "limitation": (
            "Air humidity is not a direct measurement of leaf moisture "
            "or proof that a plant has a disease."
        ),
    },
    "soil_moisture_pct": {
        "name": "Soil moisture",
        "unit": "%",
        "meaning": (
            "The sensor reports a soil-moisture value according to "
            "its measurement method and calibration."
        ),
        "interpretation": (
            "Interpret the value using the specific sensor, growing "
            "medium, sensor placement, and plant requirements."
        ),
        "limitation": (
            "A displayed percentage must not automatically be treated "
            "as a universal measure of actual soil water content."
        ),
    },
    "light_lux": {
        "name": "Light intensity",
        "unit": "lux",
        "meaning": (
            "Lux measures illumination weighted to human visual "
            "sensitivity."
        ),
        "interpretation": (
            "Readings can help compare illumination at a consistent "
            "location and measurement setup."
        ),
        "limitation": (
            "Lux is not equivalent to photosynthetically active radiation "
            "(PAR) or a direct measurement of light usable by plants."
        ),
    },
    "ph": {
        "name": "pH",
        "unit": "",
        "meaning": (
            "pH describes acidity or alkalinity on a logarithmic scale."
        ),
        "interpretation": (
            "Its relevance depends on the plant species, growing medium, "
            "measurement method, and calibration."
        ),
        "limitation": (
            "A pH value alone does not establish nutrient availability "
            "or diagnose a plant disorder."
        ),
    },
}


def explain_measurement(field):
    """Return a measurement explanation, or None for an unknown field."""
    item = ENVIRONMENTAL_KNOWLEDGE.get(field)

    if item is None:
        return None

    return {
        "field": field,
        **item,
    }
