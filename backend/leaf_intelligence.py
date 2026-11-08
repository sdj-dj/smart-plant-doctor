"""Leaf Intelligence Engine for Smart Plant Doctor.

Combines stored observations, AI predictions, environmental measurements,
and care history into an evidence-based report.

This module does not retrain or modify the existing AI model.
"""

import random
from datetime import datetime, timedelta, timezone
from backend.knowledge.environmental_knowledge import explain_measurement
from backend.knowledge_base import retrieve_relevant_knowledge

from backend.db import (
    get_connection,
    get_leaf_history,
    get_observation_predictions,
    get_sensor_history,
    get_care_history,
    save_health_summary,
)


REPORT_STYLES = {
    "simple": "simple",
    "detailed": "detailed",
    "scientific": "scientific",
    "conversational": "conversational",
    "research": "research",
    "concise": "concise",
}


def _parse_datetime(value):
    """Parse stored ISO-style timestamps without inventing missing dates."""
    if not value:
        return None

    try:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if result.tzinfo is None:
            result = result.replace(tzinfo=timezone.utc)
        return result
    except (TypeError, ValueError):
        return None


def _choose(phrases, variation=True, seed=None):
    """Select compatible wording while keeping the underlying fact unchanged."""
    if not phrases:
        return ""

    if not variation:
        return phrases[0]

    if seed is None:
        return random.choice(phrases)

    return random.Random(str(seed)).choice(phrases)


def _format_score(score):
    """Format a model score without implying diagnostic certainty."""
    if score is None:
        return "not available"

    try:
        value = float(score)
        if not 0.0 <= value <= 1.0:
            return "not available"
        return f"{value * 100:.1f}%"
    except (TypeError, ValueError):
        return "not available"


def _latest_prediction(observations):
    """Return the newest saved prediction attached to the leaf history."""
    predictions = []

    for observation in observations:
        for prediction in observation.get("predictions", []):
            item = dict(prediction)
            item["observation_id"] = observation["observation_id"]
            item["observed_at"] = observation.get("observed_at")
            predictions.append(item)

    predictions.sort(
        key=lambda item: (
            _parse_datetime(item.get("created_at"))
            or _parse_datetime(item.get("observed_at"))
            or datetime.min.replace(tzinfo=timezone.utc)
        )
    )

    return predictions[-1] if predictions else None


def _describe_current_condition(prediction, style, variation, seed):
    """Explain the saved prediction in the requested reporting style."""
    if not prediction:
        messages = {
            "simple": [
                "No AI result has been saved for this leaf yet.",
                "The leaf does not have a recorded AI prediction.",
            ],
            "conversational": [
                "I couldn't find a saved AI result for this leaf yet.",
                "There isn't a recorded AI assessment to discuss at the moment.",
            ],
            "concise": [
                "AI assessment: unavailable.",
            ],
            "scientific": [
                "No prediction record is available for analysis. "
                "The current condition therefore remains undetermined.",
            ],
            "research": [
                "The available record contains no saved prediction. "
                "This is a data gap, not evidence for or against any condition.",
            ],
            "detailed": [
                "No saved AI prediction was found in the available leaf history. "
                "A current condition cannot be established from the stored records.",
            ],
        }
        return _choose(
            messages.get(style, messages["detailed"]),
            variation,
            seed,
        )

    condition = prediction.get("predicted_condition") or "Not available"
    score = _format_score(prediction.get("condition_score"))
    species = prediction.get("predicted_species")
    species_score = _format_score(prediction.get("species_score"))

    species_text = (
        f" The predicted species is '{species}' "
        f"(model score: {species_score})."
        if species
        else ""
    )

    messages = {
        "simple": [
            f"The AI's top condition prediction is '{condition}'. "
            f"Model score: {score}. This is a prediction, not a confirmed diagnosis.",
            f"The model currently favours '{condition}' with a score of {score}. "
            "A closer check of the leaf is still important.",
        ],
        "detailed": [
            f"The latest saved assessment ranks '{condition}' as the leading "
            f"condition prediction, with a model score of {score}. "
            "This score describes the model output; it is not necessarily a "
            "calibrated probability and does not independently confirm a diagnosis."
            f"{species_text}",
            f"The stored AI result identifies '{condition}' as its top condition "
            f"candidate (model score: {score}). Interpret this alongside the image, "
            "observation notes, and any available environmental measurements. "
            "The score is not a confirmed probability."
            f"{species_text}",
        ],
        "conversational": [
            f"The model's top pick is '{condition}', with a score of {score}. "
            "Think of this as a useful clue rather than a final verdict."
            f"{species_text}",
            f"The latest saved result leans toward '{condition}' "
            f"(model score: {score}). It is worth checking the leaf's visible "
            "symptoms before drawing a conclusion."
            f"{species_text}",
        ],
        "concise": [
            f"Top condition: {condition}. Model score: {score}. "
            "Not a confirmed diagnosis.",
        ],
        "scientific": [
            f"The highest-ranked stored condition class is '{condition}' "
            f"(model output score: {score}). This score must not be interpreted "
            "as a calibrated posterior probability without calibration evidence. "
            "The prediction alone does not establish a definitive diagnosis."
            f"{species_text}",
            f"The recorded inference assigns the leading condition label "
            f"'{condition}' a score of {score}. Diagnostic interpretation remains "
            "limited by model calibration, image quality, and the available "
            "observation context."
            f"{species_text}",
        ],
        "research": [
            f"Evidence available: the saved model output ranks '{condition}' "
            f"first, with a score of {score}.{species_text} "
            "This is model-generated evidence, not independent confirmation. "
            "A stronger assessment would compare the image with documented "
            "symptoms, repeated observations, and relevant environmental records.",
            f"The database records '{condition}' as the leading condition "
            f"candidate (score: {score}). The score's calibration is unknown here, "
            "so it should not be treated as a diagnostic probability. "
            "Independent visual or expert verification may be needed."
            f"{species_text}",
        ],
    }

    options = messages.get(style, messages["detailed"])
    return _choose(options, variation, seed)


def _summarize_history(observations, style, variation, seed):
    count = len(observations)

    if count == 0:
        return "No observations have been recorded for this leaf."

    predicted = [
        observation
        for observation in observations
        if observation.get("predictions")
    ]

    if style == "simple":
        return (
            f"This leaf has {count} recorded observation(s), "
            f"with saved AI predictions for {len(predicted)} observation(s)."
        )

    if style == "concise":
        return f"Observations: {count}. With saved predictions: {len(predicted)}."

    if count == 1:
        return _choose(
            [
                "One observation is available. A time-based trend cannot yet "
                "be established.",
                "The history contains a single observation, which is insufficient "
                "to determine how the leaf has changed over time.",
            ],
            variation,
            seed,
        )

    return _choose(
        [
            f"The database contains {count} observations for this leaf, "
            f"and {len(predicted)} of them have saved predictions. These records "
            "can be reviewed chronologically, but a reliable trend requires "
            "comparable observations and predictions.",
            f"There are {count} recorded observations, including "
            f"{len(predicted)} with saved AI predictions. Their sequence provides "
            "historical context; it does not automatically prove improvement "
            "or deterioration.",
        ],
        variation,
        seed,
    )


def _get_environmental_history(plant_id, limit):
    """Get recent sensor records from stations linked to the plant."""
    with get_connection() as connection:
        stations = connection.execute(
            """
            SELECT station_id, name, location, device_type, calibration_notes
            FROM sensor_stations
            WHERE plant_id = ?
            ORDER BY station_id
            """,
            (plant_id,),
        ).fetchall()

    result = []

    for station in stations:
        station_data = dict(station)
        station_data["readings"] = get_sensor_history(
            station_data["station_id"],
            limit=limit,
        )
        result.append(station_data)

    return result


def _summarize_environment(stations, style, variation, seed):
    fields = [
        ("temperature_c", "temperature", "°C"),
        ("humidity_pct", "relative humidity", "%"),
        ("soil_moisture_pct", "soil moisture", "%"),
        ("light_lux", "light intensity", "lux"),
        ("ph", "pH", ""),
    ]

    latest_by_station = []
    all_readings = []

    for station in stations:
        readings = list(station.get("readings", []))
        all_readings.extend(readings)

        if not readings:
            continue

        latest = max(
            readings,
            key=lambda row: (
                _parse_datetime(row.get("recorded_at"))
                or datetime.min.replace(tzinfo=timezone.utc)
            ),
        )

        latest_data = dict(latest)
        latest_data["station_id"] = station.get("station_id")
        latest_data["station_name"] = station.get("name")
        latest_data["plant_id"] = station.get("plant_id")
        latest_data["quality_status"] = latest_data.get(
            "quality_status", "unknown"
        )
        latest_data["source"] = latest_data.get("source", "unknown")

        available = []
        missing = []

        for key, label, unit in fields:
            value = latest_data.get(key)

            if value is None:
                missing.append(label)
            else:
                available.append({
                    "field": key,
                    "label": label,
                    "value": value,
                    "unit": unit,
                })

        latest_data["available_fields"] = available
        latest_data["missing_fields"] = missing

        # Explain only measurements that actually have recorded values.
        latest_data["measurement_explanations"] = [
            explain_measurement(item["field"])
            for item in available
        ]

        latest_by_station.append(latest_data)

    if not latest_by_station:
        return (
            "No environmental readings were found from sensor stations "
            "associated with this plant. Environmental conditions are "
            "unknown; missing readings have not been estimated."
        ), {
            "station_count": len(stations),
            "reading_count": len(all_readings),
            "latest_reading": None,
            "latest_by_station": [],
            "available_fields": [],
            "missing_fields": [label for _, label, _ in fields],
        }

    latest_by_station.sort(
        key=lambda row: (
            _parse_datetime(row.get("recorded_at"))
            or datetime.min.replace(tzinfo=timezone.utc)
        )
    )

    latest = latest_by_station[-1]
    sections = []

    for reading in latest_by_station:
        station_name = reading.get("station_name") or (
            f"Station {reading.get('station_id')}"
        )
        status = reading.get("quality_status", "unknown")
        source = reading.get("source", "unknown")
        timestamp = reading.get("recorded_at") or "timestamp unavailable"

        measurements = reading["available_fields"]
        if measurements:
            measurement_text = "; ".join(
                f"{item['label']}: {item['value']} {item['unit']}".strip()
                for item in measurements
            )
        else:
            measurement_text = "no measurement values recorded"

        missing = reading["missing_fields"]
        missing_text = (
            " Missing fields: " + ", ".join(missing) + "."
            if missing else ""
        )

        explanation_parts = []

        for explanation in reading["measurement_explanations"]:
            explanation_parts.append(
                f"{explanation['name']}: "
                f"{explanation['meaning'].strip()} "
                f"{explanation['interpretation'].strip()} "
                f"Limitation: {explanation['limitation'].strip()}"
            )

        explanation_text = (
            " Measurement explanations: "
            + " ".join(explanation_parts)
            if explanation_parts
            else ""
        )

        sections.append(
            f"{station_name} — latest reading at {timestamp}. "
            f"Source: {source}; quality status: {status}. "
            f"Measurements: {measurement_text}.{missing_text}"
            f"{explanation_text}"
        )

    details = " ".join(sections)

    if style == "simple":
        text = (
            "Here are the latest recorded environmental measurements. "
            + details
            + " These readings provide context, not a confirmed diagnosis."
        )
    elif style == "concise":
        text = details
    elif style == "scientific":
        text = (
            "Environmental evidence, reported separately by sensor station: "
            + details
            + " Quality labels describe the stored record and do not "
            "independently establish sensor accuracy. These measurements "
            "do not establish the cause of a leaf condition."
        )
    elif style == "research":
        text = (
            "Available environmental records: "
            + details
            + " Interpret them alongside observation timestamps, sensor "
            "calibration, plant species, and other evidence. Temporal "
            "association alone does not demonstrate causation."
        )
    elif style == "conversational":
        text = (
            "Here's what the associated sensor station(s) recorded: "
            + details
            + " Think of these values as useful context, not a diagnosis."
        )
    else:
        text = _choose(
            [
                "The latest available readings from each associated sensor "
                "station are listed separately: " + details
                + " Missing values remain unknown, and environmental "
                "measurements alone cannot diagnose a leaf condition.",
                "Environmental context from the associated stations: "
                + details
                + " The quality status should be considered before relying "
                "on a measurement.",
            ],
            variation,
            seed,
        )

    return text, {
        "station_count": len(stations),
        "reading_count": len(all_readings),
        "latest_reading": latest,
        "latest_by_station": latest_by_station,
        "available_fields": latest["available_fields"],
        "missing_fields": latest["missing_fields"],
    }


def _summarize_care(actions, style):
    if not actions:
        return (
            "No care actions were found for this plant or leaf. "
            "This means no actions are recorded in the database, not that "
            "no care has taken place."
        )

    recent = actions[:5]
    descriptions = [
        f"{item.get('action_type', 'Care action')} "
        f"({item.get('action_at', 'time unavailable')})"
        for item in recent
    ]

    if style == "concise":
        return f"Recorded care actions: {len(actions)}."

    return (
        f"There are {len(actions)} recorded care action(s). "
        f"The most recent entries are: {'; '.join(descriptions)}. "
        "Recorded actions may be compared with later observations, but temporal "
        "sequence alone does not prove that an action caused a change."
    )



def _generate_recommendations(
    prediction,
    observations,
    environment,
    actions,
    style,
    variation=True,
    seed=None,
):
    """Create cautious next-step suggestions from recorded evidence."""
    recommendations = []

    if prediction:
        condition = prediction.get("predicted_condition") or "the predicted condition"

        wording = {
            "simple": [
                f"Check the visible symptoms associated with the model's '{condition}' prediction.",
                f"Look closely at the leaf and compare its appearance with the '{condition}' prediction.",
            ],
            "detailed": [
                f"Review the visible symptoms alongside the '{condition}' prediction. "
                "Do not rely on the model label alone to decide what treatment is needed.",
                f"Compare the image with documented symptoms relevant to '{condition}'. "
                "Seek knowledgeable plant-health advice if symptoms are unclear or worsening.",
            ],
            "scientific": [
                f"Validate the '{condition}' model output against visible symptoms and, "
                "where possible, an independent assessment. Do not interpret the model score "
                "as a calibrated diagnostic probability.",
            ],
            "conversational": [
                f"Use '{condition}' as a clue, then take a careful look at what the leaf actually shows.",
                f"Before choosing a treatment, compare the '{condition}' prediction with the visible signs.",
            ],
            "research": [
                f"Independently examine whether the recorded image supports the '{condition}' "
                "candidate. Document relevant symptoms and consider alternative explanations "
                "before drawing a conclusion.",
            ],
            "concise": [
                f"Verify the '{condition}' prediction against visible symptoms before treatment.",
            ],
        }

        recommendations.append({
            "category": "prediction_verification",
            "priority": "normal",
            "text": _choose(wording.get(style, wording["detailed"]), variation, seed),
            "basis": "A saved AI prediction is available.",
        })
    else:
        recommendations.append({
            "category": "missing_prediction",
            "priority": "normal",
            "text": (
                "Capture a clear, well-lit image of the leaf and run the existing "
                "AI prediction workflow before interpreting its condition."
            ),
            "basis": "No saved AI prediction is available.",
        })

    if len(observations) < 2:
        recommendations.append({
            "category": "follow_up_observation",
            "priority": "normal",
            "text": (
                "Record another observation of the same identified leaf later, "
                "using a similar angle and lighting. Compare the records before "
                "concluding that the leaf has improved or deteriorated."
            ),
            "basis": f"Only {len(observations)} observation(s) are recorded.",
        })

    missing_fields = environment.get("missing_fields", [])
    if environment.get("reading_count", 0) == 0:
        recommendations.append({
            "category": "environmental_data",
            "priority": "normal",
            "text": (
                "If environmental monitoring is available, record the growing area's "
                "temperature, humidity, soil moisture, light intensity in lux, and pH "
                "where relevant. Leave unavailable measurements blank."
            ),
            "basis": "No sensor readings are available for this plant.",
        })
    elif missing_fields:
        recommendations.append({
            "category": "incomplete_environmental_data",
            "priority": "normal",
            "text": (
                "Some measurements are missing from the latest recorded reading: "
                + ", ".join(missing_fields)
                + ". Record them when suitable sensors or measurements are available; "
                  "do not infer missing values."
            ),
            "basis": "The latest reading contains missing measurement fields.",
        })

    if not actions:
        recommendations.append({
            "category": "care_recordkeeping",
            "priority": "low",
            "text": (
                "Record any care already provided, including its date and relevant "
                "details. If no care has been provided, do not create a care entry "
                "until an action actually occurs."
            ),
            "basis": "No care actions are recorded in the database.",
        })
    else:
        recommendations.append({
            "category": "care_follow_up",
            "priority": "low",
            "text": (
                "Continue recording care actions and subsequent observations. "
                "A change after an action does not, by itself, establish that "
                "the action caused the change."
            ),
            "basis": f"{len(actions)} care action(s) are recorded.",
        })

    if style == "concise":
        recommendations = recommendations[:3]

    return recommendations



def _summarize_knowledge(knowledge, style):
    """Render retrieved knowledge without treating it as a diagnosis."""
    plant = knowledge.get("plant")
    condition = knowledge.get("condition")
    symptoms = knowledge.get("symptoms", [])
    guidance = knowledge.get("care_guidance", [])

    lines = [
        "Knowledge-base reference only; this section does not confirm "
        "the model prediction."
    ]

    if plant:
        lines.append(
            f"Plant profile: {plant.get('display_name', 'Unknown')} "
            f"(status: {plant.get('verification_status', 'unknown')})."
        )

        description = plant.get("description")
        if description:
            lines.append(f"Plant notes: {description}")

        if plant.get("environmental_guidance_json"):
            lines.append(
                "Environmental profile is available in the database. "
                "Check its verification status before relying on it."
            )
    else:
        lines.append("No matching plant profile was found.")

    if condition:
        lines.append(
            f"Condition reference: {condition.get('display_name', 'Unknown')} "
            f"(status: {condition.get('verification_status', 'unknown')})."
        )

        description = condition.get("description")
        if description:
            lines.append(f"Condition notes: {description}")
    else:
        lines.append("No matching condition record was found.")

    if symptoms:
        lines.append("Stored symptom references:")
        for item in symptoms:
            status = item.get("verification_status", "unknown")
            description = item.get("description") or "No description supplied."
            lines.append(
                f"- {item.get('symptom_name', 'Unnamed symptom')}: "
                f"{description} [status: {status}]"
            )
            if item.get("source_title"):
                lines.append(
                    f"  Source: {item['source_title']}"
                    f"{' — ' + item['source_url'] if item.get('source_url') else ''}"
                    f" [source status: {item.get('source_status', 'unknown')}]"
                )
    else:
        lines.append("No matching symptom entries have been added yet.")

    if guidance:
        lines.append("Stored care guidance:")
        for item in guidance:
            lines.append(
                f"- {item.get('title', 'Care guidance')}: "
                f"{item.get('instructions', '')} "
                f"[evidence level: {item.get('evidence_level', 'unknown')}]"
            )
            if item.get("caution"):
                lines.append(f"  Caution: {item['caution']}")
            if item.get("source_title"):
                lines.append(
                    f"  Source: {item['source_title']}"
                    f"{' — ' + item['source_url'] if item.get('source_url') else ''}"
                    f" [source status: {item.get('source_status', 'unknown')}]"
                )
    else:
        lines.append("No matching care-guidance entries have been added yet.")

    lines.append(knowledge.get(
        "notice",
        "Verify references before acting on them."
    ))

    return "\n".join(lines)


def generate_leaf_report(
    leaf_id,
    style="detailed",
    variation=True,
    sensor_limit=100,
    care_limit=100,
    save=False,
    generator_version="leaf-intelligence-v1",
    seed=None,
):
    """Generate a report from stored evidence for one registered leaf.

    Args:
        leaf_id: Existing database leaf ID.
        style: simple, detailed, scientific, conversational, research, concise.
        variation: Select compatible alternative wording when True.
        sensor_limit: Maximum recent readings per associated sensor station.
        care_limit: Maximum care actions to retrieve.
        save: Save the report and evidence to the database.
        generator_version: Report generator identifier.
        seed: Optional repeatable variation seed for testing.

    Returns:
        A dictionary containing report text, evidence, and limitations.
    """
    if isinstance(leaf_id, bool) or not isinstance(leaf_id, int) or leaf_id < 1:
        raise ValueError("leaf_id must be a positive integer.")

    if style not in REPORT_STYLES:
        raise ValueError(
            f"Unknown style '{style}'. Choose from: "
            + ", ".join(REPORT_STYLES)
        )

    if (
        isinstance(sensor_limit, bool)
        or not isinstance(sensor_limit, int)
        or not 1 <= sensor_limit <= 1000
    ):
        raise ValueError("sensor_limit must be between 1 and 1000.")

    if (
        isinstance(care_limit, bool)
        or not isinstance(care_limit, int)
        or not 1 <= care_limit <= 1000
    ):
        raise ValueError("care_limit must be between 1 and 1000.")

    history = get_leaf_history(leaf_id)

    if not history or not history.get("leaf"):
        raise ValueError(f"Leaf {leaf_id} was not found.")

    leaf = history["leaf"]
    observations = history.get("observations", [])

    # Attach stored predictions to their corresponding observations.
    for observation in observations:
        observation["predictions"] = get_observation_predictions(
            observation["observation_id"]
        )

    observations.sort(
        key=lambda item: (
            _parse_datetime(item.get("observed_at"))
            or datetime.min.replace(tzinfo=timezone.utc),
            item["observation_id"],
        )
    )

    prediction = _latest_prediction(observations)

    plant_name_for_knowledge = (
        leaf.get("species")
        or (prediction or {}).get("predicted_species")
        or leaf.get("plant_name")
        or ""
    )
    condition_name_for_knowledge = (
        (prediction or {}).get("predicted_condition") or ""
    )

    knowledge = retrieve_relevant_knowledge(
        plant_name=plant_name_for_knowledge,
        condition_name=condition_name_for_knowledge,
        limit=20,
    )
    knowledge_text = _summarize_knowledge(knowledge, style)

    stations = _get_environmental_history(leaf["plant_id"], sensor_limit)
    environment_text, environment_evidence = _summarize_environment(
        stations, style, variation, seed
    )

    actions = get_care_history(
        plant_id=leaf["plant_id"],
        leaf_id=leaf_id,
        limit=care_limit,
    )

    plant_actions = get_care_history(
        plant_id=leaf["plant_id"],
        limit=care_limit,
    )

    # Include both leaf-specific and plant-wide care entries without duplicates.
    action_map = {
        item["action_id"]: item
        for item in plant_actions + actions
    }
    all_actions = sorted(
        action_map.values(),
        key=lambda item: (
            _parse_datetime(item.get("action_at"))
            or datetime.min.replace(tzinfo=timezone.utc)
        ),
        reverse=True,
    )

    condition_text = _describe_current_condition(
        prediction, style, variation, seed
    )
    history_text = _summarize_history(
        observations, style, variation, seed
    )
    care_text = _summarize_care(all_actions, style)

    recommendations = _generate_recommendations(
        prediction=prediction,
        observations=observations,
        environment=environment_evidence,
        actions=all_actions,
        style=style,
        variation=variation,
        seed=seed,
    )

    plant_name = leaf.get("plant_name") or "Unnamed plant"
    leaf_label = leaf.get("label") or f"Leaf {leaf_id}"

    sections = [
        f"SMART PLANT DOCTOR — LEAF INTELLIGENCE REPORT",
        f"Plant: {plant_name}",
        f"Leaf: {leaf_label} (ID {leaf_id})",
        f"Species registered: {leaf.get('species') or 'Not recorded'}",
        "",
        "1. CURRENT AI ASSESSMENT",
        condition_text,
        "",
        "2. OBSERVATION HISTORY",
        history_text,
        "",
        "3. ENVIRONMENTAL CONTEXT",
        environment_text,
        "",
        "4. CARE HISTORY",
        care_text,
        "",
        "5. KNOWLEDGE-BASE REFERENCES",
        knowledge_text,
        "",
        "6. RECOMMENDATIONS",
    ]

    for number, recommendation in enumerate(recommendations, start=1):
        sections.append(f"{number}. {recommendation['text']}")


    limitations = []

    if not observations:
        limitations.append("No observations have been recorded for this leaf.")

    if not prediction:
        limitations.append("No saved AI prediction is available.")

    if not stations:
        limitations.append(
            "No sensor station is linked to this plant; environmental context "
            "is unavailable."
        )

    if stations and environment_evidence["reading_count"] == 0:
        limitations.append(
            "Associated sensor stations have no recorded readings."
        )

    if environment_evidence.get("missing_fields"):
        limitations.append(
            "The latest environmental reading is missing: "
            + ", ".join(environment_evidence["missing_fields"])
            + "."
        )

    if len(observations) < 2:
        limitations.append(
            "Fewer than two observations are available; a change-over-time "
            "assessment cannot be established."
        )

    limitations.append(
        "Environmental readings describe the associated plant sensor station, "
        "not direct measurements of the individual leaf."
    )
    limitations.append(
        "Model scores are not necessarily calibrated probabilities. "
        "Predictions are not definitive diagnoses."
    )

    sections.extend(["", "7. LIMITATIONS"])
    sections.extend(f"- {item}" for item in limitations)

    report_text = "\n".join(sections)

    evidence = {
        "leaf_id": leaf_id,
        "plant_id": leaf["plant_id"],
        "observation_count": len(observations),
        "observations": observations,
        "latest_prediction": prediction,
        "knowledge_base": knowledge,
        "environment": environment_evidence,
        "sensor_stations": stations,
        "care_actions": all_actions,
        "recommendations": recommendations,
    }

    result = {
        "leaf_id": leaf_id,
        "plant_id": leaf["plant_id"],
        "style": style,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary_text": report_text,
        "evidence": evidence,
        "limitations": limitations,
    }

    if save:
        result["summary_id"] = save_health_summary(
            plant_id=leaf["plant_id"],
            leaf_id=leaf_id,
            summary_text=report_text,
            evidence=evidence,
            limitations=limitations,
            generator_version=generator_version,
        )

    return result
