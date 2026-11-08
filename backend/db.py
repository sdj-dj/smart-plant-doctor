"""SQLite database for Smart Plant Doctor.

Stores plant identity, individual leaf histories, AI predictions,
environmental readings, care actions, and generated health summaries.
"""

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "smart_plant_doctor.db"

DB_PATH = Path(os.environ.get("SMART_PLANT_DB", str(DEFAULT_DB_PATH)))


SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS plants (
    plant_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    species TEXT NOT NULL,
    variety TEXT,
    location TEXT,
    planted_at TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS leaves (
    leaf_id INTEGER PRIMARY KEY AUTOINCREMENT,
    plant_id INTEGER NOT NULL,
    label TEXT,
    identification_method TEXT NOT NULL DEFAULT 'user_assigned',
    first_observed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    notes TEXT,
    FOREIGN KEY (plant_id) REFERENCES plants(plant_id)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS leaf_observations (
    observation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    leaf_id INTEGER NOT NULL,
    observed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    image_path TEXT,
    symptoms TEXT,
    user_notes TEXT,
    image_quality_notes TEXT,
    FOREIGN KEY (leaf_id) REFERENCES leaves(leaf_id)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS ai_predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    observation_id INTEGER NOT NULL,
    model_name TEXT NOT NULL,
    model_version TEXT,
    predicted_species TEXT,
    species_score REAL,
    predicted_condition TEXT,
    condition_score REAL,
    candidates_json TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (observation_id) REFERENCES leaf_observations(observation_id)
        ON DELETE CASCADE,
    CHECK (species_score IS NULL OR
           (species_score >= 0 AND species_score <= 1)),
    CHECK (condition_score IS NULL OR
           (condition_score >= 0 AND condition_score <= 1))
);

CREATE TABLE IF NOT EXISTS sensor_stations (
    station_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    plant_id INTEGER,
    location TEXT,
    device_type TEXT,
    calibration_notes TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (plant_id) REFERENCES plants(plant_id)
        ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS sensor_readings (
    reading_id INTEGER PRIMARY KEY AUTOINCREMENT,
    station_id INTEGER NOT NULL,
    recorded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    temperature_c REAL,
    humidity_pct REAL,
    soil_moisture_pct REAL,
    light_lux REAL,
    ph REAL,
    source TEXT NOT NULL DEFAULT 'sensor',
    quality_status TEXT NOT NULL DEFAULT 'unverified',
    FOREIGN KEY (station_id) REFERENCES sensor_stations(station_id),
    CHECK (humidity_pct IS NULL OR
           (humidity_pct >= 0 AND humidity_pct <= 100)),
    CHECK (soil_moisture_pct IS NULL OR
           (soil_moisture_pct >= 0 AND soil_moisture_pct <= 100)),
    CHECK (light_lux IS NULL OR light_lux >= 0),
    CHECK (ph IS NULL OR (ph >= 0 AND ph <= 14))
);

CREATE TABLE IF NOT EXISTS care_actions (
    action_id INTEGER PRIMARY KEY AUTOINCREMENT,
    plant_id INTEGER NOT NULL,
    leaf_id INTEGER,
    action_type TEXT NOT NULL,
    action_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    details TEXT,
    outcome_notes TEXT,
    FOREIGN KEY (plant_id) REFERENCES plants(plant_id)
        ON DELETE CASCADE,
    FOREIGN KEY (leaf_id) REFERENCES leaves(leaf_id)
        ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS health_summaries (
    summary_id INTEGER PRIMARY KEY AUTOINCREMENT,
    plant_id INTEGER NOT NULL,
    leaf_id INTEGER,
    generated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    period_start TEXT,
    period_end TEXT,
    summary_text TEXT NOT NULL,
    evidence_json TEXT,
    limitations_json TEXT,
    generator_version TEXT,
    FOREIGN KEY (plant_id) REFERENCES plants(plant_id)
        ON DELETE CASCADE,
    FOREIGN KEY (leaf_id) REFERENCES leaves(leaf_id)
        ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_leaves_plant
    ON leaves(plant_id);

CREATE INDEX IF NOT EXISTS idx_observations_leaf_time
    ON leaf_observations(leaf_id, observed_at);

CREATE INDEX IF NOT EXISTS idx_predictions_observation
    ON ai_predictions(observation_id);

CREATE INDEX IF NOT EXISTS idx_readings_station_time
    ON sensor_readings(station_id, recorded_at);

CREATE INDEX IF NOT EXISTS idx_actions_plant_time
    ON care_actions(plant_id, action_at);

CREATE INDEX IF NOT EXISTS idx_summaries_leaf_time
    ON health_summaries(leaf_id, generated_at);
"""


@contextmanager
def get_connection():
    """Open a database connection with foreign-key checks enabled."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")

    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def init_db():
    """Create the database tables and indexes if they do not exist."""
    with get_connection() as connection:
        connection.executescript(SCHEMA)

    return DB_PATH


def list_tables():
    """Return the names of application tables currently in the database."""
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        ).fetchall()

    return [row["name"] for row in rows]


def register_plant(name, species, variety=None, location=None, planted_at=None):
    """Register a plant and return its new plant ID."""
    if not name or not name.strip():
        raise ValueError("Plant name is required.")
    if not species or not species.strip():
        raise ValueError("Plant species is required.")

    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO plants (name, species, variety, location, planted_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                name.strip(),
                species.strip(),
                variety,
                location,
                planted_at,
            ),
        )
        return cursor.lastrowid


def register_leaf(plant_id, label=None, notes=None):
    """Register a leaf belonging to a plant and return its leaf ID."""
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO leaves (plant_id, label, notes)
            VALUES (?, ?, ?)
            """,
            (plant_id, label, notes),
        )
        return cursor.lastrowid


def add_leaf_observation(
    leaf_id,
    image_path=None,
    symptoms=None,
    user_notes=None,
    image_quality_notes=None,
    observed_at=None,
):
    """Save an observation of an existing leaf."""
    with get_connection() as connection:
        if observed_at is None:
            cursor = connection.execute(
                """
                INSERT INTO leaf_observations
                    (leaf_id, image_path, symptoms, user_notes,
                     image_quality_notes)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    leaf_id,
                    image_path,
                    symptoms,
                    user_notes,
                    image_quality_notes,
                ),
            )
        else:
            cursor = connection.execute(
                """
                INSERT INTO leaf_observations
                    (leaf_id, observed_at, image_path, symptoms,
                     user_notes, image_quality_notes)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    leaf_id,
                    observed_at,
                    image_path,
                    symptoms,
                    user_notes,
                    image_quality_notes,
                ),
            )

        return cursor.lastrowid


def get_leaf_history(leaf_id):
    """Return a leaf, its parent plant, and all dated observations."""
    with get_connection() as connection:
        leaf = connection.execute(
            """
            SELECT
                l.leaf_id,
                l.label,
                l.identification_method,
                l.first_observed_at,
                l.notes AS leaf_notes,
                p.plant_id,
                p.name AS plant_name,
                p.species,
                p.variety,
                p.location
            FROM leaves AS l
            JOIN plants AS p ON p.plant_id = l.plant_id
            WHERE l.leaf_id = ?
            """,
            (leaf_id,),
        ).fetchone()

        if leaf is None:
            return None

        observations = connection.execute(
            """
            SELECT *
            FROM leaf_observations
            WHERE leaf_id = ?
            ORDER BY observed_at, observation_id
            """,
            (leaf_id,),
        ).fetchall()

        return {
            "leaf": dict(leaf),
            "observations": [dict(row) for row in observations],
        }


def save_ai_prediction(
    observation_id,
    species_candidates,
    condition_candidates,
    model_name="ResNet18AgriculturalAI",
    model_version=None,
):
    """Save the model's leading species and condition candidates."""
    import json
    import math

    def validate_candidates(candidates, field_name):
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(f"{field_name} must be a non-empty list.")

        validated = []

        for candidate in candidates:
            if not isinstance(candidate, dict):
                raise ValueError("Each candidate must be a dictionary.")

            label = candidate.get("label")
            score = candidate.get("score")

            if not isinstance(label, str) or not label.strip():
                raise ValueError("Candidate labels must be non-empty strings.")

            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise ValueError("Candidate scores must be numeric.")

            score = float(score)

            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Candidate scores must be between 0 and 1.")

            validated.append({"label": label.strip(), "score": score})

        return validated

    species = validate_candidates(species_candidates, "species_candidates")
    conditions = validate_candidates(
        condition_candidates, "condition_candidates"
    )

    with get_connection() as connection:
        observation = connection.execute(
            "SELECT observation_id FROM leaf_observations WHERE observation_id = ?",
            (observation_id,),
        ).fetchone()

        if observation is None:
            raise ValueError(f"Observation {observation_id} does not exist.")

        cursor = connection.execute(
            """
            INSERT INTO ai_predictions (
                observation_id, model_name, model_version,
                predicted_species, species_score,
                predicted_condition, condition_score, candidates_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                observation_id,
                model_name,
                model_version,
                species[0]["label"],
                species[0]["score"],
                conditions[0]["label"],
                conditions[0]["score"],
                json.dumps({
                    "species_candidates": species,
                    "condition_candidates": conditions,
                }),
            ),
        )

        return cursor.lastrowid


def get_observation_predictions(observation_id):
    """Retrieve saved AI predictions for an observation."""
    import json

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT *
            FROM ai_predictions
            WHERE observation_id = ?
            ORDER BY prediction_id
            """,
            (observation_id,),
        ).fetchall()

    results = []

    for row in rows:
        item = dict(row)
        item["candidates"] = (
            json.loads(item["candidates_json"])
            if item["candidates_json"]
            else None
        )
        results.append(item)

    return results


def register_sensor_station(
    name,
    plant_id=None,
    location=None,
    device_type=None,
    calibration_notes=None,
):
    """Register a sensor station monitoring a plant or growing area."""
    if not name or not name.strip():
        raise ValueError("Sensor station name is required.")

    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO sensor_stations (
                name, plant_id, location, device_type, calibration_notes
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                name.strip(),
                plant_id,
                location,
                device_type,
                calibration_notes,
            ),
        )
        return cursor.lastrowid


def add_sensor_reading(
    station_id,
    temperature_c=None,
    humidity_pct=None,
    soil_moisture_pct=None,
    light_lux=None,
    ph=None,
    recorded_at=None,
    source="sensor",
    quality_status="unverified",
):
    """Store one timestamped environmental reading."""
    import math

    measurements = {
        "temperature_c": temperature_c,
        "humidity_pct": humidity_pct,
        "soil_moisture_pct": soil_moisture_pct,
        "light_lux": light_lux,
        "ph": ph,
    }

    for name, value in measurements.items():
        if value is None:
            continue

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be numeric or None.")

        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite.")

    if humidity_pct is not None and not 0 <= humidity_pct <= 100:
        raise ValueError("Humidity must be between 0 and 100.")

    if soil_moisture_pct is not None and not 0 <= soil_moisture_pct <= 100:
        raise ValueError("Soil moisture must be between 0 and 100.")

    if light_lux is not None and light_lux < 0:
        raise ValueError("Light intensity cannot be negative.")

    if ph is not None and not 0 <= ph <= 14:
        raise ValueError("pH must be between 0 and 14.")

    if not source or not source.strip():
        raise ValueError("Reading source is required.")

    with get_connection() as connection:
        # Confirm that the station exists before inserting.
        station = connection.execute(
            "SELECT station_id FROM sensor_stations WHERE station_id = ?",
            (station_id,),
        ).fetchone()

        if station is None:
            raise ValueError(f"Sensor station {station_id} does not exist.")

        values = (
            station_id,
            temperature_c,
            humidity_pct,
            soil_moisture_pct,
            light_lux,
            ph,
            source,
            quality_status,
        )

        if recorded_at is None:
            cursor = connection.execute(
                """
                INSERT INTO sensor_readings (
                    station_id, temperature_c, humidity_pct,
                    soil_moisture_pct, light_lux, ph, source, quality_status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )
        else:
            cursor = connection.execute(
                """
                INSERT INTO sensor_readings (
                    station_id, recorded_at, temperature_c, humidity_pct,
                    soil_moisture_pct, light_lux, ph, source, quality_status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    station_id,
                    recorded_at,
                    temperature_c,
                    humidity_pct,
                    soil_moisture_pct,
                    light_lux,
                    ph,
                    source,
                    quality_status,
                ),
            )

        return cursor.lastrowid


def get_sensor_history(station_id, limit=100):
    """Return the most recent readings for a sensor station."""
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("Limit must be a positive integer.")

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT *
            FROM sensor_readings
            WHERE station_id = ?
            ORDER BY recorded_at DESC, reading_id DESC
            LIMIT ?
            """,
            (station_id, limit),
        ).fetchall()

    return [dict(row) for row in rows]


def record_care_action(
    plant_id,
    action_type,
    leaf_id=None,
    details=None,
    outcome_notes=None,
    action_at=None,
):
    """Record a care action for a plant, optionally linked to one leaf."""
    import math
    from datetime import datetime

    if isinstance(plant_id, bool) or not isinstance(plant_id, int) or plant_id < 1:
        raise ValueError("plant_id must be a positive integer.")

    if not isinstance(action_type, str) or not action_type.strip():
        raise ValueError("action_type must be a non-empty string.")

    if leaf_id is not None and (
        isinstance(leaf_id, bool)
        or not isinstance(leaf_id, int)
        or leaf_id < 1
    ):
        raise ValueError("leaf_id must be a positive integer or None.")

    if action_at is not None:
        if not isinstance(action_at, str) or not action_at.strip():
            raise ValueError("action_at must be a non-empty timestamp string.")
        try:
            datetime.fromisoformat(action_at.strip().replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("action_at must be an ISO-format timestamp.") from exc

    with get_connection() as connection:
        plant = connection.execute(
            "SELECT plant_id FROM plants WHERE plant_id = ?",
            (plant_id,),
        ).fetchone()

        if plant is None:
            raise ValueError(f"Plant {plant_id} does not exist.")

        if leaf_id is not None:
            leaf = connection.execute(
                "SELECT plant_id FROM leaves WHERE leaf_id = ?",
                (leaf_id,),
            ).fetchone()

            if leaf is None:
                raise ValueError(f"Leaf {leaf_id} does not exist.")

            if leaf["plant_id"] != plant_id:
                raise ValueError("The selected leaf does not belong to this plant.")

        if action_at is None:
            cursor = connection.execute(
                """
                INSERT INTO care_actions (
                    plant_id, leaf_id, action_type, details, outcome_notes
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (plant_id, leaf_id, action_type.strip(), details, outcome_notes),
            )
        else:
            cursor = connection.execute(
                """
                INSERT INTO care_actions (
                    plant_id, leaf_id, action_type, action_at,
                    details, outcome_notes
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    plant_id, leaf_id, action_type.strip(), action_at.strip(),
                    details, outcome_notes,
                ),
            )

        return cursor.lastrowid


def get_care_history(plant_id, leaf_id=None, limit=100, offset=0):
    """Retrieve care actions with pagination."""
    if isinstance(plant_id, bool) or not isinstance(plant_id, int) or plant_id < 1:
        raise ValueError("plant_id must be a positive integer.")

    if leaf_id is not None and (
        isinstance(leaf_id, bool)
        or not isinstance(leaf_id, int)
        or leaf_id < 1
    ):
        raise ValueError("leaf_id must be a positive integer or None.")

    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
        raise ValueError("limit must be an integer between 1 and 1000.")

    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError("offset must be a non-negative integer.")

    query = """
        SELECT *
        FROM care_actions
        WHERE plant_id = ?
    """
    parameters = [plant_id]

    if leaf_id is not None:
        query += " AND leaf_id = ?"
        parameters.append(leaf_id)

    query += " ORDER BY action_at DESC, action_id DESC LIMIT ? OFFSET ?"
    parameters.extend([limit, offset])

    with get_connection() as connection:
        rows = connection.execute(query, parameters).fetchall()

    return [dict(row) for row in rows]


def save_health_summary(
    plant_id,
    summary_text,
    leaf_id=None,
    period_start=None,
    period_end=None,
    evidence=None,
    limitations=None,
    generator_version="rules-v1",
):
    """Save a report and the evidence/limitations used to create it."""
    import json

    if isinstance(plant_id, bool) or not isinstance(plant_id, int) or plant_id < 1:
        raise ValueError("plant_id must be a positive integer.")

    if not isinstance(summary_text, str) or not summary_text.strip():
        raise ValueError("summary_text must be a non-empty string.")

    if leaf_id is not None and (
        isinstance(leaf_id, bool)
        or not isinstance(leaf_id, int)
        or leaf_id < 1
    ):
        raise ValueError("leaf_id must be a positive integer or None.")

    with get_connection() as connection:
        plant = connection.execute(
            "SELECT plant_id FROM plants WHERE plant_id = ?",
            (plant_id,),
        ).fetchone()

        if plant is None:
            raise ValueError(f"Plant {plant_id} does not exist.")

        if leaf_id is not None:
            leaf = connection.execute(
                "SELECT plant_id FROM leaves WHERE leaf_id = ?",
                (leaf_id,),
            ).fetchone()

            if leaf is None:
                raise ValueError(f"Leaf {leaf_id} does not exist.")

            if leaf["plant_id"] != plant_id:
                raise ValueError("The selected leaf does not belong to this plant.")

        cursor = connection.execute(
            """
            INSERT INTO health_summaries (
                plant_id, leaf_id, period_start, period_end,
                summary_text, evidence_json, limitations_json,
                generator_version
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                plant_id,
                leaf_id,
                period_start,
                period_end,
                summary_text.strip(),
                json.dumps(evidence if evidence is not None else [], allow_nan=False),
                json.dumps(
                    limitations if limitations is not None else [],
                    allow_nan=False,
                ),
                generator_version,
            ),
        )

        return cursor.lastrowid


def get_health_summaries(plant_id, leaf_id=None, limit=50, offset=0):
    """Retrieve saved reports, newest first, with pagination."""
    import json

    if isinstance(plant_id, bool) or not isinstance(plant_id, int) or plant_id < 1:
        raise ValueError("plant_id must be a positive integer.")

    if leaf_id is not None and (
        isinstance(leaf_id, bool)
        or not isinstance(leaf_id, int)
        or leaf_id < 1
    ):
        raise ValueError("leaf_id must be a positive integer or None.")

    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
        raise ValueError("limit must be an integer between 1 and 1000.")

    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError("offset must be a non-negative integer.")

    query = """
        SELECT *
        FROM health_summaries
        WHERE plant_id = ?
    """
    parameters = [plant_id]

    if leaf_id is not None:
        query += " AND leaf_id = ?"
        parameters.append(leaf_id)

    query += " ORDER BY generated_at DESC, summary_id DESC LIMIT ? OFFSET ?"
    parameters.extend([limit, offset])

    with get_connection() as connection:
        rows = connection.execute(query, parameters).fetchall()

    results = []
    for row in rows:
        item = dict(row)
        item["evidence"] = json.loads(item.pop("evidence_json") or "[]")
        item["limitations"] = json.loads(item.pop("limitations_json") or "[]")
        results.append(item)

    return results
