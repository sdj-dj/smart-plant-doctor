"""Knowledge retrieval and management for Smart Plant Doctor."""

import json
from backend.db import get_connection, init_db


def search_plant_knowledge(query=None, limit=20):
    """Search registered plant profiles."""
    init_db()
    query = (query or "").strip().casefold()

    with get_connection() as connection:
        if query:
            rows = connection.execute(
                """
                SELECT *
                FROM plant_knowledge
                WHERE lower(display_name) LIKE ?
                ORDER BY display_name
                LIMIT ?
                """,
                (f"%{query}%", limit),
            ).fetchall()
        else:
            rows = connection.execute(
                """
                SELECT *
                FROM plant_knowledge
                ORDER BY display_name
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

    return [dict(row) for row in rows]


def search_condition_knowledge(query=None, limit=50):
    """Search condition classes and reviewed descriptions."""
    init_db()
    query = (query or "").strip().casefold()

    with get_connection() as connection:
        if query:
            rows = connection.execute(
                """
                SELECT *
                FROM condition_knowledge
                WHERE lower(display_name) LIKE ?
                   OR lower(category) LIKE ?
                   OR lower(COALESCE(description, '')) LIKE ?
                ORDER BY display_name
                LIMIT ?
                """,
                (
                    f"%{query}%",
                    f"%{query}%",
                    f"%{query}%",
                    limit,
                ),
            ).fetchall()
        else:
            rows = connection.execute(
                """
                SELECT *
                FROM condition_knowledge
                ORDER BY display_name
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

    return [dict(row) for row in rows]


def add_knowledge_source(
    title,
    organization=None,
    source_url=None,
    publication_date=None,
    source_type="reference",
    notes=None,
):
    """Register a source. Registration does not imply verification."""
    if not title or not title.strip():
        raise ValueError("A source title is required.")

    init_db()

    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO knowledge_sources (
                title, organization, source_url, publication_date,
                source_type, verification_status, notes
            )
            VALUES (?, ?, ?, ?, ?, 'needs_review', ?)
            """,
            (
                title.strip(),
                organization,
                source_url,
                publication_date,
                source_type,
                notes,
            ),
        )
        return cursor.lastrowid


def add_symptom(
    symptom_name,
    description=None,
    plant_key=None,
    condition_key=None,
    distinguishing_features=None,
    source_id=None,
):
    """Store a symptom entry with optional source attribution."""
    if not symptom_name or not symptom_name.strip():
        raise ValueError("A symptom name is required.")

    init_db()

    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO symptom_knowledge (
                plant_key, condition_key, symptom_name, description,
                distinguishing_features_json, source_id,
                verification_status
            )
            VALUES (?, ?, ?, ?, ?, ?, 'needs_review')
            """,
            (
                plant_key,
                condition_key,
                symptom_name.strip(),
                description,
                json.dumps(distinguishing_features or []),
                source_id,
            ),
        )
        return cursor.lastrowid


def add_care_guidance(
    title,
    instructions,
    plant_key=None,
    condition_key=None,
    caution=None,
    source_id=None,
    evidence_level="unverified",
):
    """Store guidance without presenting it as verified by default."""
    if not title or not title.strip():
        raise ValueError("A guidance title is required.")

    if not instructions or not instructions.strip():
        raise ValueError("Guidance instructions are required.")

    init_db()

    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO care_guidance (
                plant_key, condition_key, title, instructions,
                caution, evidence_level, source_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                plant_key,
                condition_key,
                title.strip(),
                instructions.strip(),
                caution,
                evidence_level,
                source_id,
            ),
        )
        return cursor.lastrowid


def retrieve_relevant_knowledge(
    plant_name,
    condition_name,
    limit=20,
):
    """
    Retrieve matching plant, condition, symptom and care records.

    This function does not diagnose a plant or infer that a model
    prediction is correct. Unverified knowledge remains identifiable.
    """
    init_db()

    plant_key = (plant_name or "").strip().casefold()
    condition_key = (condition_name or "").strip().casefold()

    with get_connection() as connection:
        plant = connection.execute(
            """
            SELECT *
            FROM plant_knowledge
            WHERE plant_key = ?
            """,
            (plant_key,),
        ).fetchone()

        condition = connection.execute(
            """
            SELECT *
            FROM condition_knowledge
            WHERE condition_key = ?
            """,
            (condition_key,),
        ).fetchone()

        symptoms = connection.execute(
            """
            SELECT s.*, ks.title AS source_title,
                   ks.source_url, ks.verification_status AS source_status
            FROM symptom_knowledge AS s
            LEFT JOIN knowledge_sources AS ks
              ON ks.source_id = s.source_id
            WHERE (s.plant_key IS NULL OR s.plant_key = ?)
              AND (s.condition_key IS NULL OR s.condition_key = ?)
            ORDER BY s.symptom_id DESC
            LIMIT ?
            """,
            (plant_key, condition_key, limit),
        ).fetchall()

        guidance = connection.execute(
            """
            SELECT g.*, ks.title AS source_title,
                   ks.source_url, ks.verification_status AS source_status
            FROM care_guidance AS g
            LEFT JOIN knowledge_sources AS ks
              ON ks.source_id = g.source_id
            WHERE (g.plant_key IS NULL OR g.plant_key = ?)
              AND (g.condition_key IS NULL OR g.condition_key = ?)
            ORDER BY g.guidance_id DESC
            LIMIT ?
            """,
            (plant_key, condition_key, limit),
        ).fetchall()

    return {
        "plant": dict(plant) if plant else None,
        "condition": dict(condition) if condition else None,
        "symptoms": [dict(row) for row in symptoms],
        "care_guidance": [dict(row) for row in guidance],
        "notice": (
            "Knowledge retrieval is not a diagnosis. Verify sources, "
            "and distinguish reviewed evidence from unverified entries."
        ),
    }


def knowledge_base_stats():
    """Return counts for all knowledge-base tables."""
    init_db()

    tables = [
        "plant_knowledge",
        "condition_knowledge",
        "plant_condition_links",
        "symptom_knowledge",
        "care_guidance",
        "knowledge_sources",
        "environmental_rules",
    ]

    with get_connection() as connection:
        return {
            table: connection.execute(
                f'SELECT COUNT(*) FROM "{table}"'
            ).fetchone()[0]
            for table in tables
        }


if __name__ == "__main__":
    print("SMART PLANT DOCTOR — KNOWLEDGE BASE")
    print("=" * 42)

    for table, count in knowledge_base_stats().items():
        print(f"{table}: {count}")

    print("\nSample plant search:")
    for plant in search_plant_knowledge("tea"):
        print(plant["display_name"], "-", plant["verification_status"])

    print("\nSample condition search:")
    for condition in search_condition_knowledge("blight"):
        print(condition["display_name"], "-", condition["category"])
