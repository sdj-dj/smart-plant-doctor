"""Bulk-import source-aware plant knowledge into Smart Plant Doctor."""

import csv
import json
import sys
from pathlib import Path

from backend.db import get_connection, init_db

ROOT = Path(__file__).resolve().parent.parent
IMPORT_DIR = ROOT / "data" / "knowledge_import"

REQUIRED_FILES = {
    "sources.csv": [
        "title", "organization", "source_url",
        "publication_date", "source_type", "notes"
    ],
    "plant_profiles.csv": [
        "plant_key", "description", "growth_stages_json",
        "environmental_guidance_json", "care_guidance_json",
        "source_id"
    ],
    "condition_details.csv": [
        "condition_key", "description", "symptoms_json",
        "possible_causes_json", "management_guidance_json",
        "source_id"
    ],
    "plant_condition_links.csv": [
        "plant_key", "condition_key", "evidence_notes", "source_id"
    ],
}


def read_csv(path, expected_columns):
    if not path.exists():
        return []

    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        missing = set(expected_columns) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{path.name} is missing columns: {sorted(missing)}"
            )

        return list(reader)


def parse_json_list(value, field):
    if not value:
        return []

    result = json.loads(value)

    if not isinstance(result, list):
        raise ValueError(f"{field} must contain a JSON list.")

    return result


def parse_json_object(value, field):
    if not value:
        return {}

    result = json.loads(value)

    if not isinstance(result, dict):
        raise ValueError(f"{field} must contain a JSON object.")

    return result


def optional_id(value):
    return int(value) if value and value.strip() else None


def import_sources(connection, rows):
    imported = 0

    for row in rows:
        title = row["title"].strip()
        if not title:
            raise ValueError("Source title cannot be empty.")

        connection.execute(
            """
            INSERT INTO knowledge_sources (
                title, organization, source_url, publication_date,
                source_type, verification_status, notes
            )
            SELECT ?, ?, ?, ?, ?, 'needs_review', ?
            WHERE NOT EXISTS (
                SELECT 1 FROM knowledge_sources
                WHERE title = ? AND COALESCE(source_url, '') = ?
            )
            """,
            (
                title,
                row["organization"] or None,
                row["source_url"] or None,
                row["publication_date"] or None,
                row["source_type"] or "reference",
                row["notes"] or None,
                title,
                row["source_url"] or "",
            ),
        )
        imported += connection.execute("SELECT changes()").fetchone()[0]

    return imported


def import_plant_profiles(connection, rows):
    imported = 0

    for row in rows:
        key = row["plant_key"].strip().casefold()
        if not key:
            raise ValueError("plant_key cannot be empty.")

        source_id = optional_id(row["source_id"])

        if source_id is not None:
            exists = connection.execute(
                "SELECT 1 FROM knowledge_sources WHERE source_id = ?",
                (source_id,),
            ).fetchone()
            if not exists:
                raise ValueError(f"Unknown source_id: {source_id}")

        cursor = connection.execute(
            """
            UPDATE plant_knowledge
            SET description = ?,
                growth_stages_json = ?,
                environmental_guidance_json = ?,
                care_guidance_json = ?,
                verification_status = CASE
                    WHEN ? IS NOT NULL THEN 'source_attached_needs_review'
                    ELSE verification_status
                END,
                updated_at = CURRENT_TIMESTAMP
            WHERE plant_key = ?
            """,
            (
                row["description"] or None,
                json.dumps(parse_json_list(
                    row["growth_stages_json"], "growth_stages_json"
                )),
                json.dumps(parse_json_object(
                    row["environmental_guidance_json"],
                    "environmental_guidance_json"
                )),
                json.dumps(parse_json_list(
                    row["care_guidance_json"], "care_guidance_json"
                )),
                source_id,
                key,
            ),
        )

        if cursor.rowcount == 0:
            raise ValueError(
                f"Plant '{key}' does not match an existing model class."
            )

        if source_id is not None and row["description"]:
            connection.execute(
                """
                INSERT INTO care_guidance (
                    plant_key, title, instructions, source_id, evidence_level
                )
                SELECT ?, 'Imported plant guidance', ?, ?, 'needs_review'
                WHERE NOT EXISTS (
                    SELECT 1 FROM care_guidance
                    WHERE plant_key = ? AND title = 'Imported plant guidance'
                      AND instructions = ?
                )
                """,
                (
                    key, row["description"], source_id,
                    key, row["description"],
                ),
            )

        imported += 1

    return imported


def import_condition_details(connection, rows):
    imported = 0

    for row in rows:
        key = row["condition_key"].strip().casefold()
        if not key:
            raise ValueError("condition_key cannot be empty.")

        source_id = optional_id(row["source_id"])

        if source_id is not None:
            exists = connection.execute(
                "SELECT 1 FROM knowledge_sources WHERE source_id = ?",
                (source_id,),
            ).fetchone()
            if not exists:
                raise ValueError(f"Unknown source_id: {source_id}")

        cursor = connection.execute(
            """
            UPDATE condition_knowledge
            SET description = ?,
                symptoms_json = ?,
                possible_causes_json = ?,
                management_guidance_json = ?,
                verification_status = CASE
                    WHEN ? IS NOT NULL THEN 'source_attached_needs_review'
                    ELSE verification_status
                END,
                updated_at = CURRENT_TIMESTAMP
            WHERE condition_key = ?
            """,
            (
                row["description"] or None,
                json.dumps(parse_json_list(
                    row["symptoms_json"], "symptoms_json"
                )),
                json.dumps(parse_json_list(
                    row["possible_causes_json"], "possible_causes_json"
                )),
                json.dumps(parse_json_list(
                    row["management_guidance_json"],
                    "management_guidance_json"
                )),
                source_id,
                key,
            ),
        )

        if cursor.rowcount == 0:
            raise ValueError(
                f"Condition '{key}' does not match an existing model class."
            )

        if source_id is not None:
            for symptom in parse_json_list(
                row["symptoms_json"], "symptoms_json"
            ):
                if not isinstance(symptom, str) or not symptom.strip():
                    continue

                connection.execute(
                    """
                    INSERT INTO symptom_knowledge (
                        condition_key, symptom_name, source_id,
                        verification_status
                    )
                    SELECT ?, ?, ?, 'needs_review'
                    WHERE NOT EXISTS (
                        SELECT 1 FROM symptom_knowledge
                        WHERE condition_key = ? AND symptom_name = ?
                    )
                    """,
                    (key, symptom.strip(), source_id, key, symptom.strip()),
                )

            for guidance in parse_json_list(
                row["management_guidance_json"],
                "management_guidance_json",
            ):
                if not isinstance(guidance, str) or not guidance.strip():
                    continue

                connection.execute(
                    """
                    INSERT INTO care_guidance (
                        condition_key, title, instructions,
                        source_id, evidence_level
                    )
                    SELECT ?, 'Imported condition guidance', ?, ?,
                           'needs_review'
                    WHERE NOT EXISTS (
                        SELECT 1 FROM care_guidance
                        WHERE condition_key = ?
                          AND instructions = ?
                    )
                    """,
                    (
                        key, guidance.strip(), source_id,
                        key, guidance.strip(),
                    ),
                )

        imported += 1

    return imported


def import_links(connection, rows):
    imported = 0

    for row in rows:
        plant_key = row["plant_key"].strip().casefold()
        condition_key = row["condition_key"].strip().casefold()
        source_id = optional_id(row["source_id"])

        if not connection.execute(
            "SELECT 1 FROM plant_knowledge WHERE plant_key = ?",
            (plant_key,),
        ).fetchone():
            raise ValueError(f"Unknown plant: {plant_key}")

        if not connection.execute(
            "SELECT 1 FROM condition_knowledge WHERE condition_key = ?",
            (condition_key,),
        ).fetchone():
            raise ValueError(f"Unknown condition: {condition_key}")

        if source_id is not None and not connection.execute(
            "SELECT 1 FROM knowledge_sources WHERE source_id = ?",
            (source_id,),
        ).fetchone():
            raise ValueError(f"Unknown source_id: {source_id}")

        connection.execute(
            """
            INSERT INTO plant_condition_links (
                plant_key, condition_key, relevance_status, evidence_notes
            )
            VALUES (?, ?, 'needs_review', ?)
            ON CONFLICT(plant_key, condition_key) DO UPDATE SET
                evidence_notes = excluded.evidence_notes
            """,
            (
                plant_key,
                condition_key,
                row["evidence_notes"] or None,
            ),
        )
        imported += 1

    return imported


def main():
    init_db()
    IMPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("SMART PLANT DOCTOR — KNOWLEDGE IMPORT")
    print("=" * 44)
    print(f"Import folder: {IMPORT_DIR}")

    missing = [
        name for name in REQUIRED_FILES
        if not (IMPORT_DIR / name).exists()
    ]

    if missing:
        for name, columns in REQUIRED_FILES.items():
            path = IMPORT_DIR / name
            if not path.exists():
                with path.open("w", newline="", encoding="utf-8") as file:
                    writer = csv.writer(file)
                    writer.writerow(columns)

        print("\nCreated CSV templates:")
        for name in missing:
            print(f"  {name}")
        print("\nFill these files with sourced information, then run:")
        print("python -m backend.import_knowledge")
        print("\nEmpty templates are safe; no knowledge was invented.")
        return

    sources = read_csv(
        IMPORT_DIR / "sources.csv", REQUIRED_FILES["sources.csv"]
    )
    plants = read_csv(
        IMPORT_DIR / "plant_profiles.csv",
        REQUIRED_FILES["plant_profiles.csv"],
    )
    conditions = read_csv(
        IMPORT_DIR / "condition_details.csv",
        REQUIRED_FILES["condition_details.csv"],
    )
    links = read_csv(
        IMPORT_DIR / "plant_condition_links.csv",
        REQUIRED_FILES["plant_condition_links.csv"],
    )

    with get_connection() as connection:
        connection.execute("PRAGMA foreign_keys = ON")

        try:
            counts = {
                "Sources": import_sources(connection, sources),
                "Plant profiles": import_plant_profiles(connection, plants),
                "Condition details": import_condition_details(
                    connection, conditions
                ),
                "Plant-condition links": import_links(connection, links),
            }
            connection.commit()
        except Exception:
            connection.rollback()
            raise

    print("\nImport completed:")
    for name, count in counts.items():
        print(f"  {name}: {count}")

    print("\nImported records still require review before being treated")
    print("as verified scientific or agricultural guidance.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, json.JSONDecodeError, csv.Error) as error:
        print(f"\nImport stopped safely: {error}", file=sys.stderr)
        sys.exit(1)
