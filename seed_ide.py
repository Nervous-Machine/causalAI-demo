#!/usr/bin/env python3
"""Seed the nervous-machine-ide MongoDB with data-centers thermal management graph.

Maps the nm causal graph into the event_pod schema so the IDE lights up
with the same graph the CLI builds.

Usage:
    python3 seed_ide.py [--user vertiv_demo]
    python3 seed_ide.py --drop  # remove seeded data
"""
import sys
from datetime import datetime, timezone
from pymongo import MongoClient

MONGO_URI = "mongodb+srv://heidi_db_user:yuh8joKLG44OPqI6@contextpods.f7blnqv.mongodb.net/?appName=ContextPods&tlsAllowInvalidCertificates=true"
USER_ID = "data-center-thermal"
POD_DB = "pod_quiteheidi_0e56"

# ── Thermal management causal graph ──
# Signal types mapped: drivers → "entity", outcomes → "metric", mechanisms → "pattern"

NODES = [
    # Drivers
    {"key": "it_workload",        "signal_type": "entity", "value": 0.8, "certainty": 0.30, "gloss": "CPU/GPU power draw per rack (kW)",         "meta": {"unit": "kW",  "role": "driver"}},
    {"key": "crac_airflow",       "signal_type": "entity", "value": 0.7, "certainty": 0.30, "gloss": "CRAC fan speed and supply temperature",     "meta": {"unit": "CFM", "role": "driver"}},
    {"key": "ambient_temperature","signal_type": "entity", "value": 0.5, "certainty": 0.30, "gloss": "Outside air temperature",                   "meta": {"unit": "°C",  "role": "driver"}},
    {"key": "rack_configuration", "signal_type": "entity", "value": 0.4, "certainty": 0.30, "gloss": "Blanking panels, cable management score",    "meta": {"unit": "score", "role": "driver"}},
    {"key": "floor_tile_layout",  "signal_type": "entity", "value": 0.4, "certainty": 0.30, "gloss": "Perforated tile placement and open %",       "meta": {"unit": "open_%", "role": "driver"}},
    # Outcomes
    {"key": "zone_temperature",   "signal_type": "metric", "value": 0.9, "certainty": 0.30, "gloss": "Per-zone inlet/outlet temperature",         "meta": {"unit": "°C",  "role": "outcome"}},
    {"key": "cooling_power",      "signal_type": "metric", "value": 0.7, "certainty": 0.30, "gloss": "Cooling demand and PUE contribution",       "meta": {"unit": "kW",  "role": "outcome"}},
    {"key": "thermal_alarm",      "signal_type": "metric", "value": 0.6, "certainty": 0.30, "gloss": "Temperature threshold trigger",             "meta": {"unit": "boolean", "role": "outcome"}},
]

EDGES = [
    {"source_key": "it_workload",        "target_key": "zone_temperature", "relationship": "ENABLES",   "note": "heat_dissipation — Z₀=0.30 (llm_training_data)"},
    {"source_key": "crac_airflow",       "target_key": "zone_temperature", "relationship": "ENABLES",   "note": "convective_cooling — Z₀=0.30 (llm_training_data)"},
    {"source_key": "ambient_temperature","target_key": "zone_temperature", "relationship": "ENABLES",   "note": "economizer_link — Z₀=0.30 (llm_training_data)"},
    {"source_key": "rack_configuration", "target_key": "zone_temperature", "relationship": "RELATED_TO","note": "recirculation — Z₀=0.30 (llm_training_data)"},
    {"source_key": "floor_tile_layout",  "target_key": "zone_temperature", "relationship": "RELATED_TO","note": "airflow_distribution — Z₀=0.30 (llm_training_data)"},
    {"source_key": "zone_temperature",   "target_key": "cooling_power",    "relationship": "ENABLES",   "note": "crac_control_response — Z₀=0.30 (llm_training_data)"},
    {"source_key": "zone_temperature",   "target_key": "thermal_alarm",    "relationship": "ENABLES",   "note": "threshold_trigger — Z₀=0.30 (llm_training_data)"},
]

# Signal types for sources/targets in links
NODE_SIGNAL = {n["key"]: n["signal_type"] for n in NODES}


def seed():
    client = MongoClient(MONGO_URI)
    db = client[POD_DB]
    now = datetime.now(timezone.utc)

    # ── Pod metadata ──
    db.pods.delete_many({"user_id": USER_ID})
    db.pods.insert_one({
        "user_id": USER_ID,
        "name": "Data Center Thermal Management",
        "pod_type": "entity_model",
        "created_at": now,
        "last_session": now,
        "total_interactions": 7,
        "total_sessions": 1,
        "overall_certainty": 0.30,
    })

    # ── Events (nodes) ──
    db.events.delete_many({"user_id": USER_ID})
    events = []
    for n in NODES:
        events.append({
            "user_id": USER_ID,
            "key": n["key"],
            "signal_type": n["signal_type"],
            "value": n["value"],
            "certainty": n["certainty"],
            "validation_type": "logic",
            "gloss": n["gloss"],
            "meta": n["meta"],
            "sources": [{
                "source_id": "llm_training_data",
                "source_type": "prior",
                "value": n["certainty"],
                "confidence": n["certainty"],
                "timestamp": now,
            }],
            "observation_count": 0,
            "created_at": now,
            "updated_at": now,
        })
    db.events.insert_many(events)

    # ── Event links (edges) ──
    db.event_links.delete_many({"user_id": USER_ID})
    links = []
    for e in EDGES:
        links.append({
            "user_id": USER_ID,
            "source_key": e["source_key"],
            "source_signal_type": NODE_SIGNAL[e["source_key"]],
            "target_key": e["target_key"],
            "target_signal_type": NODE_SIGNAL[e["target_key"]],
            "relationship": e["relationship"],
            "note": e["note"],
            "created_at": now,
        })
    db.event_links.insert_many(links)

    print(f"✓ Seeded {len(events)} nodes + {len(links)} edges → {POD_DB}")
    print(f"  User: {USER_ID}")
    print(f"  Pod:  Data Center Thermal Management")
    print(f"  All edges at Z₀ = 0.30 (llm_training_data prior)")

    # Clean up orphaned vertiv_demo if it exists
    admin_db = client["context_pod_admin"]
    admin_db.beta_users.delete_one({"user_id": "vertiv_demo"})
    client.drop_database("pod_vertiv_demo")

    client.close()


def drop():
    client = MongoClient(MONGO_URI)
    db = client[POD_DB]
    db.pods.delete_many({"user_id": USER_ID})
    db.events.delete_many({"user_id": USER_ID})
    db.event_links.delete_many({"user_id": USER_ID})
    print(f"✓ Dropped all {USER_ID} data from {POD_DB}")
    client.close()


if __name__ == "__main__":
    if "--drop" in sys.argv:
        drop()
    else:
        seed()
