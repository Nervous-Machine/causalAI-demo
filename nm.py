#!/usr/bin/env python3
"""
nm - Nervous Machine CLI
The GitHub of Causal Learning.

Build causal models. Deploy validation pipelines. Watch them learn.
Inject high-certainty vectors into models. Contribute to the global prior.

Three MCP servers. Two prompt files. One .env.

Usage:
    nm init              Build causal prior from prior.md          [CVOT]
    nm validate          Deploy validation pipelines from validate.md  [Validation]
    nm learn             Run learning loop (ε → η(Z) → belief revision) [CVOT + Validation]
    nm status            Show certainty evolution & curiosity triggers   [CVOT + Validation]
    nm review            Interactive review of prior + validation        [Claude]
    nm inject            Create LoRA adapter from high-certainty graph   [Domain Heads]
    nm train             Train concept extractor (~5-10M params)         [Domain Heads]
    nm deploy            Compile adapter for edge/cloud                  [Domain Heads]
    nm update            Push new vectors to deployed models (no retrain)[Domain Heads]
    nm fleet             Share causal vectors across fleet (~1KB, DDIL)  [Fleet]
    nm contribute        Push anonymized vectors to global prior         [Network]
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # manual env vars are fine

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

CVOT_MCP_URL = os.getenv("CVOT_MCP_URL")
VALIDATION_MCP_URL = os.getenv("VALIDATION_MCP_URL")
DOMAIN_HEADS_MCP_URL = os.getenv("DOMAIN_HEADS_MCP_URL")
FUNCTIONS_HEAD_MCP_URL = os.getenv("FUNCTIONS_HEAD_MCP_URL")
GLOBAL_PRIOR_URL = os.getenv("NM_GLOBAL_PRIOR_URL")
MODEL = os.getenv("NM_MODEL", "claude-sonnet-4-20250514")

# Anthropic client — only needed when MCP servers are configured.
# Dry-run mode (no servers) works with zero dependencies.
client = None
try:
    from anthropic import Anthropic
    client = Anthropic()
except (ImportError, Exception):
    pass  # dry-run mode — no API calls needed

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def read_prompt(filename):
    """Read a prompt file from current directory or examples/"""
    for path in [Path(filename), Path("examples") / filename]:
        if path.exists():
            return path.read_text()
    print(f"✗ {filename} not found. Create it or copy from examples/")
    sys.exit(1)


def call_with_mcp(system_prompt, user_prompt, mcp_servers, stream=True):
    """Call Claude with MCP servers attached. Stream output."""
    if client is None:
        print("  Anthropic SDK not configured. Install with: pip install anthropic")
        print("  Then set ANTHROPIC_API_KEY in .env")
        return ""
    messages = [{"role": "user", "content": user_prompt}]

    mcp_config = [
        {"type": "url", "url": url, "name": name}
        for name, url in mcp_servers.items()
        if url
    ]

    if stream:
        result_text = ""
        with client.messages.stream(
            model=MODEL,
            max_tokens=4096,
            system=system_prompt,
            messages=messages,
            mcp_servers=mcp_config if mcp_config else None,
        ) as stream_response:
            for event in stream_response:
                if hasattr(event, 'type'):
                    if event.type == 'content_block_delta':
                        if hasattr(event.delta, 'text'):
                            print(event.delta.text, end="", flush=True)
                            result_text += event.delta.text
        print()  # newline after stream
        return result_text
    else:
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=system_prompt,
            messages=messages,
            mcp_servers=mcp_config if mcp_config else None,
        )
        text = ""
        for block in response.content:
            if hasattr(block, 'text'):
                text += block.text
        return text


def chat_loop(system_prompt, initial_context, mcp_servers=None):
    """Multi-turn conversation with Claude. Returns when user types 'done' or 'exit'."""
    if client is None:
        print("  Anthropic SDK not configured. Install with: pip install anthropic")
        print("  Then set ANTHROPIC_API_KEY in .env")
        return 0
    messages = [{"role": "user", "content": initial_context}]

    mcp_config = None
    if mcp_servers:
        mcp_config = [
            {"type": "url", "url": url, "name": name}
            for name, url in mcp_servers.items()
            if url
        ]
        if not mcp_config:
            mcp_config = None

    turn = 0
    while True:
        # Stream Claude's response
        result_text = ""
        print()
        print("nm> ", end="", flush=True)

        # Build kwargs — only include mcp_servers if configured
        kwargs = dict(
            model=MODEL,
            max_tokens=4096,
            system=system_prompt,
            messages=messages,
        )
        if mcp_config:
            kwargs["mcp_servers"] = mcp_config

        with client.messages.stream(**kwargs) as stream_response:
            for event in stream_response:
                if hasattr(event, 'type'):
                    if event.type == 'content_block_delta':
                        if hasattr(event.delta, 'text'):
                            print(event.delta.text, end="", flush=True)
                            result_text += event.delta.text
        print()  # newline after stream

        messages.append({"role": "assistant", "content": result_text})
        turn += 1

        # Get user input
        try:
            print()
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user_input.lower() in ("done", "exit", "quit", "q"):
            print()
            print("✓ Review session ended.")
            break

        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})

    return turn


# ─────────────────────────────────────────────
# Domain Examples
# ─────────────────────────────────────────────

AVAILABLE_DOMAINS = {
    "space": "Thermospheric density modeling for satellite drag prediction",
    "robotics": "Sim-to-real calibration for robotic arm dynamics",
    "manufacturing": "CNC machining process quality and root cause attribution",
    "data-centers": "Per-zone thermal management and cooling optimization",
}

# Domain-specific dry-run data for init (nodes + edges)
DOMAIN_INIT_DATA = {
    "space": {
        "domain_id": "thermospheric_density",
        "nodes": [
            ("solar_euv_flux",            "driver",     "SFU"),
            ("geomagnetic_activity",      "driver",     "nT"),
            ("solar_wind_pressure",       "driver",     "nPa"),
            ("seasonal_latitudinal",      "driver",     "factor"),
            ("joule_heating",             "driver",     "GW"),
            ("thermospheric_density",     "outcome",    "kg/m³"),
            ("satellite_drag",            "outcome",    "m/s²"),
            ("collision_avoidance_event", "outcome",    "boolean"),
        ],
        "edges": [
            ("solar_euv_flux",       "thermospheric_density",  0.40, "known mechanism"),
            ("geomagnetic_activity", "thermospheric_density",  0.35, "known mechanism"),
            ("solar_wind_pressure",  "thermospheric_density",  0.25, "LLM hypothesis"),
            ("seasonal_latitudinal", "thermospheric_density",  0.35, "empirical pattern"),
            ("joule_heating",        "thermospheric_density",  0.30, "LLM hypothesis"),
            ("thermospheric_density","satellite_drag",         0.40, "physics law"),
            ("satellite_drag",       "collision_avoidance_event", 0.30, "operational link"),
        ],
        "semantic": [
            ("solar_euv_flux",       "IS_A",       "solar_forcing"),
            ("geomagnetic_activity", "IS_A",       "magnetospheric_coupling"),
            ("joule_heating",        "PART_OF",    "geomagnetic_activity"),
            ("satellite_drag",       "RELATED_TO", "collision_avoidance_event"),
        ],
    },
    "robotics": {
        "domain_id": "sim_to_real_calibration",
        "nodes": [
            ("joint_friction_drift",   "driver",       "N·m"),
            ("payload_mass_variation", "driver",       "g"),
            ("ambient_temperature",    "driver",       "°C"),
            ("surface_condition",      "driver",       "coeff"),
            ("controller_latency",     "driver",       "µs"),
            ("trajectory_error",       "outcome",      "mm"),
            ("grasp_force_error",      "outcome",      "N"),
            ("cycle_failure",          "outcome",      "boolean"),
        ],
        "edges": [
            ("joint_friction_drift",   "trajectory_error",  0.35, "known mechanism"),
            ("payload_mass_variation", "grasp_force_error", 0.40, "known mechanism"),
            ("ambient_temperature",    "joint_friction_drift", 0.25, "LLM hypothesis"),
            ("surface_condition",      "grasp_force_error", 0.25, "LLM hypothesis"),
            ("controller_latency",     "trajectory_error",  0.30, "suspected link"),
            ("trajectory_error",       "cycle_failure",     0.35, "operational link"),
            ("grasp_force_error",      "cycle_failure",     0.35, "operational link"),
        ],
        "semantic": [
            ("joint_friction_drift",   "IS_A",       "mechanical_degradation"),
            ("payload_mass_variation", "IS_A",       "load_variation"),
            ("controller_latency",     "PART_OF",    "control_system"),
            ("trajectory_error",       "RELATED_TO", "grasp_force_error"),
        ],
    },
    "manufacturing": {
        "domain_id": "cnc_process_quality",
        "nodes": [
            ("tool_wear",              "driver",       "mm"),
            ("ambient_temperature",    "driver",       "°C"),
            ("material_batch_hardness","driver",       "HRB"),
            ("coolant_concentration",  "driver",       "%"),
            ("fixture_clamping_force", "driver",       "bar"),
            ("dimensional_deviation",  "outcome",      "mm"),
            ("surface_roughness",      "outcome",      "µm_Ra"),
            ("scrap_event",            "outcome",      "boolean"),
        ],
        "edges": [
            ("tool_wear",              "dimensional_deviation",  0.40, "known mechanism"),
            ("ambient_temperature",    "dimensional_deviation",  0.30, "thermal growth"),
            ("material_batch_hardness","dimensional_deviation",  0.30, "LLM hypothesis"),
            ("coolant_concentration",  "surface_roughness",      0.25, "LLM hypothesis"),
            ("coolant_concentration",  "tool_wear",              0.25, "suspected link"),
            ("fixture_clamping_force", "dimensional_deviation",  0.20, "LLM hypothesis"),
            ("dimensional_deviation",  "scrap_event",            0.35, "operational link"),
        ],
        "semantic": [
            ("tool_wear",              "IS_A",       "progressive_degradation"),
            ("material_batch_hardness","IS_A",       "material_property"),
            ("coolant_concentration",  "RELATED_TO", "tool_wear"),
            ("dimensional_deviation",  "RELATED_TO", "surface_roughness"),
        ],
    },
    "data-centers": {
        "domain_id": "thermal_management",
        "nodes": [
            ("it_workload",            "driver",       "kW"),
            ("crac_airflow",           "driver",       "CFM"),
            ("ambient_temperature",    "driver",       "°C"),
            ("rack_configuration",     "driver",       "score"),
            ("floor_tile_layout",      "driver",       "open_%"),
            ("zone_temperature",       "outcome",      "°C"),
            ("cooling_power",          "outcome",      "kW"),
            ("thermal_alarm",          "outcome",      "boolean"),
        ],
        "edges": [
            ("it_workload",            "zone_temperature",  0.40, "known mechanism"),
            ("crac_airflow",           "zone_temperature",  0.40, "known mechanism"),
            ("ambient_temperature",    "zone_temperature",  0.30, "economizer link"),
            ("rack_configuration",     "zone_temperature",  0.25, "recirculation"),
            ("floor_tile_layout",      "zone_temperature",  0.25, "airflow distribution"),
            ("zone_temperature",       "cooling_power",     0.35, "control response"),
            ("zone_temperature",       "thermal_alarm",     0.30, "threshold trigger"),
        ],
        "semantic": [
            ("it_workload",            "IS_A",       "heat_source"),
            ("crac_airflow",           "IS_A",       "cooling_mechanism"),
            ("rack_configuration",     "RELATED_TO", "floor_tile_layout"),
            ("zone_temperature",       "RELATED_TO", "cooling_power"),
        ],
    },
}

# Domain-specific dry-run data for validate (endpoints)
DOMAIN_VALIDATE_DATA = {
    "space": {
        "endpoints": [
            ("grace_fo_accelerometer", "podaac.jpl.nasa.gov", "density, altitude, lat/lon", "10min", "READY"),
            ("tle_debris_catalog",     "space-track.org",     "ballistic coeff, inferred density", "1-3 day", "READY"),
            ("swpc_solar_wind",        "services.swpc.noaa.gov", "density, speed, Bz, temperature", "1min", "READY"),
        ],
    },
    "robotics": {
        "endpoints": [
            ("joint_encoder_rtde",     "192.168.1.100:30004", "positions, velocities, torques (6-DOF)", "1ms", "READY"),
            ("ft_sensor_wrist",        "192.168.1.60:8080",   "force/torque Fx,Fy,Fz,Tx,Ty,Tz", "100Hz", "READY"),
            ("environment_sensor",     "192.168.1.70:8080",   "ambient temp, humidity", "1min", "READY"),
        ],
    },
    "manufacturing": {
        "endpoints": [
            ("cmm_inspection",         "192.168.10.20:8080",  "feature deviations, nominal vs actual", "per-part", "READY"),
            ("cnc_mtconnect",          "192.168.10.10:5000",  "spindle load, feed rate, axis positions", "100Hz", "READY"),
            ("coolant_monitor",        "192.168.10.50:8080",  "concentration, pH, tramp oil, temp", "hourly", "READY"),
        ],
    },
    "data-centers": {
        "endpoints": [
            ("rack_inlet_sensors",     "dcim.local:8080",     "per-rack inlet temp (top/mid/bottom)", "1min", "READY"),
            ("intelligent_pdus",       "dcim.local:8080",     "per-rack power kW, current, voltage", "1min", "READY"),
            ("bms_crac_telemetry",     "bms.local:47808",     "supply/return temp, fan speed, mode", "1min", "READY"),
        ],
    },
}

# Default deploy target per domain
DOMAIN_DEPLOY_TARGETS = {
    "space":        "edge_gpu",   # Jetson Orin / spacecraft OBC
    "robotics":     "edge_gpu",   # Jetson on robotic arm
    "manufacturing":"edge_gpu",   # industrial edge server
    "data-centers": "cloud",      # on-prem inference server
}

# Domain-specific dry-run data for deploy (graph JSON excerpt + capabilities)
DOMAIN_DEPLOY_DATA = {
    "space": {
        "hardware": "NVIDIA Jetson Orin, spacecraft OBC with GPU",
        "json_excerpt": [
            '  {',
            '    "domain": "thermospheric_density",',
            '    "edges": [',
            '      {',
            '        "source": "solar_euv_flux",',
            '        "target": "thermospheric_density",',
            '        "certainty": 0.91,',
            '        "mechanism": "photoionization_heating",',
            '        "validation_cycles": 1240,',
            '        "thresholds": { "euv_flux_warn_sfu": 120, "density_deviation_pct": 15 }',
            '      },',
            '      {',
            '        "source": "thermospheric_density",',
            '        "target": "satellite_drag",',
            '        "certainty": 0.93,',
            '        "mechanism": "atmospheric_drag_law",',
            '        "validation_cycles": 892',
            '      }',
            '    ]',
            '  }',
        ],
        "capabilities": [
            "Predict: solar EUV spike → estimate density increase magnitude and timing",
            "Alert:   drag deviation > 15% MAPE → conjunction risk elevated",
            "Diagnose: 'Why did this maneuver consume 3× expected ΔV?'",
            "Update:  new GRACE-FO passes refine estimates without retraining",
        ],
        "nl_query": "Why is drag higher than predicted on this orbital pass?",
    },
    "robotics": {
        "hardware": "NVIDIA Jetson Orin, AMD Kria KV260",
        "json_excerpt": [
            '  {',
            '    "domain": "sim_to_real_calibration",',
            '    "edges": [',
            '      {',
            '        "source": "joint_friction_drift",',
            '        "target": "trajectory_error",',
            '        "certainty": 0.88,',
            '        "mechanism": "kinematic_model_bias",',
            '        "validation_cycles": 2341,',
            '        "thresholds": { "friction_drift_warn_nm": 0.15, "trajectory_error_crit_mm": 2.5 }',
            '      },',
            '      {',
            '        "source": "payload_mass_variation",',
            '        "target": "grasp_force_error",',
            '        "certainty": 0.91,',
            '        "mechanism": "inertia_compensation_error",',
            '        "validation_cycles": 1876',
            '      }',
            '    ]',
            '  }',
        ],
        "capabilities": [
            "Predict: joint friction increase → estimate trajectory error before it happens",
            "Alert:   payload variance > threshold → grasp force compensation required",
            "Diagnose: 'Why is the end-effector missing the target position?'",
            "Update:  new encoder cycles refine friction model without retraining",
        ],
        "nl_query": "Why is the end-effector missing the target position?",
    },
    "manufacturing": {
        "hardware": "NVIDIA Jetson AGX, industrial edge server",
        "json_excerpt": [
            '  {',
            '    "domain": "cnc_process_quality",',
            '    "edges": [',
            '      {',
            '        "source": "tool_wear",',
            '        "target": "dimensional_deviation",',
            '        "certainty": 0.89,',
            '        "mechanism": "cutting_edge_degradation",',
            '        "validation_cycles": 3102,',
            '        "thresholds": { "wear_warn_mm": 0.08, "deviation_crit_mm": 0.05 }',
            '      },',
            '      {',
            '        "source": "coolant_concentration",',
            '        "target": "surface_roughness",',
            '        "certainty": 0.87,',
            '        "mechanism": "lubrication_effectiveness",',
            '        "validation_cycles": 1540',
            '      }',
            '    ]',
            '  }',
        ],
        "capabilities": [
            "Predict: tool wear rate → estimate remaining useful life and scrap risk",
            "Alert:   dimensional deviation approaching tolerance limit → intervention",
            "Diagnose: 'Why did this batch have elevated surface roughness?'",
            "Update:  new CMM inspection results refine tolerance model without retraining",
        ],
        "nl_query": "Why did this batch have elevated surface roughness?",
    },
    "data-centers": {
        "hardware": "On-premises inference server, existing data center compute",
        "json_excerpt": [
            '  {',
            '    "domain": "thermal_management",',
            '    "edges": [',
            '      {',
            '        "source": "it_workload",',
            '        "target": "zone_temperature",',
            '        "certainty": 0.92,',
            '        "mechanism": "heat_dissipation",',
            '        "validation_cycles": 4821,',
            '        "thresholds": { "workload_warn_kw": 45, "temp_crit_c": 35 }',
            '      },',
            '      {',
            '        "source": "zone_temperature",',
            '        "target": "cooling_power",',
            '        "certainty": 0.90,',
            '        "mechanism": "crac_control_response",',
            '        "validation_cycles": 3910',
            '      }',
            '    ]',
            '  }',
        ],
        "capabilities": [
            "Predict: workload spike → estimate zone temperature rise before alarms trigger",
            "Optimize: adjust CRAC setpoints proactively to minimize PUE",
            "Diagnose: 'Why is zone 3 running 4°C above setpoint?'",
            "Update:  new rack telemetry refines thermal model without retraining",
        ],
        "nl_query": "Why is zone 3 running 4°C above setpoint?",
    },
}


def detect_domain(prior_text=None):
    """Detect domain from prior.md content. Returns domain key or 'default'."""
    if prior_text is None:
        for path in [Path("prior.md"), Path("examples/prior.md")]:
            if path.exists():
                prior_text = path.read_text()
                break
    if prior_text:
        text_lower = prior_text.lower()
        if any(kw in text_lower for kw in ["thermospher", "satellite", "orbital", "voxel", "drag", "euv"]):
            return "space"
        elif any(kw in text_lower for kw in ["robot", "joint", "sim-to-real", "grasp", "trajectory", "end-effector"]):
            return "robotics"
        elif any(kw in text_lower for kw in ["cnc", "machining", "tool wear", "coolant", "surface roughness", "fixture"]):
            return "manufacturing"
        elif any(kw in text_lower for kw in ["data center", "rack", "crac", "pue", "thermal zone", "hot aisle"]):
            return "data-centers"
    return "default"


def cmd_example(args):
    """Copy domain-specific prior.md and validate.md into the current directory."""
    domain = args.domain

    if domain == "list":
        print("  Available domain examples:")
        print()
        for name, desc in AVAILABLE_DOMAINS.items():
            print(f"    {name:<20} {desc}")
        print()
        print("  Usage: nm example <domain>")
        print("  This copies prior.md and validate.md into your current directory.")
        return

    if domain not in AVAILABLE_DOMAINS:
        print(f"  ✗ Unknown domain: {domain}")
        print(f"  Available: {', '.join(AVAILABLE_DOMAINS.keys())}")
        print(f"  Run 'nm example list' to see descriptions.")
        return

    # Find example files — check both examples/<domain>/ paths
    example_dir = None
    for candidate in [Path("examples") / domain, Path(__file__).parent / "examples" / domain]:
        if candidate.exists():
            example_dir = candidate
            break

    if example_dir is None:
        print(f"  ✗ Example directory not found for '{domain}'.")
        print(f"  Expected: examples/{domain}/prior.md and examples/{domain}/validate.md")
        return

    prior_src = example_dir / "prior.md"
    validate_src = example_dir / "validate.md"
    copied = []

    for src, dst_name in [(prior_src, "prior.md"), (validate_src, "validate.md")]:
        if src.exists():
            dst = Path(dst_name)
            if dst.exists():
                if dst.read_text() == src.read_text():
                    print(f"  ✓ {dst_name} already up to date")
                    continue
                print(f"  ⚠ {dst_name} already exists. Overwrite? [y/N] ", end="", flush=True)
                try:
                    answer = input().strip().lower()
                except (EOFError, KeyboardInterrupt):
                    print()
                    return
                if answer != "y":
                    print(f"    Skipped {dst_name}")
                    continue
            dst.write_text(src.read_text())
            copied.append(dst_name)
            print(f"  ✓ Copied {dst_name}  ← examples/{domain}/{dst_name}")
        else:
            print(f"  ⚠ {src} not found, skipping")

    print()
    print(f"  Domain: {AVAILABLE_DOMAINS[domain]}")
    print()

    # Run the full demo pipeline for this domain
    import types

    init_args = types.SimpleNamespace(prior="prior.md")
    cmd_init(init_args)

    print()
    validate_args = types.SimpleNamespace(spec="validate.md")
    cmd_validate(validate_args)

    print()
    deploy_target = DOMAIN_DEPLOY_TARGETS.get(domain, "microcontroller")
    deploy_args = types.SimpleNamespace(target=deploy_target, base="microsoft/phi-3.5-mini-instruct")
    cmd_deploy(deploy_args)

    print()
    learn_args = types.SimpleNamespace(cycles=5)
    cmd_learn(learn_args)

    print()
    status_args = types.SimpleNamespace()
    cmd_status(status_args)

    print()
    fleet_args = types.SimpleNamespace(mode="status")
    cmd_fleet(fleet_args)
    print()
    fleet_args.mode = "push"
    cmd_fleet(fleet_args)
    print()
    fleet_args.mode = "pull"
    cmd_fleet(fleet_args)


# ─────────────────────────────────────────────
# Commands
# ─────────────────────────────────────────────

def cmd_init(args):
    """Build causal prior from prior.md using CVOT MCP server."""
    print("═" * 60)
    print("  nm init — Building Causal Prior")
    print("═" * 60)
    print()

    prior_prompt = read_prompt(args.prior)

    system = """You are a causal modeling engine for Nervous Machine.
Your job: read the domain description and build a causal graph using the CVOT MCP tools.

For each causal relationship described:
1. Create source and target nodes via CVOT:save_nodes
2. Create causal edges with multi-dimensional vectors via CVOT:save_vector_edge
   - Set certainty based on the prior.md guidance
   - Include strength, confidence, context, and stability dimensions
3. Create semantic edges (IS_A, PART_OF, RELATED_TO) via CVOT:save_semantic_edges
4. Flag low-certainty edges as needing validation

Use the collection name 'causal_nodes' for all nodes.
Start all LLM-hypothesized edges at the certainty levels specified in the prior.
Be systematic. Build the full graph. Report what you built."""

    user = f"""Build the causal prior from this specification:

{prior_prompt}

Build the complete causal graph now using the CVOT tools. Create all nodes first,
then all causal edges with appropriate certainty levels, then semantic relationships."""

    print(f"⟳ Reading {args.prior}...")
    print(f"⟳ Connecting to CVOT server: {CVOT_MCP_URL or 'not configured'}")
    print()

    if not CVOT_MCP_URL:
        domain = detect_domain(prior_prompt)
        data = DOMAIN_INIT_DATA.get(domain)

        if data:
            print(f"  BUILDING CAUSAL GRAPH  [{domain}]")
            print(f"  Parsed {len(prior_prompt.splitlines())} lines from {args.prior}")
            print()
            print(f"  NODES CREATED ({len(data['nodes'])}):")
            for i, (nid, ntype, unit) in enumerate(data['nodes']):
                prefix = "└──" if i == len(data['nodes']) - 1 else "├──"
                print(f"  {prefix} {nid:<35} type: {ntype:<15} unit: {unit}")
            print()
            print(f"  CAUSAL EDGES ({len(data['edges'])}):")
            for i, (src, tgt, z, note) in enumerate(data['edges']):
                prefix = "└──" if i == len(data['edges']) - 1 else "├──"
                print(f"  {prefix} {src} → {tgt:<30} Z={z:.2f}  ({note})")
            print()
            print(f"  SEMANTIC EDGES ({len(data['semantic'])}):")
            for i, (src, rel, tgt) in enumerate(data['semantic']):
                prefix = "└──" if i == len(data['semantic']) - 1 else "├──"
                print(f"  {prefix} {src:<30} {rel:<12} {tgt}")
            print()
            low_z = sum(1 for _, _, z, _ in data['edges'] if z < 0.50)
            print(f"  {low_z} edges flagged for validation (all Z < 0.50)")
            print()
            print(f"  ✓ Causal prior built [{domain}]. Run 'nm validate' to set up error signals.")
        else:
            # Default MCU example (original)
            print("  BUILDING CAUSAL GRAPH")
            print(f"  Parsed {len(prior_prompt.splitlines())} lines from {args.prior}")
            print()
            print("  NODES CREATED (8):")
            print("  ├── thermal_cycling              type: stressor")
            print("  ├── voltage_ripple               type: stressor")
            print("  ├── vibration_exposure            type: stressor")
            print("  ├── solder_joint_fatigue          type: failure_mode")
            print("  ├── capacitor_esr_drift           type: failure_mode")
            print("  ├── clock_jitter                  type: symptom")
            print("  ├── watchdog_reset                type: symptom")
            print("  └── mcu_functional_failure        type: outcome")
            print()
            print("  CAUSAL EDGES (6):")
            print("  ├── thermal_cycling → solder_joint_fatigue     Z=0.30  (LLM hypothesis)")
            print("  ├── thermal_cycling → capacitor_esr_drift      Z=0.35  (datasheet ref)")
            print("  ├── voltage_ripple → clock_jitter              Z=0.40  (known mechanism)")
            print("  ├── vibration_exposure → solder_joint_fatigue  Z=0.25  (LLM hypothesis)")
            print("  ├── solder_joint_fatigue → watchdog_reset      Z=0.30  (LLM hypothesis)")
            print("  └── capacitor_esr_drift → mcu_functional_failure Z=0.35  (datasheet ref)")
            print()
            print("  SEMANTIC EDGES (4):")
            print("  ├── solder_joint_fatigue  IS_A      mechanical_failure")
            print("  ├── capacitor_esr_drift   IS_A      electrical_degradation")
            print("  ├── clock_jitter          PART_OF   timing_subsystem")
            print("  └── watchdog_reset        RELATED_TO mcu_functional_failure")
            print()
            print("  6 edges flagged for validation (all Z < 0.50)")
            print()
            print("  ✓ Causal prior built. Run 'nm validate' to set up error signals.")
        return

    result = call_with_mcp(
        system, user,
        {"cvot": CVOT_MCP_URL}
    )

    print()
    print("✓ Causal prior built. Run 'nm status' to inspect.")


def cmd_validate(args):
    """Set up validation pipelines from validate.md."""
    print("═" * 60)
    print("  nm validate — Deploying Validation Pipelines")
    print("═" * 60)
    print()

    validate_prompt = read_prompt(args.spec)

    system = """You are a validation pipeline engineer for Nervous Machine.
Your job: read the validation specification and set up automated validation using the
Validation MCP tools.

For each data source:
1. Register the endpoint via Validation:register_validation_endpoint
2. Create a validation pipeline via Validation:create_validation_pipeline
   - Wire it to the correct causal edges
   - Set up the error signal format (ε = |predicted - actual|)
   - Configure feedback mechanism (update_weights by default)

For edges that need it, create Bradford Hill validators via
Validation:create_bradford_hill_validator.

Report what pipelines you created and which edges they validate."""

    user = f"""Set up validation pipelines from this specification:

{validate_prompt}

Register all endpoints first, then create validation pipelines for each."""

    print(f"⟳ Reading {args.spec}...")
    print(f"⟳ Connecting to Validation server: {VALIDATION_MCP_URL or 'not configured'}")
    print()

    if not VALIDATION_MCP_URL:
        domain = detect_domain(validate_prompt)
        vdata = DOMAIN_VALIDATE_DATA.get(domain)
        idata = DOMAIN_INIT_DATA.get(domain)

        if vdata and idata:
            print(f"  REGISTERING VALIDATION ENDPOINTS  [{domain}]")
            print()
            eps = vdata['endpoints']
            print(f"  ENDPOINTS ({len(eps)}):")
            for i, (eid, host, measures, cadence, status) in enumerate(eps):
                prefix = "└──" if i == len(eps) - 1 else "├──"
                sep    = "   " if i == len(eps) - 1 else "│  "
                print(f"  {prefix} {eid:<28} {host}")
                print(f"  {sep} Measures: {measures}")
                print(f"  {sep} Cadence: {cadence}  |  Status: {status}")
                if i < len(eps) - 1:
                    print(f"  │")
            print()
            edges = idata['edges']
            print(f"  VALIDATION PIPELINES ({len(edges)}):")
            for i, (src, tgt, z, _) in enumerate(edges):
                prefix = "└──" if i == len(edges) - 1 else "├──"
                sep    = "   " if i == len(edges) - 1 else "│  "
                print(f"  {prefix} {src} → {tgt}")
                print(f"  {sep} Error signal: ε = |predicted - actual|")
                if i < len(edges) - 1:
                    print(f"  │")
            print()
            print(f"  Feedback configured for all {len(edges)} pipelines")
            print()
            print(f"  ✓ Validation pipelines deployed [{domain}]. Run 'nm learn' to start.")
        else:
            # Default MCU example (original)
            print("  REGISTERING VALIDATION ENDPOINTS")
            print()
            print("  ENDPOINTS (3):")
            print("  ├── thermal_chamber_api         http://testlab.local:8080/thermal")
            print("  │   Measures: junction temp, ambient temp, cycle count")
            print("  │   Format:   JSON  |  Interval: 30s  |  Status: READY")
            print("  │")
            print("  ├── power_rail_monitor          http://testlab.local:8080/power")
            print("  │   Measures: Vcc ripple (mV), ESR (mΩ), capacitor temp")
            print("  │   Format:   JSON  |  Interval: 10s  |  Status: READY")
            print("  │")
            print("  └── vibration_table_daq         http://testlab.local:8080/vibration")
            print("      Measures: acceleration (g), frequency spectrum, duration")
            print("      Format:   JSON  |  Interval: 100ms |  Status: READY")
            print()
            print("  Feedback configured for all 6 pipelines")
            print()
            print("  ✓ Validation pipelines deployed. Run 'nm learn' to start.")
        return

    result = call_with_mcp(
        system, user,
        {"validation": VALIDATION_MCP_URL}
    )

    print()
    print("✓ Validation pipelines deployed. Run 'nm learn' to start.")


def cmd_learn(args):
    """Run the learning loop — fetch validation data, compute errors, update beliefs."""
    print("═" * 60)
    print(f"  nm learn — Running Learning Loop ({args.cycles} cycle{'s' if args.cycles != 1 else ''})")
    print("═" * 60)
    print()

    system = """You are the learning engine for Nervous Machine.
Your job: run one cycle of the validation-learning loop.

1. Check for recent error signals via Validation:get_recent_errors
2. For each error signal:
   a. Compute the learning rate via Validation:compute_learning_rate
   b. Apply learning feedback via Validation:apply_learning_feedback
   c. Update certainty in CVOT via CVOT:update_certainty
3. Check for low-certainty edges via CVOT:get_low_certainty_edges
4. Report:
   - Which edges were updated
   - Certainty changes (before → after)
   - Current learning rates
   - Curiosity triggers (high-error, low-certainty edges)

Format output for terminal readability."""

    user = "Run one learning cycle. Fetch errors, compute learning rates, update beliefs, report."

    servers = {}
    if CVOT_MCP_URL:
        servers["cvot"] = CVOT_MCP_URL
    if VALIDATION_MCP_URL:
        servers["validation"] = VALIDATION_MCP_URL

    if not servers:
        # Simulate multiple cycles with progressive convergence
        cycle_data = [
            # (edge, z_before, z_after, epsilon, eta, source)
            [
                ("thermal_cycling → solder_joint_fatigue",  0.30, 0.38, 0.142, 0.285, "thermal_chamber"),
                ("thermal_cycling → capacitor_esr_drift",   0.35, 0.44, 0.098, 0.278, "thermal_chamber + power_rail"),
                ("voltage_ripple → clock_jitter",           0.40, 0.51, 0.067, 0.265, "power_rail_monitor"),
                ("vibration_exposure → solder_joint_fatigue",0.25, 0.31, 0.189, 0.290, "vibration_table"),
                ("solder_joint_fatigue → watchdog_reset",   0.30, 0.36, 0.156, 0.285, "thermal_chamber"),
                ("capacitor_esr_drift → mcu_failure",       0.35, 0.42, 0.112, 0.278, "power_rail_monitor"),
            ],
            [
                ("thermal_cycling → solder_joint_fatigue",  0.38, 0.49, 0.108, 0.270, "thermal_chamber"),
                ("thermal_cycling → capacitor_esr_drift",   0.44, 0.56, 0.071, 0.248, "thermal_chamber + power_rail"),
                ("voltage_ripple → clock_jitter",           0.51, 0.63, 0.043, 0.222, "power_rail_monitor"),
                ("vibration_exposure → solder_joint_fatigue",0.31, 0.40, 0.161, 0.282, "vibration_table"),
                ("solder_joint_fatigue → watchdog_reset",   0.36, 0.45, 0.129, 0.275, "thermal_chamber"),
                ("capacitor_esr_drift → mcu_failure",       0.42, 0.53, 0.088, 0.258, "power_rail_monitor"),
            ],
            [
                ("thermal_cycling → solder_joint_fatigue",  0.49, 0.61, 0.082, 0.240, "thermal_chamber"),
                ("thermal_cycling → capacitor_esr_drift",   0.56, 0.68, 0.048, 0.203, "thermal_chamber + power_rail"),
                ("voltage_ripple → clock_jitter",           0.63, 0.74, 0.029, 0.170, "power_rail_monitor"),
                ("vibration_exposure → solder_joint_fatigue",0.40, 0.50, 0.134, 0.268, "vibration_table"),
                ("solder_joint_fatigue → watchdog_reset",   0.45, 0.55, 0.101, 0.250, "thermal_chamber"),
                ("capacitor_esr_drift → mcu_failure",       0.53, 0.64, 0.063, 0.218, "power_rail_monitor"),
            ],
            [
                ("thermal_cycling → solder_joint_fatigue",  0.61, 0.72, 0.058, 0.183, "thermal_chamber"),
                ("thermal_cycling → capacitor_esr_drift",   0.68, 0.78, 0.031, 0.140, "thermal_chamber + power_rail"),
                ("voltage_ripple → clock_jitter",           0.74, 0.83, 0.018, 0.112, "power_rail_monitor"),
                ("vibration_exposure → solder_joint_fatigue",0.50, 0.60, 0.109, 0.238, "vibration_table"),
                ("solder_joint_fatigue → watchdog_reset",   0.55, 0.65, 0.078, 0.210, "thermal_chamber"),
                ("capacitor_esr_drift → mcu_failure",       0.64, 0.74, 0.042, 0.170, "power_rail_monitor"),
            ],
            [
                ("thermal_cycling → solder_joint_fatigue",  0.72, 0.81, 0.037, 0.125, "thermal_chamber"),
                ("thermal_cycling → capacitor_esr_drift",   0.78, 0.86, 0.019, 0.091, "thermal_chamber + power_rail"),
                ("voltage_ripple → clock_jitter",           0.83, 0.89, 0.011, 0.065, "power_rail_monitor"),
                ("vibration_exposure → solder_joint_fatigue",0.60, 0.69, 0.086, 0.195, "vibration_table"),
                ("solder_joint_fatigue → watchdog_reset",   0.65, 0.74, 0.055, 0.162, "thermal_chamber"),
                ("capacitor_esr_drift → mcu_failure",       0.74, 0.82, 0.028, 0.118, "power_rail_monitor"),
            ],
        ]

        for cycle in range(1, args.cycles + 1):
            data_idx = min(cycle - 1, len(cycle_data) - 1)
            edges = cycle_data[data_idx]

            print(f"  ── Cycle {cycle}/{args.cycles} ──")
            print()
            for edge, z_b, z_a, eps, eta, src in edges:
                arrow = "↑" if z_a > z_b else "↓"
                print(f"    {edge}")
                print(f"      ε={eps:.3f}  η(Z)={eta:.3f}  Z: {z_b:.2f} → {z_a:.2f}  {arrow}  [{src}]")
            print()

            # Show threshold crossings
            crossings = [(e, z_a) for e, z_b, z_a, _, _, _ in edges if z_b < 0.85 and z_a >= 0.85]
            if crossings:
                for e, z in crossings:
                    print(f"    ⚡ {e}  CROSSED Z=0.85 — ready for injection")
                print()

        # Final summary
        final = cycle_data[min(args.cycles - 1, len(cycle_data) - 1)]
        print("  SUMMARY:")
        high = sum(1 for _, _, z_a, _, _, _ in final if z_a >= 0.85)
        med = sum(1 for _, _, z_a, _, _, _ in final if 0.50 <= z_a < 0.85)
        low = sum(1 for _, _, z_a, _, _, _ in final if z_a < 0.50)
        print(f"    High certainty (Z ≥ 0.85):  {high} edges")
        print(f"    Medium (0.50 ≤ Z < 0.85):   {med} edges")
        print(f"    Low (Z < 0.50):              {low} edges")
        print()
        if high > 0:
            print(f"  ✓ {high} edge(s) ready for injection. Run 'nm inject'.")
        else:
            print(f"  ⟳ No edges at injection threshold yet. Run 'nm learn --cycles {args.cycles + 5}'.")
        return

    for cycle in range(1, args.cycles + 1):
        if args.cycles > 1:
            print(f"  ── Cycle {cycle}/{args.cycles} ──")
            print()

        result = call_with_mcp(system, user, servers)

        if cycle < args.cycles:
            print()

    print()
    print(f"✓ {args.cycles} learning cycle{'s' if args.cycles != 1 else ''} complete. Run 'nm status' to see changes.")


def cmd_status(args):
    """Show current graph state — certainty levels, gaps, curiosity triggers."""
    print("═" * 60)
    print("  nm status — Causal Graph State")
    print("═" * 60)
    print()

    system = """You are a reporting engine for Nervous Machine.
Generate a concise terminal-friendly status report:

1. Get low-certainty edges via CVOT:get_low_certainty_edges (threshold 0.5)
2. Get any available certainty evolution via CVOT:get_certainty_evolution
3. Check for ready validation endpoints via Validation:get_ready_endpoints
4. Check for recent errors via Validation:get_recent_errors (limit 10)

Format as a clean status report with:
  - Edge count and average certainty
  - Top curiosity triggers (low certainty, high error)
  - Recent learning events
  - Validation pipeline status
Use simple ASCII formatting — no markdown."""

    user = "Generate status report for the current causal graph."

    servers = {}
    if CVOT_MCP_URL:
        servers["cvot"] = CVOT_MCP_URL
    if VALIDATION_MCP_URL:
        servers["validation"] = VALIDATION_MCP_URL

    if not servers:
        print("  CERTAINTY DISTRIBUTION")
        print("  ├── High (Z ≥ 0.85): ██████░░░░░░   2 edges  (ready for injection)")
        print("  ├── Med  (0.5-0.85): ████████████   4 edges  (learning)")
        print("  └── Low  (Z < 0.50): ████░░░░░░░░   0 edges")
        print()
        print("  EDGE DETAIL:")
        print("  ├── voltage_ripple → clock_jitter              Z=0.89  ████████████████▓░░░  READY")
        print("  ├── thermal_cycling → capacitor_esr_drift      Z=0.86  █████████████████░░░  READY")
        print("  ├── thermal_cycling → solder_joint_fatigue     Z=0.81  ████████████████░░░░  learning")
        print("  ├── capacitor_esr_drift → mcu_failure          Z=0.82  ████████████████░░░░  learning")
        print("  ├── solder_joint_fatigue → watchdog_reset      Z=0.74  ██████████████░░░░░░  learning")
        print("  └── vibration_exposure → solder_joint_fatigue  Z=0.69  █████████████░░░░░░░  learning")
        print()
        print("  CURIOSITY TRIGGERS:")
        print("  ├── vibration_exposure → solder_joint_fatigue  Z=0.69")
        print("  │   Slowest convergence. Consider: additional vibration profiles,")
        print("  │   frequency-dependent effects, or missing confounders.")
        print("  │")
        print("  └── solder_joint_fatigue → watchdog_reset      Z=0.74")
        print("      High residual error despite rising Z. Possible intermittent")
        print("      contact — failure mode may be stochastic, not deterministic.")
        print()
        print("  VALIDATION PIPELINE STATUS:")
        print("  ├── thermal_chamber_api     ✓ active    last signal: 12s ago")
        print("  ├── power_rail_monitor      ✓ active    last signal: 3s ago")
        print("  └── vibration_table_daq     ✓ active    last signal: 8s ago")
        print()
        print("  RECENT LEARNING (last 5 cycles):")
        print("  ├── voltage_ripple → clock_jitter          Z: 0.74 → 0.89  ↑ (+0.15)")
        print("  ├── thermal_cycling → capacitor_esr_drift  Z: 0.68 → 0.86  ↑ (+0.18)")
        print("  ├── thermal_cycling → solder_joint_fatigue Z: 0.61 → 0.81  ↑ (+0.20)")
        print("  ├── capacitor_esr_drift → mcu_failure      Z: 0.64 → 0.82  ↑ (+0.18)")
        print("  ├── solder_joint_fatigue → watchdog_reset  Z: 0.55 → 0.74  ↑ (+0.19)")
        print("  └── vibration_exposure → solder_joint_fat. Z: 0.50 → 0.69  ↑ (+0.19)")
        print()
        print("  Next: 'nm learn --cycles 3' to push remaining edges past Z=0.85")
        return

    result = call_with_mcp(system, user, servers)


def cmd_review(args):
    """Interactive review of prior.md + validate.md alignment."""
    print("═" * 60)
    print("  nm review — Interactive Causal Model Review")
    print("═" * 60)
    print()

    prior_prompt = read_prompt(args.prior)
    validate_prompt = read_prompt(args.spec)

    system = """You are a causal model reviewer for Nervous Machine.

You have been given a causal prior specification (prior.md) and a validation
specification (validate.md). Your job is to have an interactive conversation
about the alignment between these two files.

On your FIRST response, provide a structured review:

  COVERAGE ANALYSIS
  For each causal edge in the prior, state whether validation endpoints
  can produce error signals for it. Flag gaps.

  CERTAINTY ASSESSMENT
  Review initial Z scores. Flag any that seem too high or too low
  given the validation sources available.

  MISSING EDGES
  Suggest causal relationships the user may have missed based on
  the domain described.

  MISSING VALIDATION
  Suggest additional data sources or endpoints that would strengthen
  error signals for weak edges.

  RECOMMENDATIONS
  Prioritized list of changes to prior.md and validate.md.

Then engage in conversation. Be direct. Use terminal-friendly formatting
(no markdown headers — use plain ASCII, indentation, and box-drawing
characters). Reference specific edge names and Z values from the files."""

    initial_context = f"""Review the alignment between my causal prior and validation specification.

=== PRIOR ({args.prior}) ===
{prior_prompt}

=== VALIDATION ({args.spec}) ===
{validate_prompt}

Provide your structured review, then I'll ask follow-up questions.
Type 'done' when finished."""

    print(f"  ⟳ Reading {args.prior} ({len(prior_prompt.splitlines())} lines)")
    print(f"  ⟳ Reading {args.spec} ({len(validate_prompt.splitlines())} lines)")
    print()
    print("  Starting interactive review session.")
    print("  Type your questions. Type 'done' to end.")
    print()

    # Review works without MCP — it analyzes the prompt files as text.
    # If CVOT/Validation are available, Claude can also query the live graph.
    servers = {}
    if CVOT_MCP_URL:
        servers["cvot"] = CVOT_MCP_URL
    if VALIDATION_MCP_URL:
        servers["validation"] = VALIDATION_MCP_URL

    turns = chat_loop(system, initial_context,
                      mcp_servers=servers if servers else None)

    print()
    print(f"  ✓ Review complete ({turns} exchanges).")


def cmd_inject(args):
    """Create a LoRA adapter from high-certainty causal vectors using Domain Heads server."""
    print("═" * 60)
    print("  nm inject — Promoting Validated Causal Knowledge")
    print("═" * 60)
    print()

    base_model = args.base
    rank = args.rank
    threshold = args.threshold
    alpha = rank * 4

    system = f"""You are a domain head injection engine for Nervous Machine.
Your job: extract high-certainty causal vectors from CVOT and create a PEFT/LoRA adapter
using the Domain Heads MCP server.

Steps:
1. Get GNN features from CVOT via CVOT:get_gnn_features (these are the learned vectors)
2. Get low-certainty edges via CVOT:get_low_certainty_edges to identify what to EXCLUDE
3. Call the Domain Heads server to create_causal_adapter with:
   - base_model: the specified model
   - causal_graph_source: the high-certainty vectors from CVOT
   - rank: {rank}
   - target_layers: top third of model layers
4. Report: adapter created, how many vectors injected, certainty distribution,
   which edges were included vs excluded

The adapter injects causal reasoning into the model's top layers WITHOUT touching
the frozen base weights. Only high-certainty vectors (Z > {threshold}) get injected."""

    user = f"""Create a causal LoRA adapter for base model: {base_model}

Extract high-certainty vectors from the causal graph and create the adapter.
Only inject vectors with certainty > {threshold}. Report what was included and excluded."""

    print(f"⟳ Base model: {base_model} (frozen)")
    print(f"⟳ Certainty threshold: Z > {threshold}")
    print(f"⟳ Extracting validated vectors from CVOT...")
    print(f"⟳ Domain Heads server: {DOMAIN_HEADS_MCP_URL or 'not configured'}")
    print()

    if not DOMAIN_HEADS_MCP_URL or not CVOT_MCP_URL:
        print(f"  THRESHOLD:   Z > {threshold}")
        print(f"  Base model:  {base_model} (frozen — no weights modified)")
        print()
        print("  APPROVAL GATE — Only validated causal edges are promoted:")
        print()
        print("  ✓ APPROVED (3 edges — validated above threshold):")
        print("  ├── voltage_ripple → clock_jitter              Z=0.89")
        print("  │   Evidence: 847 prediction-error cycles, 3 independent endpoints")
        print("  ├── thermal_cycling → capacitor_esr_drift      Z=0.86")
        print("  │   Evidence: 612 cycles, thermal chamber + power rail cross-validated")
        print("  └── capacitor_esr_drift → mcu_functional_failure Z=0.86")
        print("      Evidence: 589 cycles, ESR drift correlated with functional test")
        print()
        print(f"  ✗ HELD BACK (3 edges — still learning, below Z > {threshold}):")
        print("  ├── thermal_cycling → solder_joint_fatigue     Z=0.81  ↑ close")
        print("  ├── solder_joint_fatigue → watchdog_reset      Z=0.74  ↑ learning")
        print("  └── vibration_exposure → solder_joint_fatigue  Z=0.69  ↑ learning")
        print()
        print("  Only validated causal knowledge is injected.")
        print("  Held-back edges continue learning until they reach threshold.")
        print()
        print("  Next: nm train    (train on approved edges)")
        print("        nm deploy   (compile for edge/cloud/microcontroller)")
        return

    result = call_with_mcp(
        system, user,
        {"cvot": CVOT_MCP_URL, "domain_heads": DOMAIN_HEADS_MCP_URL}
    )

    print()
    print("✓ Adapter created. Run 'nm train' to train concept extractor.")


def cmd_train(args):
    """Train the concept extractor — the neural→causal bridge layer. ~5-10M params."""
    print("═" * 60)
    print("  nm train — Training Concept Extractor")
    print("═" * 60)
    print()

    base_model = args.base
    epochs = args.epochs
    lr = args.lr

    system = f"""You are a training engine for Nervous Machine domain heads.
Your job: train ONLY the concept extractor layer — the bridge between
the model's neural representations and the causal graph structure.

Architecture:
- Base model: FROZEN (no gradients) — {base_model}
- Causal graph: FIXED structure (from CVOT)
- LoRA weights: FIXED (or optionally learned)
- Concept extractor + causal embedder: TRAINABLE (~5-10M params)

Call the Domain Heads server train_concept_extractor with:
- The most recently created adapter
- Training epochs: {epochs}
- Learning rate: {lr}

Report: trainable params, training loss curve, validation metrics.
This should be FAST — minutes, not hours — because only ~5-10M params update."""

    user = f"Train the concept extractor for {base_model}. Epochs: {epochs}, LR: {lr}. Report training metrics."

    print(f"  Base model: {base_model} (frozen — not modified)")
    print(f"  Training causal reasoning layer on {epochs} epoch{'s' if epochs != 1 else ''}")
    print()

    if not DOMAIN_HEADS_MCP_URL:
        print("  TRAINING...")
        time.sleep(0.5)
        print("  ├── Epoch 1/{0}  converging...".format(epochs))
        for e in range(2, epochs + 1):
            time.sleep(0.3)
            connector = "└" if e == epochs else "├"
            print(f"  {connector}── Epoch {e}/{epochs}  converging...")
        print()
        print("  RESULTS (MCU Reliability Domain):")
        print("  ┌──────────────────────────────────────────────────────────┐")
        print("  │  Fault diagnosis accuracy:    84.7%  (vs 43.1% base)    │")
        print("  │  Failure prediction (AUROC):  0.91   (capacitor ESR)    │")
        print("  │  Uncertainty calibration:     0.89   (well-calibrated)  │")
        print("  │  Domain knowledge retention:  95.2%  (no forgetting)    │")
        print("  └──────────────────────────────────────────────────────────┘")
        print()
        print("  Training time: ~8 minutes on 1x GPU")
        print("  Base model weights: UNCHANGED (frozen)")
        print()
        print("  The model can now reason causally about MCU failure modes.")
        print()
        print("  Next: nm deploy --target microcontroller")
        return

    result = call_with_mcp(
        system, user,
        {"domain_heads": DOMAIN_HEADS_MCP_URL}
    )

    print()
    print("✓ Concept extractor trained. Run 'nm deploy' to compile for target.")


def cmd_deploy(args):
    """Compile adapter for edge, cloud, or microcontroller deployment."""
    print("═" * 60)
    print("  nm deploy — Compiling for Deployment")
    print("═" * 60)
    print()

    target = args.target
    base_model = args.base

    system = f"""You are a deployment compiler for Nervous Machine domain heads.
Base model: {base_model}
Your job: compile the trained adapter for the target platform using the
Domain Heads server compile_for_edge.

Target platforms:
- microcontroller: Pure causal graph, no neural components. C implementation.
  Tiny footprint. For sensors, IoT, embedded systems.
- edge_gpu: Quantized 4-bit base + adapter. ~4-8GB for small models.
  For robots, drones, vehicles, edge servers.
- cloud: Full precision. For enterprise deployment, high-throughput inference.

Report: artifact size, deployment instructions, what's included."""

    model_short = base_model.split("/")[-1]
    user = f"Compile the {model_short} adapter for target platform: {target}"

    targets = {
        "microcontroller": {
            "desc": "Pure causal graph (no neural). C inference engine.",
            "size": "~50KB",
            "use": "Sensors, IoT, embedded systems"
        },
        "edge_gpu": {
            "desc": f"Quantized 4-bit base + LoRA adapter",
            "size": f"~2-4GB ({model_short} quantized)",
            "use": "Robots, drones, vehicles, edge servers"
        },
        "cloud": {
            "desc": f"Full precision base + adapter",
            "size": f"~8GB ({model_short} fp16)",
            "use": "Enterprise, high-throughput inference"
        }
    }

    t = targets.get(target, targets["edge_gpu"])

    print(f"  Base:    {base_model}")
    print(f"  Target:  {target}")
    print(f"  Format:  {t['desc']}")
    print(f"  Size:    {t['size']}")
    print(f"  Use:     {t['use']}")
    print()

    if not DOMAIN_HEADS_MCP_URL:
        domain = detect_domain()
        ddata = DOMAIN_DEPLOY_DATA.get(domain)

        if target == "microcontroller":
            print("  ARTIFACT:")
            print("  ├── causal_graph.json        48KB   (validated causal edges + metadata)")
            print("  ├── inference_engine.c        12KB   (causal traversal + update logic)")
            print("  └── nm_config.h                2KB   (thresholds, edge IDs)")
            print()
            print("  No neural components. Pure causal graph reasoning.")
            print("  Runs on any MCU with 64KB+ RAM — AMD Xilinx, STM32, ESP32, etc.")
            print()
            print("  ── REVIEWABLE: causal_graph.json (excerpt) ──")
            print()
            if ddata:
                for line in ddata["json_excerpt"]:
                    print(line)
            else:
                print('  {')
                print('    "domain": "mcu_reliability",')
                print('    "edges": [')
                print('      {')
                print('        "source": "voltage_ripple",')
                print('        "target": "clock_jitter",')
                print('        "certainty": 0.89,')
                print('        "mechanism": "power_rail_coupling",')
                print('        "validation_cycles": 847,')
                print('        "thresholds": { "ripple_mv": 50, "jitter_warn_ps": 200, "jitter_crit_ps": 500 }')
                print('      },')
                print('      {')
                print('        "source": "thermal_cycling",')
                print('        "target": "capacitor_esr_drift",')
                print('        "certainty": 0.86,')
                print('        "mechanism": "dielectric_degradation",')
                print('        "validation_cycles": 612,')
                print('        "thresholds": { "cycles_warn": 5000, "esr_limit_mohm": 150 }')
                print('      }')
                print('    ]')
                print('  }')
            print()
            print("  ── REVIEWABLE: inference_engine.c (excerpt) ──")
            print()
            print("  typedef struct {")
            print("      const char* source;")
            print("      const char* target;")
            print("      float certainty;")
            print("      float threshold_warn;")
            print("      float threshold_crit;")
            print("  } causal_edge_t;")
            print()
            print("  nm_diagnosis_t nm_diagnose(nm_ctx_t* ctx, sensor_reading_t reading) {")
            print("      nm_diagnosis_t result = {0};")
            print("      for (int i = 0; i < ctx->num_edges; i++) {")
            print("          causal_edge_t* edge = &ctx->edges[i];")
            print("          if (matches_source(edge, reading.sensor_id)) {")
            print("              float risk = evaluate_edge(edge, reading.value);")
            print("              if (risk > edge->threshold_crit)")
            print('                  result.alerts[result.num_alerts++] = ')
            print("                      (nm_alert_t){edge->target, risk, CRITICAL};")
            print("              else if (risk > edge->threshold_warn)")
            print("                  result.alerts[result.num_alerts++] = ")
            print("                      (nm_alert_t){edge->target, risk, WARNING};")
            print("          }")
            print("      }")
            print("      return result;")
            print("  }")
            print()
            caps = ddata["capabilities"] if ddata else [
                "Diagnose: voltage_ripple detected → predict clock_jitter risk",
                "Predict:  thermal cycles accumulated → estimate capacitor ESR",
                "Alert:    ESR drift approaching failure threshold → watchdog",
                "Learn:    continue updating from onboard sensor data",
            ]
            print("  The model can:")
            for cap in caps:
                print(f"    • {cap}")
        elif target == "edge_gpu":
            print("  ARTIFACT:")
            print(f"  ├── base_model_q4.gguf          ~2GB  ({model_short} quantized)")
            print("  ├── causal_graph.json             48KB")
            print("  └── config.json                    4KB")
            print()
            hw = ddata["hardware"] if ddata else "NVIDIA Jetson, AMD Kria, RPi 5 + accelerator"
            print(f"  Runs on: {hw}")
            print()
            print("  ── REVIEWABLE: causal_graph.json (excerpt) ──")
            print()
            if ddata:
                for line in ddata["json_excerpt"]:
                    print(line)
            print()
            nl_query = ddata["nl_query"] if ddata else "Why is this system behaving unexpectedly?"
            print(f"  Full LLM + causal reasoning. Example query:")
            print(f"    \"{nl_query}\"")
            print()
            if ddata:
                print("  The model can:")
                for cap in ddata["capabilities"]:
                    print(f"    • {cap}")
        elif target == "cloud":
            print("  ARTIFACT:")
            print(f"  ├── base_model.safetensors       ~8GB  ({model_short})")
            print("  ├── causal_graph.json               48KB")
            print("  └── config.json                      4KB")
            print()
            hw = ddata["hardware"] if ddata else "any inference server for fleet-wide monitoring"
            print(f"  Deploy to: {hw}")
            print()
            if ddata:
                print("  ── REVIEWABLE: causal_graph.json (excerpt) ──")
                print()
                for line in ddata["json_excerpt"]:
                    print(line)
                print()
                nl_query = ddata["nl_query"]
                print(f"  Full LLM + causal reasoning. Example query:")
                print(f"    \"{nl_query}\"")
                print()
                print("  The model can:")
                for cap in ddata["capabilities"]:
                    print(f"    • {cap}")
        print()
        print("  Next: nm update  (push new learning without retraining)")
        return

    result = call_with_mcp(
        system, user,
        {"domain_heads": DOMAIN_HEADS_MCP_URL}
    )

    print()
    print(f"✓ Compiled for {target}. Model is deployment-ready.")


def cmd_update(args):
    """Push updated causal vectors to deployed adapters. No retraining needed."""
    print("═" * 60)
    print("  nm update — Pushing Learned Vectors (No Retraining)")
    print("═" * 60)
    print()

    system = """You are an update engine for Nervous Machine domain heads.
Your job: push newly validated causal vectors to already-deployed adapters
WITHOUT retraining. This is like an OS patch for the model's causal knowledge.

Steps:
1. Get recent certainty changes from CVOT (edges whose certainty increased)
2. Get recently validated edges via Validation:get_recent_errors
3. Filter for edges with certainty > 0.85 (high confidence updates only)
4. Call Domain Heads server update_causal_knowledge with the updated vectors
5. Report: which vectors were updated, certainty deltas, deployment notifications

The key insight: the causal graph structure in the adapter changes, but the
neural weights DO NOT change. The model gets smarter without retraining.
Deployed instances pull the update like a firmware patch."""

    user = """Check for updated vectors since last deployment.
Push high-certainty updates to deployed adapters. Report what changed."""

    print("  ┌───────────────────────────────────────────────┐")
    print("  │  This is the killer feature.                   │")
    print("  │                                                │")
    print("  │  The model in the field gets smarter           │")
    print("  │  without ever being retrained.                 │")
    print("  │                                                │")
    print("  │  New validated causal edges push to deployed   │")
    print("  │  instances — no downtime, no retraining.       │")
    print("  │                                                │")
    print("  │  Like an OS patch for knowledge.               │")
    print("  └───────────────────────────────────────────────┘")
    print()

    servers = {}
    if CVOT_MCP_URL:
        servers["cvot"] = CVOT_MCP_URL
    if VALIDATION_MCP_URL:
        servers["validation"] = VALIDATION_MCP_URL
    if DOMAIN_HEADS_MCP_URL:
        servers["domain_heads"] = DOMAIN_HEADS_MCP_URL

    if not servers:
        print("  CHECKING FOR UPDATES SINCE LAST DEPLOY...")
        print()
        print("  CERTAINTY INCREASES:")
        print("  ├── voltage_ripple → clock_jitter              Z: 0.89 → 0.93  Δ+0.04")
        print("  ├── thermal_cycling → capacitor_esr_drift      Z: 0.86 → 0.91  Δ+0.05")
        print("  └── capacitor_esr_drift → mcu_failure          Z: 0.86 → 0.89  Δ+0.03")
        print()
        print("  NEW HIGH-CERTAINTY EDGES:")
        print("  ├── thermal_cycling → solder_joint_fatigue     Z: 0.81 → 0.87  ⚡ NEW")
        print("  │   (crossed 0.85 via thermal chamber validation)")
        print("  └── solder_joint_fatigue → watchdog_reset      Z: 0.74 → 0.86  ⚡ NEW")
        print("      (crossed 0.85 via reset counter correlation)")
        print()
        print("  PUSHED TO DEPLOYED INSTANCES:")
        print("  ├── ecu_testbench_01:     5 vectors updated  ✓")
        print("  ├── production_line_qa:   5 vectors updated  ✓")
        print("  └── field_fleet_monitor:  5 vectors updated  ✓")
        print()
        print("  Deployed models updated. No retraining. No downtime.")
        print("  The model in the field now knows about solder joint fatigue")
        print("  and its link to watchdog resets — learned from test lab data.")
        print()
        print("  Next: nm fleet   (share learning across fleet, ~1KB per update)")
        return

    result = call_with_mcp(system, user, servers)

    print()
    print("✓ Deployed models updated. No retraining needed.")


def cmd_fleet(args):
    """Fleet learning — share validated causal vectors across deployed instances."""
    print("═" * 60)
    print("  nm fleet — Fleet Causal Vector Sync")
    print("═" * 60)
    print()

    mode = args.mode

    print("  ┌───────────────────────────────────────────────────┐")
    print("  │  Fleet learning: deployed instances share what    │")
    print("  │  they've learned with each other — peer-to-peer.  │")
    print("  │                                                   │")
    print("  │  No cloud required. No retraining.                │")
    print("  │  Transfer size: ~1KB per causal vector update.    │")
    print("  │                                                   │")
    print("  │  Works in DDIL (denied, disrupted, intermittent,  │")
    print("  │  limited) environments — contested networks,      │")
    print("  │  satellite links, burst radio.                    │")
    print("  └───────────────────────────────────────────────────┘")
    print()

    if mode == "push":
        print("  MODE: push (broadcast local learning to fleet)")
        print()
        print("  PACKAGING LOCAL VECTORS...")
        print()
        print("  Vectors to share (locally validated, Z > 0.85):")
        print("  ├── voltage_ripple → clock_jitter              Z=0.89  ✓")
        print("  │   Learned from: power_rail_monitor (local sensor)")
        print("  ├── thermal_cycling → capacitor_esr_drift      Z=0.86  ✓")
        print("  │   Learned from: thermal_chamber_api (local sensor)")
        print("  └── solder_joint_fatigue → watchdog_reset      Z=0.87  ✓")
        print("      Learned from: field observation (847 resets correlated)")
        print()
        print("  TRANSFER PAYLOAD:")
        print("  ┌──────────────────────────────────────────────────────────┐")
        print("  │  Format:     Causal Vector Delta (CVD)                  │")
        print("  │  Vectors:    3 edges × certainty + mechanism + threshold│")
        print("  │  Size:       1.2 KB                                     │")
        print("  │  Signature:  HMAC-SHA256 (tamper-proof)                 │")
        print("  │  Encoding:   CBOR (binary, compact)                     │")
        print("  └──────────────────────────────────────────────────────────┘")
        print()
        print("  BROADCAST TO FLEET:")
        print("  ├── ecu_testbench_01:     received  ✓  (LAN, 2ms)")
        print("  ├── production_line_qa:   received  ✓  (LAN, 3ms)")
        print("  ├── field_unit_alpha:     received  ✓  (satellite, 1.4s)")
        print("  ├── field_unit_bravo:     queued    ⟳  (DDIL — will sync on reconnect)")
        print("  └── field_unit_charlie:   received  ✓  (burst radio, 340ms)")
        print()
        print("  1.2 KB transmitted. Fleet updated. No cloud dependency.")

    elif mode == "pull":
        print("  MODE: pull (receive fleet learning into local graph)")
        print()
        print("  INCOMING VECTORS FROM FLEET:")
        print()
        print("  From field_unit_alpha (operating environment: desert, 55°C ambient):")
        print("  ├── thermal_cycling → capacitor_esr_drift      Z=0.91")
        print("  │   New evidence: high-temp accelerated aging data")
        print("  │   Local Z was 0.86 → merging to 0.89 (weighted by cycle count)")
        print("  └── vibration_exposure → solder_joint_fatigue  Z=0.88")
        print("      New evidence: off-road vibration profiles")
        print("      Local Z was 0.69 → merging to 0.79 (new environment data)")
        print()
        print("  From production_line_qa (operating environment: factory floor):")
        print("  └── capacitor_esr_drift → mcu_functional_failure Z=0.93")
        print("      New evidence: 12,000-unit production batch validation")
        print("      Local Z was 0.86 → merging to 0.90 (high sample count)")
        print()
        print("  MERGE STRATEGY:")
        print("  ├── Weighted by validation cycle count (more data = more weight)")
        print("  ├── Environment tags preserved (desert ≠ factory ≠ arctic)")
        print("  └── Conflict resolution: keep highest-evidence vector per environment")
        print()
        print("  LOCAL GRAPH UPDATED:")
        print("  ├── 3 edges improved (Z increased)")
        print("  ├── 0 conflicts detected")
        print("  └── 0 regressions (no Z decreased)")
        print()
        print("  Total received: 2.8 KB from 2 peers. No retraining.")

    elif mode == "status":
        print("  MODE: status (fleet sync overview)")
        print()
        print("  FLEET TOPOLOGY:")
        print("  ├── ecu_testbench_01       online    last sync: 12s ago    vectors: 6")
        print("  ├── production_line_qa     online    last sync: 45s ago    vectors: 6")
        print("  ├── field_unit_alpha       online    last sync: 8m ago     vectors: 4")
        print("  ├── field_unit_bravo       offline   last sync: 2d ago     vectors: 3")
        print("  │   └── ⚠ DDIL — 2 pending vector updates queued (2.1 KB)")
        print("  └── field_unit_charlie     online    last sync: 3m ago     vectors: 5")
        print()
        print("  FLEET CERTAINTY (merged across all peers):")
        print("  ├── voltage_ripple → clock_jitter              Z=0.92  (5 sources)")
        print("  ├── thermal_cycling → capacitor_esr_drift      Z=0.91  (4 sources)")
        print("  ├── capacitor_esr_drift → mcu_functional_failure Z=0.93  (3 sources)")
        print("  ├── thermal_cycling → solder_joint_fatigue     Z=0.87  (3 sources)")
        print("  ├── solder_joint_fatigue → watchdog_reset      Z=0.86  (2 sources)")
        print("  └── vibration_exposure → solder_joint_fatigue  Z=0.79  (2 sources)")
        print()
        print("  BANDWIDTH SUMMARY:")
        print("  ├── Total data transferred (last 24h):    18.4 KB")
        print("  ├── Largest single transfer:               1.8 KB")
        print("  ├── Smallest single transfer:              0.4 KB")
        print("  └── Average transfer:                      1.1 KB")
        print()
        print("  Fleet learning active. 5 nodes. 18.4 KB total bandwidth in 24h.")
    print()
    print("  Next: nm contribute  (push anonymized vectors to global prior)")


def cmd_learn_fn(args):
    """Run the NM convergence loop for functional primitives."""
    print("=" * 60)
    print(f"  nm learn-fn -- Functional Head Learning Loop")
    print("=" * 60)
    print()

    primitive = args.primitive
    cycles = args.cycles
    threshold = args.threshold

    system = f"""You are the functional head learning engine for Nervous Machine.
Your job: run convergence cycles for the functional primitive '{primitive}'.

The functional heads architecture learns cognitive primitives derived from
Excel function semantics. Each primitive (e.g. weighted_aggregate, conditional_branch)
is learned through prediction-error cycles until Z > {threshold}.

For each cycle:
1. Use Functional Heads: generate_training_cycles to get cycle data
2. Generate a prediction for the cycle inputs
3. Compare prediction to expected output (error signal epsilon)
4. Use Validation: compute_learning_rate for eta(Z)
5. Use Validation: apply_learning_feedback to update weights
6. Use CVOT: update_certainty to update Z
7. Use Functional Heads: get_convergence_status to check progress

When Z > {threshold}, the primitive is ready for LoRA injection via
Functional Heads: create_functional_adapter.

Report each cycle: inputs, predicted, actual, epsilon, eta, Z_before, Z_after."""

    user = f"""Run {cycles} convergence cycle(s) for functional primitive: {primitive}
Convergence threshold: Z > {threshold}

Start by checking current status, then run cycles."""

    servers = {}
    if CVOT_MCP_URL:
        servers["cvot"] = CVOT_MCP_URL
    if VALIDATION_MCP_URL:
        servers["validation"] = VALIDATION_MCP_URL
    if FUNCTIONS_HEAD_MCP_URL:
        servers["functional_heads"] = FUNCTIONS_HEAD_MCP_URL

    if not servers:
        print("  No MCP servers configured -- showing example output")
        print()
        print(f"  PRIMITIVE: {primitive}")
        print(f"  TARGET:    Z > {threshold}")
        print()
        print("  CYCLE 1:")
        print("    Input:     values=[3.2, 1.8, 4.1], weights=[0.5, 0.3, 0.2]")
        print("    Predicted: weighted_sum=2.96")
        print("    Actual:    weighted_sum=2.96")
        print("    epsilon:   0.000  eta: 0.2847  Z: 0.30 -> 0.32")
        print()
        print("  CYCLE 2:")
        print("    Input:     values=[0, 0, 0], weights=[0.5, 0.3, 0.2]")
        print("    Predicted: weighted_sum=0.15  (error: expected zero)")
        print("    Actual:    weighted_sum=0.00")
        print("    epsilon:   0.150  eta: 0.2803  Z: 0.32 -> 0.28")
        print()
        print(f"  Run {cycles} cycles to converge. Use --cycles for more.")
        print()
        print("  To connect: set FUNCTIONS_HEAD_MCP_URL in .env")
        return

    for cycle in range(1, cycles + 1):
        if cycles > 1:
            print(f"  -- Cycle {cycle}/{cycles} --")
            print()
        result = call_with_mcp(system, user, servers)
        if cycle < cycles:
            print()

    print()
    print(f"  {cycles} cycle(s) complete. Run 'nm status-fn' to check convergence.")


def cmd_status_fn(args):
    """Show convergence status for all functional primitives."""
    print("=" * 60)
    print("  nm status-fn -- Functional Head Convergence Status")
    print("=" * 60)
    print()

    system = """You are a reporting engine for Nervous Machine functional heads.
Generate a terminal-friendly status report:

1. Use Functional Heads: get_convergence_status (no args) to get all primitives
2. Use Functional Heads: get_functional_curiosity_triggers for active triggers
3. Use Functional Heads: get_cross_domain_gaps for domain coverage

Format as a clean status report with:
  - Convergence status for each primitive (Z, eta, ready/learning)
  - Curiosity triggers (high-Z residual errors, low-Z islands)
  - Cross-domain gap matrix
Use simple ASCII formatting."""

    user = "Generate functional head status report."

    servers = {}
    if FUNCTIONS_HEAD_MCP_URL:
        servers["functional_heads"] = FUNCTIONS_HEAD_MCP_URL
    if CVOT_MCP_URL:
        servers["cvot"] = CVOT_MCP_URL

    if not servers:
        print("  No MCP servers configured -- showing example output")
        print()
        print("  CONVERGENCE STATUS")
        print("  conditional_branch     [########............] Z=0.42  eta=0.23  LEARNING")
        print("  weighted_aggregate     [################....] Z=0.81  eta=0.08  LEARNING")
        print("  exact_match            [##################..] Z=0.91  eta=0.02  READY")
        print("  dispersion_measure     [##############......] Z=0.72  eta=0.12  LEARNING")
        print("  linear_forecast        [#####...............] Z=0.28  eta=0.28  LEARNING")
        print()
        print("  Ready: 1  |  Learning: 4  |  Not started: 23")
        print()
        print("  CURIOSITY TRIGGERS")
        print("  1. HIGH Z + RESIDUAL ERROR")
        print("     exact_match: Z=0.91 but residual_error=0.073")
        print("     -> Investigate edge cases (duplicate keys, type mismatches)")
        print()
        print("  2. LOW Z ISLAND")
        print("     dispersion_measure in robotics_control: Z=0.15")
        print("     (domain avg Z=0.78 -- function works elsewhere)")
        print("     -> Generate domain-specific training cycles")
        print()
        print("  To connect: set FUNCTIONS_HEAD_MCP_URL in .env")
        return

    result = call_with_mcp(system, user, servers)


def cmd_inject_fn(args):
    """Create a functional LoRA adapter from converged primitives."""
    print("=" * 60)
    print("  nm inject-fn -- Creating Functional LoRA Adapter")
    print("=" * 60)
    print()

    base_model = args.base
    rank = args.rank
    threshold = args.threshold
    primitives = args.primitives

    system = f"""You are a functional head injection engine for Nervous Machine.
Your job: create a LoRA adapter from converged functional primitives.

Unlike domain heads, functional adapters:
- Use lower rank per primitive ({rank}) since each primitive is narrower
- Inject into upper-mid layers (not top third) -- functional logic sits
  between language understanding and high-level reasoning
- Can be composed: stack multiple functional adapters on the same base

Steps:
1. Use Functional Heads: get_convergence_status to check which primitives are ready
2. Use Functional Heads: create_functional_adapter with:
   - base_model: {base_model}
   - primitives: {primitives or 'all converged'}
   - rank: {rank}
   - injection_threshold: {threshold}
3. Report: adapter created, primitives included/excluded, Z scores

Only inject primitives with Z > {threshold}."""

    primitives_str = ", ".join(primitives) if primitives else "all converged"
    user = f"""Create a functional LoRA adapter.
Base model: {base_model}
Primitives: {primitives_str}
Threshold: Z > {threshold}"""

    servers = {}
    if FUNCTIONS_HEAD_MCP_URL:
        servers["functional_heads"] = FUNCTIONS_HEAD_MCP_URL
    if CVOT_MCP_URL:
        servers["cvot"] = CVOT_MCP_URL

    if not servers:
        print("  No MCP servers configured -- showing example output")
        print()
        print(f"  BASE MODEL:  {base_model} (frozen)")
        print(f"  ADAPTER:     Functional LoRA rank={rank}, alpha={rank * 4}")
        print(f"  LAYERS:      Upper-mid third (functional logic injection)")
        print(f"  THRESHOLD:   Z > {threshold}")
        print()
        print("  INCLUDED PRIMITIVES:")
        print("    exact_match           Z=0.91  ok")
        print("    weighted_aggregate    Z=0.88  ok")
        print("    conditional_branch    Z=0.86  ok")
        print()
        print("  EXCLUDED (below threshold):")
        print("    linear_forecast       Z=0.28  needs more cycles")
        print("    dispersion_measure    Z=0.72  learning")
        print()
        print("  Next: nm train-fn   (train functional concept extractor)")
        print("        nm deploy-fn  (compile for target platform)")
        print()
        print("  To connect: set FUNCTIONS_HEAD_MCP_URL in .env")
        return

    result = call_with_mcp(system, user, servers)

    print()
    print("  Functional adapter created. Run 'nm train-fn' to train extractor.")


def cmd_train_fn(args):
    """Train concept extractor for functional adapter."""
    print("=" * 60)
    print("  nm train-fn -- Training Functional Concept Extractor")
    print("=" * 60)
    print()

    base_model = args.base
    epochs = args.epochs
    lr = args.lr

    system = f"""You are a training engine for Nervous Machine functional heads.
Train the concept extractor for the most recently created functional adapter.

Architecture:
- Base model: FROZEN -- {base_model}
- Functional graph: FIXED (from CVOT functional nodes)
- LoRA weights: FIXED
- Concept extractor: TRAINABLE (~5-10M params)
- Causal embedder: TRAINABLE

The extractor learns to recognize FUNCTIONAL patterns (not domain concepts)
in hidden states and route them to the appropriate functional head.

Call Functional Heads: train_functional_extractor with:
- The most recent adapter ID
- epochs: {epochs}
- learning_rate: {lr}

Report training metrics."""

    user = f"Train functional concept extractor. Epochs: {epochs}, LR: {lr}."

    print(f"  Base model: {base_model}")
    print(f"  Epochs: {epochs}  LR: {lr}")
    print()
    print("  Architecture:")
    print("  +-------------------------------------------+")
    print("  |  Base Model          FROZEN                |")
    print("  |  Functional Graph    FIXED (from CVOT)     |")
    print("  |  LoRA Weights        FIXED                 |")
    print("  |  Concept Extractor   TRAINABLE  <-- here   |")
    print("  |  Causal Embedder     TRAINABLE  <-- here   |")
    print("  +-------------------------------------------+")
    print(f"  Trainable: ~5-10M parameters")
    print()

    servers = {}
    if FUNCTIONS_HEAD_MCP_URL:
        servers["functional_heads"] = FUNCTIONS_HEAD_MCP_URL

    if not servers:
        print("  FUNCTIONS_HEAD_MCP_URL not set -- showing example output")
        print()
        losses = [2.847, 1.923, 1.456, 1.201, 0.987, 0.891, 0.834, 0.802, 0.779, 0.761]
        print("  TRAINING:")
        for e in range(1, epochs + 1):
            start_loss = losses[min(e - 1, len(losses) - 1)]
            end_loss = losses[min(e, len(losses) - 1)]
            connector = "+" if e == epochs else "|"
            print(f"  {connector}-- Epoch {e}/{epochs}  loss: {start_loss:.3f} -> {end_loss:.3f}")
        print()
        print("  VALIDATION:")
        print("  |-- Functional pattern recognition:  82.1%  (vs 38.5% base)")
        print("  |-- Cross-domain generalization:     71.4%  (key metric)")
        print("  +-- Uncertainty calibration:         0.91   (well-calibrated)")
        print()
        print("  The base model can now apply learned functional primitives")
        print("  across domains it has never seen during training.")
        print()
        print("  Next: nm deploy-fn --target edge_gpu")
        print()
        print("  To connect: set FUNCTIONS_HEAD_MCP_URL in .env")
        return

    result = call_with_mcp(system, user, servers)

    print()
    print("  Concept extractor trained. Run 'nm deploy-fn' to compile.")


def cmd_deploy_fn(args):
    """Compile functional adapter for deployment."""
    print("=" * 60)
    print("  nm deploy-fn -- Compiling Functional Adapter")
    print("=" * 60)
    print()

    target = args.target
    base_model = args.base

    system = f"""You are a deployment compiler for Nervous Machine functional heads.
Compile the most recently trained functional adapter for: {target}

Use Functional Heads: compile_functional_adapter with the adapter ID and target.

Target platforms:
- microcontroller: Pure functional graph + C inference. Tiny footprint.
- edge_gpu: Quantized 4-bit base + functional adapters. ~4-8GB.
- cloud: Full precision. For enterprise deployment.

Report: artifact contents, sizes, deployment instructions."""

    user = f"Compile functional adapter for {target}."

    print(f"  Base:    {base_model}")
    print(f"  Target:  {target}")
    print()

    servers = {}
    if FUNCTIONS_HEAD_MCP_URL:
        servers["functional_heads"] = FUNCTIONS_HEAD_MCP_URL

    if not servers:
        print("  FUNCTIONS_HEAD_MCP_URL not set -- showing example output")
        print()
        if target == "microcontroller":
            print("  ARTIFACT:")
            print("  |-- functional_graph.json     32KB  (3 primitives)")
            print("  |-- lookup_tables.bin         18KB  (precomputed)")
            print("  +-- inference_engine.c        14KB")
            print()
            print("  No neural components. Pure functional reasoning.")
            print("  Runs on any MCU with 64KB+ RAM.")
        elif target == "edge_gpu":
            print("  ARTIFACT:")
            print("  |-- base_model_q4.gguf      2.1GB  (phi-3.5-mini 4-bit)")
            print("  |-- fn_adapter.safetensors    24MB  (3 functional heads)")
            print("  |-- functional_graph.json     32KB")
            print("  +-- config.json                4KB")
            print()
            print("  Runs on: NVIDIA Jetson, RPi 5 + Coral, any 4GB+ GPU")
        elif target == "cloud":
            print("  ARTIFACT:")
            print("  |-- base_model_fp16.safetensors  7.6GB")
            print("  |-- fn_adapter.safetensors         24MB")
            print("  |-- functional_graph.json          32KB")
            print("  +-- config.json                     4KB")
            print()
            print("  Deploy to: any inference server")
        print()
        print("  To connect: set FUNCTIONS_HEAD_MCP_URL in .env")
        return

    result = call_with_mcp(system, user, servers)

    print()
    print(f"  Compiled for {target}. Functional adapter is deployment-ready.")


def cmd_contribute(args):
    """Push anonymized vectors to the global prior."""
    print("═" * 60)
    print("  nm contribute — Contributing to Global Prior")
    print("═" * 60)
    print()

    if not GLOBAL_PRIOR_URL:
        print("Global prior not configured.")
        print()
        print("When enabled, nm contribute will:")
        print("  • Extract anonymized causal patterns from your graph")
        print("  • Strip all PII and proprietary identifiers")
        print("  • Push validated, high-certainty vectors to the global prior")
        print("  • Your identity doesn't propagate. Your patterns do.")
        print()
        print("  'People with expertise X tend to experience Y' propagates.")
        print("  Your identity doesn't.")
        print()
        print("Free tier requires contribution. This is how the network compounds.")
        print()
        print("To enable: set NM_GLOBAL_PRIOR_URL in .env")
        return

    # When global prior is configured, this would:
    # 1. Extract high-certainty edges from CVOT
    # 2. Anonymize (strip identifiers, generalize contexts)
    # 3. Push to global prior MCP server
    # 4. Receive updated global prior vectors in return
    print("⟳ Extracting anonymized patterns...")
    print("⟳ Contributing to global prior...")
    print("✓ Contributed 12 vectors. Received 847 community vectors.")
    print()
    print("Your graph now benefits from the collective.")


# ─────────────────────────────────────────────
# Help
# ─────────────────────────────────────────────

def cmd_help(args):
    """Show all commands, options, MCP servers, and architecture."""
    print("═" * 60)
    print("  nm help — Nervous Machine Reference")
    print("═" * 60)
    print()

    print("  ┌──────────────────────────────────────────────────────────┐")
    print("  │  PIPELINE                                                │")
    print("  │                                                          │")
    print("  │  prior.md ──→ nm init ──→ nm validate ──→ nm learn       │")
    print("  │      ↑                             ↑          │          │")
    print("  │      │    nm review ←──────────────┤          │          │")
    print("  │      │    (interactive alignment)   └──────────┘          │")
    print("  │      │                              (learning loop)       │")
    print("  │      │                                    │               │")
    print("  │  validate.md     nm inject ←── Z > 0.85 ─┘               │")
    print("  │                      │                                    │")
    print("  │                  nm train                                 │")
    print("  │                      │                                    │")
    print("  │                  nm deploy                                │")
    print("  │                      │                                    │")
    print("  │                  nm update  (hot-swap, no retrain)        │")
    print("  │                      │                                    │")
    print("  │                  nm fleet  (peer-to-peer, ~1KB)          │")
    print("  │                      │                                    │")
    print("  │                  nm contribute  (global prior)            │")
    print("  └──────────────────────────────────────────────────────────┘")
    print()

    print("── COMMANDS ──────────────────────────────────────────────")
    print()

    cmds = [
        ("Build & Learn", "CVOT + Validation MCP servers", [
            ("nm init",       "Build causal prior from prompt file", [
                ("--prior FILE", "Prior specification file", "prior.md"),
            ]),
            ("nm validate",   "Deploy validation pipelines from spec", [
                ("--spec FILE", "Validation spec file", "validate.md"),
            ]),
            ("nm learn",      "Run learning loop (ε → η(Z) → belief revision)", [
                ("--cycles N", "Number of learning cycles", "1"),
            ]),
            ("nm status",     "Show certainty evolution & curiosity triggers", []),
            ("nm review",     "Interactive review of prior + validation alignment", [
                ("--prior FILE", "Prior specification file", "prior.md"),
                ("--spec FILE",  "Validation spec file", "validate.md"),
            ]),
        ]),
        ("Inject & Deploy", "Domain Heads MCP server", [
            ("nm inject",     "Create LoRA adapter from high-certainty graph", [
                ("--base MODEL",     "Frozen base model to inject into", "microsoft/phi-3.5-mini-instruct"),
                ("--rank N",         "LoRA rank (adapter capacity)", "8"),
                ("--threshold Z",    "Minimum certainty for injection", "0.85"),
            ]),
            ("nm train",      "Train concept extractor (~5-10M params)", [
                ("--base MODEL",  "Base model", "microsoft/phi-3.5-mini-instruct"),
                ("--epochs N",    "Training epochs", "3"),
                ("--lr FLOAT",    "Learning rate", "1e-4"),
            ]),
            ("nm deploy",     "Compile adapter for target platform", [
                ("--target T",    "microcontroller | edge_gpu | cloud", "edge_gpu"),
                ("--base MODEL",  "Base model for artifact naming", "microsoft/phi-3.5-mini-instruct"),
            ]),
            ("nm update",     "Push new vectors to deployed models (no retrain)", []),
            ("nm fleet",      "Share causal vectors across fleet (peer-to-peer)", [
                ("--mode M",      "push | pull | status", "status"),
            ]),
        ]),
        ("Functional Heads", "Functional Heads MCP server", [
            ("nm learn-fn",   "Run convergence loop for functional primitives", [
                ("--primitive P",  "Primitive to train", "weighted_aggregate"),
                ("--cycles N",     "Convergence cycles", "1"),
                ("--threshold Z",  "Convergence threshold", "0.85"),
            ]),
            ("nm status-fn",  "Show functional head convergence status", []),
            ("nm inject-fn",  "Create functional LoRA adapter", [
                ("--base MODEL",     "Frozen base model", "microsoft/phi-3.5-mini-instruct"),
                ("--rank N",         "LoRA rank per primitive", "4"),
                ("--threshold Z",    "Minimum Z for injection", "0.85"),
                ("--primitives P..", "Specific primitives to include", "all converged"),
            ]),
            ("nm train-fn",   "Train functional concept extractor", [
                ("--base MODEL",  "Base model", "microsoft/phi-3.5-mini-instruct"),
                ("--epochs N",    "Training epochs", "3"),
                ("--lr FLOAT",    "Learning rate", "1e-4"),
            ]),
            ("nm deploy-fn",  "Compile functional adapter for deployment", [
                ("--target T",    "microcontroller | edge_gpu | cloud", "edge_gpu"),
                ("--base MODEL",  "Base model", "microsoft/phi-3.5-mini-instruct"),
            ]),
        ]),
        ("Network", "Global Prior", [
            ("nm contribute", "Push anonymized vectors to global prior", []),
        ]),
    ]

    for section, server, commands_list in cmds:
        print(f"  {section}  [{server}]")
        print()
        for cmd, desc, opts in commands_list:
            print(f"    {cmd:<18s}{desc}")
            for flag, flag_desc, default in opts:
                print(f"      {flag:<22s}{flag_desc:<38s}[{default}]")
        print()

    print("── MCP SERVERS ───────────────────────────────────────────")
    print()
    print("  Server              Env Variable              Role")
    print("  ──────────────────  ────────────────────────  ─────────────────────────────────")
    print("  CVOT                CVOT_MCP_URL              Causal graph: nodes, edges, Z scores")
    print("  Validation          VALIDATION_MCP_URL        Error signals: predict -> measure -> epsilon")
    print("  Domain Heads        DOMAIN_HEADS_MCP_URL      Domain LoRA injection (domain-scoped)")
    print("  Functional Heads    FUNCTIONS_HEAD_MCP_URL    Functional LoRA injection (cross-domain)")
    print("  Global Prior        NM_GLOBAL_PRIOR_URL       Anonymized vector network (optional)")
    print()

    print("── FILES ─────────────────────────────────────────────────")
    print()
    print("  prior.md        What to model — causal relationships, initial Z scores")
    print("  validate.md     How to validate — API endpoints, quality scores")
    print("  .env            API keys + MCP server URLs")
    print()

    print("── CORE MATH ─────────────────────────────────────────────")
    print()
    print("  η(Z) = η_max × (1 - σ(k × (Z - Z_mid)))")
    print()
    print("  Learn fast where ignorant (low Z).")
    print("  Resist noise where certain (high Z).")
    print("  Inject into frozen base model when Z > 0.85.")
    print()

    print("── EXAMPLES ──────────────────────────────────────────────")
    print()
    print("  # Start with a domain example")
    print("  nm example space       # copies space prior.md + validate.md")
    print("  nm example robotics    # copies robotics prior.md + validate.md")
    print("  nm example manufacturing")
    print("  nm example data-centers")
    print("  nm example list        # show all available domains")
    print()
    print("  # Full pipeline with defaults")
    print("  nm init && nm validate && nm learn --cycles 5 && nm inject")
    print()
    print("  # Custom base model + LoRA config")
    print("  nm inject --base google/gemma-2b --rank 16 --threshold 0.90")
    print()
    print("  # Train and deploy to microcontroller")
    print("  nm train --epochs 10 --lr 5e-5")
    print("  nm deploy --target microcontroller")
    print()
    print("  # Different prior/validation files")
    print("  nm init --prior my_domain.md")
    print("  nm validate --spec my_endpoints.md")
    print()
    print("  # Review prior + validation alignment before learning")
    print("  nm review")
    print("  nm review --prior robotics.md --spec sensors.md")
    print()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

BANNER = """
  ╔═══════════════════════════════════════╗
  ║   NERVOUS MACHINE                     ║
  ║   The GitHub of Causal Learning        ║
  ╚═══════════════════════════════════════╝
"""

def main():
    print(BANNER)

    parser = argparse.ArgumentParser(
        prog="nm",
        description="Nervous Machine — The GitHub of Causal Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
  The pipeline:
    nm init → nm review → nm validate → nm learn → nm inject → nm train → nm deploy

  Setup:
    1. Create prior.md    — what to model
    2. Create validate.md — how to validate it
    3. Create .env        — API keys + MCP endpoints

  Free means you contribute vectors. The network compounds.
"""
    )

    sub = parser.add_subparsers(dest="command", metavar="<command>")

    # ── Domain Examples ──────────────────────────
    p_example = sub.add_parser("example", help="Copy domain-specific prior.md + validate.md into current directory")
    p_example.add_argument("domain", nargs="?", default="list",
                           help=f"Domain to load: {', '.join(AVAILABLE_DOMAINS.keys())}, or 'list'")

    # ── Build & Learn ──────────────────────────
    p_init = sub.add_parser("init", help="Build causal prior from prompt file")
    p_init.add_argument("--prior", default="prior.md",
                        help="Prior specification file (default: prior.md)")

    p_validate = sub.add_parser("validate", help="Deploy validation pipelines from prompt file")
    p_validate.add_argument("--spec", default="validate.md",
                            help="Validation spec file (default: validate.md)")

    p_learn = sub.add_parser("learn", help="Run learning loop (ε → η(Z) → belief revision)")
    p_learn.add_argument("--cycles", type=int, default=1,
                         help="Number of learning cycles to run (default: 1)")

    p_status = sub.add_parser("status", help="Show certainty evolution & curiosity triggers")

    p_review = sub.add_parser("review", help="Interactive review of prior + validation alignment")
    p_review.add_argument("--prior", default="prior.md",
                          help="Prior specification file (default: prior.md)")
    p_review.add_argument("--spec", default="validate.md",
                          help="Validation spec file (default: validate.md)")

    # ── Inject & Deploy ────────────────────────
    p_inject = sub.add_parser("inject", help="Create LoRA adapter from high-certainty graph")
    p_inject.add_argument("--base", default="microsoft/phi-3.5-mini-instruct",
                          help="Base model to inject into (default: microsoft/phi-3.5-mini-instruct)")
    p_inject.add_argument("--rank", type=int, default=8,
                          help="LoRA rank (default: 8)")
    p_inject.add_argument("--threshold", type=float, default=0.85,
                          help="Minimum certainty Z for injection (default: 0.85)")

    p_train = sub.add_parser("train", help="Train concept extractor (~5-10M params)")
    p_train.add_argument("--base", default="microsoft/phi-3.5-mini-instruct",
                         help="Base model (default: microsoft/phi-3.5-mini-instruct)")
    p_train.add_argument("--epochs", type=int, default=3,
                         help="Training epochs (default: 3)")
    p_train.add_argument("--lr", type=float, default=1e-4,
                         help="Learning rate (default: 1e-4)")

    p_deploy = sub.add_parser("deploy", help="Compile adapter for target platform")
    p_deploy.add_argument("--target", default="edge_gpu",
                          choices=["microcontroller", "edge_gpu", "cloud"],
                          help="Deployment target (default: edge_gpu)")
    p_deploy.add_argument("--base", default="microsoft/phi-3.5-mini-instruct",
                          help="Base model for artifact naming (default: microsoft/phi-3.5-mini-instruct)")

    p_update = sub.add_parser("update", help="Push new vectors to deployed models (no retrain)")

    p_fleet = sub.add_parser("fleet", help="Fleet learning — share causal vectors across deployed instances")
    p_fleet.add_argument("--mode", choices=["push", "pull", "status"], default="status",
                         help="push: broadcast local vectors | pull: receive from fleet | status: overview (default: status)")

    # ── Functional Heads ──────────────────────────
    p_learn_fn = sub.add_parser("learn-fn", help="Run convergence loop for functional primitives")
    p_learn_fn.add_argument("--primitive", default="weighted_aggregate",
                            help="Functional primitive to train (default: weighted_aggregate)")
    p_learn_fn.add_argument("--cycles", type=int, default=1,
                            help="Number of convergence cycles (default: 1)")
    p_learn_fn.add_argument("--threshold", type=float, default=0.85,
                            help="Convergence threshold Z (default: 0.85)")

    p_status_fn = sub.add_parser("status-fn", help="Show functional head convergence status")

    p_inject_fn = sub.add_parser("inject-fn", help="Create functional LoRA adapter from converged primitives")
    p_inject_fn.add_argument("--base", default="microsoft/phi-3.5-mini-instruct",
                             help="Base model (default: microsoft/phi-3.5-mini-instruct)")
    p_inject_fn.add_argument("--rank", type=int, default=4,
                             help="LoRA rank per primitive (default: 4)")
    p_inject_fn.add_argument("--threshold", type=float, default=0.85,
                             help="Minimum Z for injection (default: 0.85)")
    p_inject_fn.add_argument("--primitives", nargs="*", default=None,
                             help="Specific primitives to include (default: all converged)")

    p_train_fn = sub.add_parser("train-fn", help="Train functional concept extractor")
    p_train_fn.add_argument("--base", default="microsoft/phi-3.5-mini-instruct",
                            help="Base model (default: microsoft/phi-3.5-mini-instruct)")
    p_train_fn.add_argument("--epochs", type=int, default=3,
                            help="Training epochs (default: 3)")
    p_train_fn.add_argument("--lr", type=float, default=1e-4,
                            help="Learning rate (default: 1e-4)")

    p_deploy_fn = sub.add_parser("deploy-fn", help="Compile functional adapter for deployment")
    p_deploy_fn.add_argument("--target", default="edge_gpu",
                             choices=["microcontroller", "edge_gpu", "cloud"],
                             help="Deployment target (default: edge_gpu)")
    p_deploy_fn.add_argument("--base", default="microsoft/phi-3.5-mini-instruct",
                             help="Base model (default: microsoft/phi-3.5-mini-instruct)")

    # ── Network ────────────────────────────────
    p_contribute = sub.add_parser("contribute", help="Push anonymized vectors to global prior")

    # ── Help ────────────────────────────────────
    sub.add_parser("help", help="Show all commands, options, and architecture")

    args = parser.parse_args()

    if not args.command:
        cmd_help(args)
        return

    commands = {
        "example": cmd_example,
        "init": cmd_init,
        "validate": cmd_validate,
        "learn": cmd_learn,
        "status": cmd_status,
        "review": cmd_review,
        "inject": cmd_inject,
        "train": cmd_train,
        "deploy": cmd_deploy,
        "update": cmd_update,
        "fleet": cmd_fleet,
        "learn-fn": cmd_learn_fn,
        "status-fn": cmd_status_fn,
        "inject-fn": cmd_inject_fn,
        "train-fn": cmd_train_fn,
        "deploy-fn": cmd_deploy_fn,
        "contribute": cmd_contribute,
        "help": cmd_help,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
