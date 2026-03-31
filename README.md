# Nervous Machine

**The GitHub of Causal Learning.**

Build causal models from domain expertise. Validate them against real sensor data. Deploy to microcontrollers, edge GPUs, or cloud. Share learning across fleets at 1KB.

## Quick Start

```bash
pip install .
git checkout space          # or: robotics, manufacturing, data-centers
nm init
nm validate
nm deploy
nm learn --cycles 5
nm status
nm fleet
```

That's the whole pipeline. Two prompt files. Zero dependencies. No GPU required.

## Domain Branches

Each domain has its own branch with `prior.md` and `validate.md` pre-configured for that vertical. Check out a branch and run the pipeline — domain detection is automatic.

```bash
git checkout space           # thermospheric density / satellite drag
git checkout robotics        # sim-to-real calibration / 6-DOF arms
git checkout manufacturing   # CNC process quality / root cause attribution
git checkout data-centers    # per-zone thermal management / PUE optimization
```

| Domain | Drivers | Validation Sources | Use Case |
|--------|---------|-------------------|----------|
| **space** | Solar EUV, geomagnetic activity, solar wind, Joule heating, seasonal | GRACE-FO accelerometer, TLE debris catalog, SWPC feeds | Voxel-level thermospheric density for satellite drag prediction |
| **robotics** | Joint friction, payload mass, ambient temp, surface condition, controller latency | Joint encoders (RTDE), force/torque sensor, environment sensors | Sim-to-real calibration for robotic arm dynamics |
| **manufacturing** | Tool wear, ambient temp, material hardness, coolant, fixture clamping | CMM inspection, CNC MTConnect, coolant monitor | Per-line process quality and root cause attribution |
| **data-centers** | IT workload, CRAC airflow, ambient temp, rack config, floor tiles | Rack inlet sensors, intelligent PDUs, BMS/CRAC telemetry | Per-zone thermal management and cooling optimization |

Each domain auto-detects when you run `nm init` — the CLI renders domain-specific nodes, edges, and endpoints in dry-run mode.

If you have your own domain, stay on `main` and write `prior.md` + `validate.md` from scratch.

## What It Does

Nervous Machine turns domain expertise into validated causal graphs that run on embedded hardware.

**You write two files:**

- `prior.md` — your causal hypotheses ("solar EUV flux drives thermospheric heating")
- `validate.md` — your sensor endpoints ("GRACE-FO accelerometer at orbit cadence")

**The system does the rest:**

1. Builds a causal graph from your hypotheses
2. Connects to your sensors and computes prediction error
3. Runs learning cycles until edges reach high certainty
4. Deploys the validated causal graph to your target hardware
5. Shares validated learning across your fleet at ~1KB per update

Every stage produces reviewable artifacts — your domain experts can inspect the graph JSON, the validation functions, and the deployed inference code before anything runs on hardware.

## Commands

### Core Pipeline

| Command | What it does |
|---------|-------------|
| `nm init` | Build causal graph from `prior.md` |
| `nm validate` | Connect validation pipelines from `validate.md` |
| `nm deploy` | Deploy to the domain's default target (auto-detected) |
| `nm learn --cycles N` | Run prediction-error learning cycles |
| `nm status` | Show certainty evolution and curiosity triggers |
| `nm fleet` | Share learning across deployed instances (~1KB) |

### Extended Pipeline (LLM Integration)

| Command | What it does |
|---------|-------------|
| `nm inject` | Promote validated edges (Z > 0.85) into a frozen base model |
| `nm train` | Train causal reasoning layer (~8 min on 1x GPU) |
| `nm deploy --target edge_gpu` | Deploy LLM + causal reasoning to Jetson, Kria, etc. |
| `nm deploy --target cloud` | Deploy for fleet-wide monitoring |
| `nm update` | Push new validated edges to deployed models (no retrain) |

## Deployment Targets

| Target | What deploys | Runs on | Size |
|--------|-------------|---------|------|
| `microcontroller` | Causal graph + Rust inference engine | Any MCU, 64KB+ RAM (~$2 device) | ~50KB |
| `edge_gpu` | Quantized LLM + causal graph | Jetson Orin, Kria, RPi 5 | ~2GB |
| `cloud` | Full precision LLM + causal graph | Any inference server | ~8GB |

Domain branches default to the most likely available hardware for each vertical — `space`, `robotics`, and `manufacturing` default to `edge_gpu`; `data-centers` defaults to `cloud`. All targets are interchangeable: pass `--target microcontroller` to any domain to see pure causal graph inference compiled to Rust with no neural network, no cloud dependency, and no GPU — running on any MCU with 64KB+ RAM.

## How Certainty Works

Every causal edge starts as a hypothesis with a low certainty score (Z).

Each learning cycle:
- The system **predicts** an outcome based on the causal graph
- It **observes** the real value from your sensor
- It computes **error** (prediction vs. observation)
- It **updates certainty** — fast when ignorant, cautious when confident

When an edge crosses Z = 0.85, it's considered validated. Only validated edges get deployed or promoted into models.

Edges that don't converge trigger **curiosity** — the system identifies what it doesn't know and flags it for investigation.

## Fleet Learning

Deployed instances share validated causal vectors peer-to-peer.

- **~1KB per update** — fits in a burst radio packet or satellite ping
- **No cloud required** — works in DDIL (denied, disrupted, intermittent, limited) networks
- **HMAC-signed** — tamper-proof payloads prevent adversarial injection
- **Environment-tagged** — desert data stays tagged as desert, factory as factory

```bash
nm fleet                # show fleet topology and sync status
nm fleet --mode push    # broadcast local learning to fleet
nm fleet --mode pull    # receive and merge fleet learning
```

## Project Structure

```
prior.md              # Your causal hypotheses (edit this)
validate.md           # Your sensor endpoints (edit this)
nm.py                 # CLI — run with `nm` after install
SKILL.md              # Agent-readable demo walkthrough

Git branches:
  space               # prior.md + validate.md for thermospheric density
  robotics            # prior.md + validate.md for sim-to-real calibration
  manufacturing       # prior.md + validate.md for CNC process quality
  data-centers        # prior.md + validate.md for thermal management
```

## Requirements

- Python 3.10+
- No other dependencies (demo mode)
- For live MCP server connections: `pip install ".[live]"`

## License

Proprietary. Contact [nervousmachine.com](https://nervousmachine.com) for licensing.
