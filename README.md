# Nervous Machine

**The GitHub of Causal Learning.**

Build causal models from domain expertise. Validate them against real sensor data. Deploy to microcontrollers, edge GPUs, or cloud. Share learning across fleets at 1KB.

## Quick Start

```bash
pip install .
nm init
nm validate
nm deploy --target microcontroller
nm learn --cycles 5
nm status
nm fleet
```

That's the whole pipeline. Two prompt files. Zero dependencies. No GPU required.

## What It Does

Nervous Machine turns domain expertise into validated causal graphs that run on embedded hardware.

**You write two files:**

- `prior.md` — your causal hypotheses ("thermal cycling causes capacitor ESR drift")
- `validate.md` — your sensor endpoints ("thermal chamber API at port 8080")

**The system does the rest:**

1. Builds a causal graph from your hypotheses
2. Connects to your sensors and computes prediction error
3. Runs learning cycles until edges reach high certainty
4. Deploys a 48KB causal graph + 12KB C inference engine to your MCU
5. Shares validated learning across your fleet at ~1KB per update

Every stage produces reviewable artifacts — your domain experts can inspect the graph JSON, the validation functions, and the deployed C code before anything runs on hardware.

## Commands

### Core Pipeline

| Command | What it does |
|---------|-------------|
| `nm init` | Build causal graph from `prior.md` |
| `nm validate` | Connect validation pipelines from `validate.md` |
| `nm deploy --target microcontroller` | Compile graph to C for any MCU with 64KB+ RAM |
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

## Deployment Targets

| Target | What deploys | Runs on | Size |
|--------|-------------|---------|------|
| `microcontroller` | Causal graph + C inference engine | Any MCU, 64KB+ RAM | ~50KB |
| `edge_gpu` | Quantized LLM + causal graph | Jetson, Kria, RPi 5 | ~2GB |
| `cloud` | Full precision LLM + causal graph | Any inference server | ~8GB |

The MCU target requires **no neural network** — it's pure causal graph traversal compiled to C.

## Example Domain: MCU Reliability

This demo ships with an MCU reliability scenario:

- **Stressors:** thermal cycling, voltage ripple, vibration
- **Failure modes:** solder joint fatigue, capacitor ESR drift
- **Symptoms:** clock jitter, watchdog resets
- **Outcome:** MCU functional failure

The `examples/` directory contains the prior and validation specs. Replace them with your own domain.

## Project Structure

```
prior.md          # Your causal hypotheses (edit this)
validate.md       # Your sensor endpoints (edit this)
nm.py             # CLI — run with `nm` after install
SKILL.md          # Agent-readable demo walkthrough
```

## Requirements

- Python 3.10+
- No other dependencies (demo mode)
- For live MCP server connections: `pip install ".[live]"`

## License

Proprietary. Contact [nervousmachine.com](https://nervousmachine.com) for licensing.
