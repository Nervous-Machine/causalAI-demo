# Nervous Machine Demo — Agent Skill

Run the full NM learning pipeline demo. Walk through each stage, explain what's happening at every step, and show how causal certainty evolves from hypothesis to deployment-ready.

## Prerequisites

Install the CLI (zero dependencies):

```bash
pip install .
```

## Step 0 — Choose a Domain

Before starting the pipeline, check out a domain branch. Each branch has `prior.md` and `validate.md` pre-configured for that vertical.

Run:
```bash
git branch -a           # list all available domain branches
```

Then check out a domain:
```bash
git checkout space           # thermospheric density / satellite drag
git checkout robotics        # sim-to-real calibration / 6-DOF arms
git checkout manufacturing   # CNC process quality / root cause attribution
git checkout data-centers    # per-zone thermal management / PUE optimization
```

Explain to the user:
- Each branch has a domain-specific `prior.md` and `validate.md` already committed
- Each domain has 5–7 causal drivers, 3+ validation endpoints, and a phased validation strategy
- The rest of the pipeline works identically regardless of domain — `nm init`, `nm validate`, `nm learn`
- They can inspect and edit both files before proceeding — the system starts with their domain expertise
- If the user has their own domain, stay on `main` and write `prior.md` + `validate.md` from scratch

The pipeline auto-detects which domain is loaded and renders domain-specific output in dry-run mode.

**For the default demo (no specific audience):** use `git checkout space` — it has the strongest case study data (16% MAPE vs 85% JB2008, >0.91 certainty on GRACE-FO).

**For robotics audiences:** use `git checkout robotics` — sim-to-real gap is immediately relatable.

**For manufacturing audiences:** use `git checkout manufacturing` — root cause attribution and per-line quality are high-value pain points.

**For data center audiences:** use `git checkout data-centers` — PUE optimization and overcooling waste are concrete cost problems.

---

## Part 1 — Core Pipeline

This is the main demo. No neural network, no LLM, no GPU required for the core pipeline. Pure causal graph reasoning.

### 1. Build the Causal Prior

Run:
```bash
nm init
```

Explain to the user:
- This reads `prior.md` — a domain expert's causal hypotheses
- The CLI auto-detects the domain and renders the appropriate nodes and edges
- For **space**: 8 nodes (5 drivers like solar EUV, geomagnetic activity + 3 outcomes), 7 causal edges with Z between 0.25–0.40
- For **robotics**: 8 nodes (5 drivers like joint friction, payload mass + 3 outcomes), 7 causal edges
- For **manufacturing**: 8 nodes (5 drivers like tool wear, material hardness + 3 outcomes), 7 causal edges
- For **data-centers**: 8 nodes (5 drivers like IT workload, CRAC airflow + 3 outcomes), 7 causal edges
- These are *hypotheses* — the machine doesn't trust them yet
- Semantic edges (IS_A, PART_OF) give the model ontological context
- Point out the **reviewable output** — this is what the customer's domain experts approve before anything runs

### 2. Deploy Validation Pipelines

Run:
```bash
nm validate
```

Explain to the user:
- This reads `validate.md` — specifications for real-world data sources
- The CLI renders domain-specific endpoints:
  - **space**: GRACE-FO accelerometer, TLE debris catalog, SWPC solar wind feeds
  - **robotics**: joint encoders via RTDE, wrist force/torque sensor, environment sensors
  - **manufacturing**: CMM inspection, CNC MTConnect feed, coolant monitor
  - **data-centers**: rack inlet sensors, intelligent PDUs, BMS/CRAC telemetry
- Each edge gets an error signal: ε = |prediction - observation|
- The error signals close the learning loop — without them, Z never moves
- Point out the **reviewable validation functions** — these show exactly how each sensor connects to each causal edge
- Domain experts can review these functions to verify the prediction-vs-observation logic makes physical sense

### 3. Deploy

Run:
```bash
nm deploy
```

Explain to the user:
- Show them what they're building *before* the learning starts — this grounds the whole demo
- Domain branches default to the most likely available hardware for each vertical:
  - **space** and **robotics**: NVIDIA Jetson Orin (edge GPU, field-deployable)
  - **manufacturing**: industrial edge GPU (local inference, no cloud dependency)
  - **data-centers**: cloud inference server (fleet-wide monitoring)
- These defaults reflect real deployment environments — but all targets are interchangeable
- To see pure causal graph inference on a tiny device, run `nm deploy --target microcontroller`:
  - Output is a 48KB JSON graph + 12KB C inference engine + 2KB config header
  - Runs on any MCU with 64KB+ RAM (~$2 device): STM32, ESP32, AMD Xilinx, etc.
  - No neural network, no cloud, no GPU — just causal graph traversal compiled to C
- Point out the **reviewable code** — the output shows the causal graph JSON and the inference engine source
- Right now the Z scores are low — the graph is a hypothesis. Next we validate it.

### 4. Run the Learning Loop

Run:
```bash
nm learn --cycles 5
```

Explain to the user:
- Each cycle: predict → observe → compute error → update certainty (Z)
- Watch Z climb across cycles — some edges converge faster than others
- The system adapts its learning rate: fast when ignorant (low Z), cautious when certain (high Z)
- When an edge crosses Z=0.85, it's flagged as validated
- After 5 cycles, the strongest causal pathways should cross the threshold
- The deployed graph gets better with every cycle — same inference code, updated certainty values

### 5. Check Status

Run:
```bash
nm status
```

Explain to the user:
- Shows the certainty distribution across all edges
- Curiosity triggers are active for edges with slow convergence or high residual error
- Curiosity drives what the system investigates next — it's not random exploration
- Validation pipelines show real-time status

### 6. Fleet Learning

Run all three modes in sequence:
```bash
nm fleet                # status — show fleet topology
nm fleet --mode push    # broadcast local learning to fleet
nm fleet --mode pull    # receive fleet learning into local graph
```

Explain to the user:
- Deployed instances share validated causal vectors with each other — **peer-to-peer, no cloud required**
- Transfer size is ~1KB per update — this fits in a burst radio packet, a satellite ping, or a degraded network connection
- **DDIL-ready**: works in denied, disrupted, intermittent, limited network environments — exactly the scenario in defense, maritime, remote mining, field robotics
- `push` shows the 1.2KB payload: 3 causal edges encoded in CBOR with HMAC-SHA256 signature
- `pull` shows merge strategy: vectors are weighted by validation cycle count, environment tags are preserved (desert ≠ factory ≠ arctic), conflicts resolved by evidence weight
- `status` shows the full fleet topology with sync times, pending queues for offline nodes, and merged certainty across all peers
- Key number: **18.4 KB total bandwidth in 24 hours** for a 5-node fleet learning continuously — compare that to any federated learning approach

For defense audiences specifically:
- No cloud dependency means no vulnerability to network denial
- HMAC-signed payloads prevent adversarial vector injection
- Each node learns independently and shares only validated knowledge — no single point of failure
- A vehicle, drone, or forward-deployed unit can operate disconnected and sync when connectivity returns

---

## Part 2 — Extended Pipeline (LLM Integration)

If the audience is interested in LLM-powered reasoning (edge GPU, cloud deployment), continue with this section. This adds natural language causal reasoning on top of the validated graph.

### 7. Promote Validated Knowledge

Run:
```bash
nm inject
```

Explain to the user:
- This is the **approval gate** — only edges above Z=0.85 get promoted into the LLM
- Each promoted edge shows how many prediction-error cycles validated it and from how many independent endpoints
- Edges below threshold are held back — they continue learning
- The model only absorbs knowledge that has been rigorously validated against reality

### 8. Train on Approved Edges

Run:
```bash
nm train
```

Explain to the user:
- The base model stays **frozen** — its weights are never modified
- Training takes ~8 minutes on a single GPU
- Results are domain-specific — the output shows accuracy, AUROC, and retention numbers for the loaded domain
- Domain knowledge retention: 95%+ (no catastrophic forgetting) across all domains

### 9. Deploy to Edge GPU

Run:
```bash
nm deploy --target edge_gpu
```

Explain to the user:
- This is the LLM-powered deployment — full causal reasoning plus natural language queries
- Runs on NVIDIA Jetson Orin, AMD Kria, RPi 5 + accelerator
- Can answer domain-specific causal questions and reason through the validated graph

Also mention:
```bash
nm deploy --target cloud       # Fleet-wide monitoring, enterprise
```

### 10. Push Updates (No Retrain)

Run:
```bash
nm update
```

Explain to the user:
- New validated edges push to deployed models without retraining
- Edges that crossed Z=0.85 since the last deploy get added automatically
- Multiple deployed instances update simultaneously
- No downtime, no retraining — just updated causal vectors
- Knowledge from one environment propagates to all deployed instances

## Key Talking Points

When the demo is complete, summarize these differentiators:

1. **Branch per domain, instant start**: `git checkout space` gives you a full prior + validation spec committed and ready — then customize from there
2. **Two prompt files, full pipeline**: `prior.md` + `validate.md` are all a customer writes
3. **Expert-reviewable artifacts**: Every stage produces code and data that domain experts can inspect and approve — the graph JSON, validation functions, and deployed inference code
4. **Certainty, not confidence**: Z scores are earned through prediction-error cycles against real sensor data, not statistical confidence intervals
5. **Runs anywhere**: Deploy targets are interchangeable — the same causal graph runs on a $2 MCU (pure C, no neural network), a Jetson Orin, or a cloud server. The MCU target shows the minimum footprint: 60KB total, zero cloud dependency
6. **Fleet learning at 1KB**: Deployed instances share learning peer-to-peer — works in DDIL, burst radio, satellite, contested networks
7. **Updates without retraining**: New validated edges deploy in seconds, not hours
8. **LLM integration is optional**: The core pipeline works without any neural network — the LLM layer adds natural language reasoning on top for edge GPU / cloud targets
9. **Customer owns the loop**: They write the prior, they connect the sensors, they approve what gets promoted and deployed
