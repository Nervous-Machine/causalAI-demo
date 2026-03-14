# Nervous Machine Demo — Agent Skill

Run the full NM learning pipeline demo. Walk through each stage, explain what's happening at every step, and show how causal certainty evolves from hypothesis to deployment-ready.

## Prerequisites

Install the CLI (zero dependencies):

```bash
pip install .
```

## Part 1 — Core Pipeline (MCU / Embedded)

This is the main demo. No neural network, no LLM, no GPU required. Pure causal graph reasoning that compiles to a microcontroller.

### 1. Build the Causal Prior

Run:
```bash
nm init
```

Explain to the user:
- This reads `prior.md` — a domain expert's causal hypotheses about MCU reliability
- 8 nodes represent stressors (thermal cycling, vibration), failure modes (solder fatigue, ESR drift), symptoms (watchdog resets), and outcomes (MCU failure)
- 6 causal edges are created, each with an initial certainty score Z between 0.25–0.40
- These are *hypotheses* — the machine doesn't trust them yet
- Semantic edges (IS_A, PART_OF) give the model ontological context
- Point out the **reviewable JSON graph** — this is what the customer's domain experts approve before anything runs

### 2. Deploy Validation Pipelines

Run:
```bash
nm validate
```

Explain to the user:
- This reads `validate.md` — specifications for real-world data sources
- 3 physical endpoints: thermal chamber, power rail monitor, vibration table
- Each edge gets an error signal: ε = |prediction - observation|
- The error signals close the learning loop — without them, Z never moves
- Point out the **reviewable validation functions** — these show exactly how each sensor connects to each causal edge
- Domain experts can review these functions to verify the prediction-vs-observation logic makes physical sense

### 3. Deploy to Microcontroller

Run:
```bash
nm deploy --target microcontroller
```

Explain to the user:
- Show them what they're building *before* the learning starts — this grounds the whole demo
- For MCU targets: no neural network — pure causal graph reasoning
- Output is a 48KB JSON graph + 12KB C inference engine + 2KB config header
- Runs on any MCU with 64KB+ RAM: AMD Xilinx, STM32, ESP32, etc.
- Point out the **reviewable code** — the output shows both the causal graph JSON and the C inference engine source
- Domain experts can read the exact C code that will run on their MCU: the structs, the traversal logic, the threshold checks
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
- After 5 cycles, voltage_ripple → clock_jitter and thermal_cycling → capacitor_esr_drift should cross the threshold
- The deployed MCU graph gets better with every cycle — same C code, updated certainty values

### 5. Check Status

Run:
```bash
nm status
```

Explain to the user:
- Shows the certainty distribution across all edges
- Two curiosity triggers are active:
  - vibration → solder_joint_fatigue: slowest convergence, may need more data
  - solder_joint_fatigue → watchdog_reset: high residual error, possibly stochastic
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
- Results speak for themselves:
  - Fault diagnosis accuracy: 84.7% (vs 43.1% base model alone)
  - Failure prediction AUROC: 0.91 (capacitor ESR drift)
  - Domain knowledge retention: 95.2% (no catastrophic forgetting)

### 9. Deploy to Edge GPU

Run:
```bash
nm deploy --target edge_gpu
```

Explain to the user:
- This is the LLM-powered deployment — full causal reasoning plus natural language queries
- Runs on NVIDIA Jetson, AMD Kria, RPi 5 + accelerator
- Can answer questions like "Why is this MCU resetting?" and reason causally through the validated graph

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
- Two edges crossed Z=0.85 since the last deploy — they get added automatically
- Three deployed instances updated simultaneously: testbench, production QA, field fleet
- No downtime, no retraining — just updated causal vectors
- The model in the field now knows about solder joint fatigue from test lab data

## Key Talking Points

When the demo is complete, summarize these differentiators:

1. **Two prompt files, full pipeline**: `prior.md` + `validate.md` are all a customer writes
2. **Expert-reviewable artifacts**: Every stage produces code and data that domain experts can inspect and approve — the graph JSON, validation functions, and deployed C code
3. **Certainty, not confidence**: Z scores are earned through prediction-error cycles against real sensor data, not statistical confidence intervals
4. **Compiles to MCU**: Causal knowledge runs on a $2 microcontroller — no cloud dependency, no neural network, no GPU
5. **Fleet learning at 1KB**: Deployed instances share learning peer-to-peer — works in DDIL, burst radio, satellite, contested networks
6. **Updates without retraining**: New validated edges deploy in seconds, not hours
7. **LLM integration is optional**: The core pipeline works without any neural network — the LLM layer adds natural language reasoning on top for edge GPU / cloud targets
8. **Customer owns the loop**: They write the prior, they connect the sensors, they approve what gets promoted and deployed
