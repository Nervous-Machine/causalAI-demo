# Robotics — Sim-to-Real Calibration Causal Prior

## Domain
Industrial robotic arm (6-DOF) operating in a pick-and-place cell. Sim-to-real transfer calibration for trajectory accuracy and grasp reliability.

## Context
The robot was trained in simulation (MuJoCo/Isaac Sim) with idealized physics. Deployed performance degrades over time as real-world conditions diverge from simulation assumptions. This prior seeds a per-robot causal model that learns each unit's actual dynamics and identifies which sim assumptions are breaking.

## Causal Hypotheses

### Joint Friction Drift → Trajectory Error
- Simulation assumes constant Coulomb + viscous friction coefficients per joint.
- Real friction increases with wear, temperature, and lubrication degradation.
- Joints 4–6 (wrist) degrade faster due to higher duty cycles in pick-and-place.
- Expected signature: gradual, monotonic increase in position error at high-acceleration segments.
- Expected weight: primary driver of trajectory error after 500+ operating hours.

### Payload Mass Variation → Grasp Force Error
- Simulation trains on nominal payload mass ± 10% uniform distribution.
- Real payloads have batch-dependent density variation, asymmetric center of mass, and surface friction differences.
- Grasp force overshoot → product damage; grasp force undershoot → drops.
- Expected weight: dominant driver of grasp failure events.

### Ambient Temperature → Joint Compliance
- Thermal expansion affects link lengths (µm-scale) and lubricant viscosity.
- Simulation assumes 22°C constant. Factory floor ranges 15–35°C seasonally.
- Morning cold-start behavior differs from afternoon steady-state.
- Expected signature: systematic bias in first-hour trajectories, correlated with ambient temperature.
- Expected weight: moderate; significant during seasonal transitions.

### Surface Condition → Placement Accuracy
- Conveyor belt wear, debris accumulation, and product residue change effective friction at pickup/placement zones.
- Simulation assumes uniform, static surface coefficients.
- Expected signature: placement offset drift correlated with hours since last belt maintenance.
- Expected weight: low-to-moderate; episodic rather than continuous.

### Controller Latency Jitter → Dynamic Tracking Error
- Simulation assumes deterministic control loop timing (1 kHz).
- Real-time OS jitter, sensor bus contention, and compute load cause ±200µs variance.
- Effect amplified during high-speed trajectory segments.
- Expected signature: intermittent spikes in tracking error uncorrelated with mechanical factors.
- Expected weight: low under normal conditions; diagnostic flag for compute issues.

## Expected Interactions
- Joint friction and ambient temperature are coupled: higher temperature reduces friction short-term but accelerates lubricant degradation long-term.
- Payload mass variation and grasp force are directly causal; surface condition modulates the effect.
- Controller latency is independent of mechanical factors but may mask other error sources if not isolated.

## Known Gaps
- Backlash in gear reducers is not modeled in the simulation and may contribute unattributed error.
- Vibration coupling between adjacent cells is not captured.
- Tool-tip deflection under load may need a separate compliance model.
