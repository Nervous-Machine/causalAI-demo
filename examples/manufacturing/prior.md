# Manufacturing — Process Quality Causal Prior

## Domain
CNC machining cell producing aluminum aerospace brackets. Per-line quality prediction and root cause attribution for dimensional tolerance drift.

## Context
Quality models are trained on aggregate data across lines and shifts. They flag out-of-spec parts but cannot attribute the root cause to a specific driver for a specific line. This prior seeds a per-line causal model that learns the actual relationships between process parameters and dimensional outcomes.

## Causal Hypotheses

### Tool Wear → Dimensional Drift
- Progressive flank wear increases cutting forces and deflects the workpiece.
- Simulation assumes nominal tool geometry; real tools degrade non-linearly.
- Expected signature: monotonic drift in critical dimensions, accelerating after ~70% of tool life.
- Wear rate varies by material batch hardness, coolant condition, and feed rate.
- Expected weight: primary driver of dimensional drift over a tool's life cycle.

### Ambient Temperature → Thermal Growth
- Machine structure (cast iron bed, ball screws, spindle) expands with temperature.
- Shop floor temperature varies 5–10°C diurnally; more during seasonal transitions.
- Simulation assumes 20°C reference. Real parts are machined on thermally drifting machines.
- Expected signature: systematic bias in Z-axis dimensions correlated with ambient temperature, worst during morning warm-up.
- Expected weight: moderate; dominant during first 2 hours of each shift.

### Material Batch Hardness → Cutting Force Variation
- Aluminum alloy hardness varies ±8% between supplier batches and within billets.
- Harder material → higher cutting forces → more deflection → different dimensional outcome.
- QC measures hardness per incoming lot but process models don't use it as a real-time input.
- Expected signature: step change in dimensional bias coinciding with batch changeover.
- Expected weight: moderate; episodic but significant when present.

### Coolant Concentration → Surface Finish & Tool Life
- Coolant degrades over time: concentration drops, tramp oil accumulates, pH drifts.
- Low concentration → poor lubricity → accelerated tool wear → dimensional and surface finish degradation.
- Refractometer readings taken daily; actual concentration varies within the day.
- Expected signature: correlated degradation in surface finish and accelerated dimensional drift between coolant changes.
- Expected weight: low-to-moderate as direct driver; amplifies tool wear pathway.

### Fixture Clamping Force → Part Distortion
- Thin-wall brackets distort under clamping. Distortion relaxes after unclamping, causing dimensional shift.
- Clamping force set once per setup; actual hydraulic pressure drifts with temperature and seal wear.
- Expected signature: intermittent dimensional outliers in thin-wall features, correlated with clamping pressure sensor readings.
- Expected weight: low under normal conditions; diagnostic flag for fixture maintenance.

## Expected Interactions
- Tool wear and material hardness are coupled: harder batches accelerate wear, compounding dimensional drift.
- Ambient temperature affects both machine thermal growth and coolant viscosity.
- Coolant degradation amplifies tool wear, creating a compounding effect over days.

## Known Gaps
- Spindle bearing preload changes with thermal state; may contribute unattributed Z-axis error.
- Chip evacuation effectiveness varies with geometry and may cause intermittent re-cutting.
- Workholding repeatability (part-to-part positioning variance) is assumed constant but may not be.
