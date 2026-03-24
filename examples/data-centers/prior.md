# Data Centers — Thermal Management Causal Prior

## Domain
Data center hot aisle / cold aisle cooling optimization. Per-zone thermal modeling for PUE reduction and hot spot prevention.

## Context
CFD models predict airflow and temperature distribution for the designed rack layout. In practice, workload shifts, rack reconfigurations, missing blanking panels, and seasonal ambient changes cause thermal behavior to diverge from the simulation zone by zone. Operators compensate by overcooling. This prior seeds a per-zone causal model that learns actual thermal dynamics and attributes temperature deviations to specific drivers.

## Causal Hypotheses

### IT Workload → Heat Dissipation
- Server power draw scales with compute utilization. GPU-heavy workloads produce 2–5x the heat of idle racks.
- Workload migration (VM live migration, batch scheduling) causes rapid thermal transients.
- Expected signature: rack inlet temperature tracks workload with 2–10 minute thermal lag depending on airflow velocity.
- Expected weight: dominant driver of zone-level temperature variation.

### CRAC/CRAH Airflow → Cold Aisle Temperature
- Computer Room Air Conditioning/Handling units supply cold air to the raised floor plenum.
- Fan speed, supply air temperature setpoint, and damper positions determine delivery to each zone.
- CFD models assume designed plenum pressure; real pressure varies with tile placement, cable obstructions, and neighboring zones.
- Expected signature: cold aisle temperature responds to CRAC setpoint changes with zone-specific lag and magnitude.
- Expected weight: primary controllable driver; the lever for optimization.

### Ambient / Outside Air Temperature → Economizer Effectiveness
- Free cooling (economizer mode) effectiveness depends on outside air temperature and humidity.
- Seasonal transitions and diurnal cycles shift the balance between mechanical and free cooling.
- Expected signature: PUE increases when ambient temperature exceeds economizer threshold; zone temperatures drift unevenly.
- Expected weight: moderate; seasonal and diurnal modulation of cooling capacity.

### Rack Configuration → Recirculation
- Missing blanking panels, open cable cutouts, and non-standard rack depths create hot air recirculation paths.
- Hot exhaust air bypasses the hot aisle containment and re-enters the cold aisle.
- Expected signature: persistent hot spots at specific rack positions uncorrelated with workload; exacerbated under high total load.
- Expected weight: low-to-moderate as a continuous factor; high as a root cause for specific chronic hot spots.

### Raised Floor Tile Configuration → Airflow Distribution
- Perforated tile placement and open percentage determine per-zone airflow delivery.
- Tiles near CRAC units receive disproportionate flow; far zones are starved.
- Cable bundles under the floor create pressure shadows.
- Expected signature: systematic temperature gradient from CRAC-proximal to CRAC-distal zones.
- Expected weight: moderate; primarily spatial bias rather than temporal variation.

## Expected Interactions
- Workload and CRAC airflow are the primary dynamic pair: workload creates heat, CRAC responds.
- Ambient temperature modulates CRAC effectiveness, creating a three-way interaction.
- Rack configuration and tile layout are quasi-static but create the spatial bias that dynamic factors modulate.
- Recirculation effects worsen non-linearly under high workload — a hot spot that's manageable at 50% load may be critical at 90%.

## Known Gaps
- Server-level fan speed variation (different servers respond differently to thermal stress).
- Humidity effects on cooling efficiency are not modeled.
- Adjacent zone coupling (one zone's exhaust affecting another's intake) may need explicit modeling.
