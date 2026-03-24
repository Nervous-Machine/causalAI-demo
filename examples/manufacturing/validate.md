# Manufacturing — Validation Endpoints

## Quality Measurement

### CMM (Coordinate Measuring Machine)
- **Source**: In-line or near-line CMM (e.g., Zeiss, Hexagon)
- **Cadence**: Per-part (every part or sampling plan, e.g., 1-in-5)
- **Format**: [timestamp, part_id, feature_name, nominal_mm, actual_mm, deviation_mm, tolerance_mm, pass_fail]
- **Endpoint**: `http://192.168.10.20:8080/api/cmm_results`
- **Use**: Primary error signal — deviation from nominal is the ground truth for dimensional quality

### Surface Roughness Gauge
- **Source**: In-process or post-process profilometer
- **Cadence**: Per-part or per-batch
- **Format**: [timestamp, part_id, surface_id, Ra_um, Rz_um]
- **Endpoint**: `http://192.168.10.21:8080/api/surface_finish`
- **Use**: Surface quality validation; correlate with tool wear and coolant condition

## Process Parameters

### CNC Machine Controller
- **Source**: Machine tool controller (e.g., Fanuc, Siemens via MTConnect/OPC-UA)
- **Cadence**: Real-time (100 Hz for axis positions; per-program-block for feeds/speeds)
- **Format**: MTConnect XML or OPC-UA JSON [timestamp, spindle_load_pct, feed_rate_mmpm, axis_positions[3], program_block]
- **Endpoint**: `mtconnect://192.168.10.10:5000/current` or `opcua://192.168.10.10:4840`
- **Use**: Monitor cutting forces (via spindle load proxy), feed rates, and cycle time for process signature

### Tool Management System
- **Source**: Tool presetter or in-spindle probe
- **Cadence**: Per-tool-change or per-shift
- **Format**: [timestamp, tool_id, tool_type, flank_wear_mm, edge_radius_um, cuts_since_new, remaining_life_pct]
- **Endpoint**: `http://192.168.10.30:8080/api/tool_status`
- **Use**: Track tool wear progression; correlate with dimensional drift and cutting force changes

## Environmental Sensors

### Shop Floor Temperature & Humidity
- **Source**: Wireless sensor network (e.g., Monnit, Sensirion)
- **Cadence**: 1-minute
- **Format**: [timestamp, sensor_id, zone, temp_c, humidity_pct]
- **Endpoint**: `http://192.168.10.40:8080/api/environment`
- **Use**: Correlate ambient conditions with thermal growth bias; track diurnal and seasonal patterns

### Coolant Monitoring
- **Source**: Inline refractometer + pH sensor on coolant tank
- **Cadence**: Hourly (automated) or daily (manual entry)
- **Format**: [timestamp, machine_id, concentration_pct, pH, tramp_oil_pct, temp_c]
- **Endpoint**: `http://192.168.10.50:8080/api/coolant`
- **Use**: Track coolant degradation; correlate with surface finish and tool life acceleration

## Material & Fixturing

### Incoming Material QC
- **Source**: Quality lab (hardness tester, material cert)
- **Cadence**: Per-lot
- **Format**: [timestamp, lot_id, alloy, hardness_HRB, supplier, cert_url]
- **Endpoint**: `http://192.168.10.60:8080/api/material_qc`
- **Use**: Material batch hardness as process input; detect step changes at batch boundaries

### Fixture Clamping Pressure
- **Source**: Hydraulic pressure sensor on fixture
- **Cadence**: Per-cycle (clamping event)
- **Format**: [timestamp, fixture_id, clamp_pressure_bar, target_pressure_bar, delta_bar]
- **Endpoint**: `http://192.168.10.70:8080/api/fixture_clamp`
- **Use**: Monitor clamping force drift; correlate with thin-wall distortion outliers

## Validation Strategy
1. **Baseline**: Collect 500 parts with full CMM data + all process/environmental feeds. Establish per-feature deviation distributions under "normal" conditions.
2. **Driver isolation**: Introduce controlled perturbations — tool change, batch change, temperature shift — and verify the causal model attributes the dimensional change to the correct driver.
3. **Causal learning**: Run `nm learn` cycles. Target: identify per-feature driver weights for this specific line/machine combination.
4. **Curiosity triggers**: Watch for persistent residuals unexplained by the five hypothesized drivers. Likely candidates: spindle thermal growth, chip re-cutting, fixture repeatability.
5. **Cross-line transfer**: Deploy to a second CNC cell machining the same part. Validate which causal vectors transfer (e.g., material-hardness→force relationship) and which are line-specific (e.g., thermal growth profile).
