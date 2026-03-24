# Data Centers — Validation Endpoints

## Temperature Sensors

### Rack Inlet Temperature (Cold Aisle)
- **Source**: Per-rack temperature sensors (BMC/IPMI or external wireless sensors)
- **Cadence**: 1-minute
- **Format**: [timestamp, rack_id, zone_id, inlet_temp_c, position (top/mid/bottom)]
- **Endpoint**: `http://dcim.local:8080/api/sensors/rack_inlet`
- **Use**: Primary error signal — predicted vs. actual inlet temperature per rack is the core learning target

### Rack Exhaust Temperature (Hot Aisle)
- **Source**: Per-rack rear-door sensors or hot aisle containment sensors
- **Cadence**: 1-minute
- **Format**: [timestamp, rack_id, zone_id, exhaust_temp_c, delta_t_c]
- **Endpoint**: `http://dcim.local:8080/api/sensors/rack_exhaust`
- **Use**: Compute per-rack heat rejection (delta_t × airflow); detect recirculation events

### Under-Floor Plenum Temperature
- **Source**: Wireless sensors in raised floor plenum
- **Cadence**: 5-minute
- **Format**: [timestamp, sensor_id, zone_id, plenum_temp_c, plenum_pressure_pa]
- **Endpoint**: `http://dcim.local:8080/api/sensors/plenum`
- **Use**: Validate CRAC delivery effectiveness; detect pressure shadows from cable obstructions

## IT Load

### Per-Rack Power Draw
- **Source**: Intelligent PDUs (e.g., Raritan, ServerTech, Vertiv)
- **Cadence**: 1-minute
- **Format**: [timestamp, rack_id, pdu_id, power_kw, current_a, voltage_v]
- **Endpoint**: `http://dcim.local:8080/api/power/rack` or `snmp://pdu-rack01.local`
- **Use**: Real-time workload proxy; compute heat dissipation per rack

### Server Utilization (Optional, Higher Resolution)
- **Source**: Prometheus / node_exporter / IPMI
- **Cadence**: 15-second
- **Format**: [timestamp, server_id, rack_id, cpu_util_pct, gpu_util_pct, power_w]
- **Endpoint**: `http://prometheus.local:9090/api/v1/query?query=node_cpu_utilization`
- **Use**: Fine-grained workload attribution; distinguish CPU vs. GPU thermal signatures

## Cooling Infrastructure

### CRAC/CRAH Unit Telemetry
- **Source**: BMS (Building Management System) via BACnet or Modbus
- **Cadence**: 1-minute
- **Format**: [timestamp, unit_id, supply_temp_c, return_temp_c, fan_speed_pct, cooling_capacity_kw, mode (mechanical/economizer)]
- **Endpoint**: `bacnet://192.168.20.10:47808` or `http://bms.local:8080/api/crac`
- **Use**: Track CRAC response to setpoint changes; learn per-zone delivery lag and magnitude

### Chilled Water Plant (if applicable)
- **Source**: BMS
- **Cadence**: 5-minute
- **Format**: [timestamp, supply_temp_c, return_temp_c, flow_rate_lpm, chiller_power_kw]
- **Endpoint**: `http://bms.local:8080/api/chiller`
- **Use**: Upstream cooling capacity constraint; detect plant-level limitations affecting zone delivery

## Ambient Conditions

### Outside Air Temperature & Humidity
- **Source**: Weather station on building or local weather API
- **Cadence**: 5-minute
- **Format**: [timestamp, temp_c, humidity_pct, dewpoint_c, wet_bulb_c]
- **Endpoint**: `http://weather-station.local:8080/api/current` or `https://api.openweathermap.org/data/3.0/onecall`
- **Use**: Economizer effectiveness modeling; seasonal PUE correlation

## Configuration (Quasi-Static)

### Rack Layout & Blanking Panel Status
- **Source**: DCIM asset database or manual audit
- **Cadence**: Updated on change (event-driven)
- **Format**: [rack_id, zone_id, row, position, ru_populated, ru_blanked, ru_open, last_audit_date]
- **Endpoint**: `http://dcim.local:8080/api/assets/rack_config`
- **Use**: Identify recirculation risk factors; correlate with persistent hot spots

### Raised Floor Tile Map
- **Source**: DCIM or manual survey
- **Cadence**: Updated on change
- **Format**: [tile_id, zone_id, tile_type (solid/25%/50%/grate), x_position, y_position]
- **Endpoint**: `http://dcim.local:8080/api/assets/tile_map`
- **Use**: Airflow distribution model; explain spatial temperature gradients

## Validation Strategy
1. **Baseline**: Collect 7 days of all feeds at steady-state workload. Establish per-zone thermal response characteristics and steady-state temperature distributions.
2. **Workload perturbation**: Coordinate with operations to migrate a known workload between racks. Verify the causal model tracks the thermal transient in both source and destination zones.
3. **CRAC setpoint test**: Lower one CRAC unit's supply temperature by 2°C. Measure per-zone response delay and magnitude. Validate the model's prediction vs. reality.
4. **Causal learning**: Run `nm learn` cycles. Target: per-zone driver weights and thermal lag structure for workload, CRAC, ambient, and configuration factors.
5. **Optimization pilot**: Use the causal model to raise CRAC setpoints in zones where the model predicts safe headroom. Monitor for 48 hours. Measure PUE reduction.
6. **Cross-hall transfer**: Deploy to a second data hall. Validate which causal vectors transfer (e.g., workload→temperature relationship for same server type) and which are hall-specific (e.g., plenum pressure profile).
