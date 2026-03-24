# Robotics — Validation Endpoints

## Trajectory & Dynamics Sensors

### Joint Encoder Feedback
- **Source**: Robot controller (e.g., Fanuc, KUKA, UR via RTDE)
- **Cadence**: 1 kHz (1 ms)
- **Format**: JSON or binary stream [timestamp, joint_positions[6], joint_velocities[6], joint_torques[6]]
- **Endpoint**: `rtde://192.168.1.100:30004` (UR example) or `ros2 topic /joint_states`
- **Use**: Primary error signal — compare commanded vs. actual joint positions/torques

### End-Effector Pose (External Tracking)
- **Source**: Vision system (e.g., OptiTrack, Intel RealSense, overhead camera)
- **Cadence**: 30–120 Hz
- **Format**: [timestamp, x, y, z, qx, qy, qz, qw, confidence]
- **Endpoint**: `http://192.168.1.50:8080/api/pose` or `ros2 topic /tool_pose`
- **Use**: Ground truth for trajectory accuracy; independent of joint encoder drift

### Grasp Force Sensor
- **Source**: Force/torque sensor at wrist (e.g., ATI, OnRobot)
- **Cadence**: 100–1000 Hz
- **Format**: [timestamp, fx, fy, fz, tx, ty, tz]
- **Endpoint**: `http://192.168.1.60:8080/api/ft_data`
- **Use**: Grasp force validation; detect overshoot/undershoot vs. commanded force

## Environmental Sensors

### Ambient Temperature
- **Source**: Thermocouple or RTD near robot base
- **Cadence**: 1-minute
- **Format**: [timestamp, temp_c, humidity_pct]
- **Endpoint**: `http://192.168.1.70:8080/api/environment`
- **Use**: Correlate thermal conditions with trajectory bias and friction drift

### Conveyor Surface Condition
- **Source**: Friction sensor or proxy (slip detection from grasp events)
- **Cadence**: Per-cycle (each pick/place event)
- **Format**: [timestamp, pickup_slip_detected, placement_offset_mm, belt_hours_since_maintenance]
- **Endpoint**: `http://192.168.1.80:8080/api/conveyor_status`
- **Use**: Surface condition tracking for placement accuracy model

### Controller Diagnostics
- **Source**: Robot controller internal telemetry
- **Cadence**: Per-cycle or 10 Hz
- **Format**: [timestamp, control_loop_time_us, cpu_load_pct, bus_latency_us]
- **Endpoint**: `http://192.168.1.100:8080/api/diagnostics`
- **Use**: Isolate controller latency jitter as an independent error source

## Payload Reference

### Payload Scale (Inline Weigh Station)
- **Source**: Inline scale at pickup station
- **Cadence**: Per-cycle
- **Format**: [timestamp, mass_g, estimated_com_offset_mm]
- **Endpoint**: `http://192.168.1.90:8080/api/payload`
- **Use**: Ground truth for payload mass variation; correlate with grasp force errors

## Validation Strategy
1. **Baseline**: Record 1000 pick-and-place cycles under nominal conditions. Establish per-joint error distributions and grasp success rate.
2. **Sim comparison**: Run identical trajectories in simulation. Compute residual (sim prediction − real observation) per joint, per cycle.
3. **Causal learning**: Run `nm learn` cycles. Target: identify which drivers (friction, temperature, payload, surface, latency) explain the residual and at what weight.
4. **Curiosity triggers**: Monitor for persistent residuals unexplained by the five hypothesized drivers. Flag as potential missing factors (backlash, vibration coupling, tool deflection).
5. **Fleet propagation**: Deploy to multiple robots in the cell. Validate that causal vectors from Robot #1 improve predictions on Robot #2 in similar configuration.
