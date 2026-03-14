# MCU Reliability — Validation Specification
# Error signal sources for automotive ECU causal model

## Validation Endpoints

### Thermal Chamber API
- URL: http://testlab.local:8080/thermal
- Format: JSON
- Interval: 30 seconds
- Measures:
  - junction_temp_c: MCU die temperature (thermocouple)
  - ambient_temp_c: Chamber ambient temperature
  - cycle_count: Accumulated thermal cycles
  - dwell_time_min: Time at temperature extreme
- Validates edges:
  - thermal_cycling → solder_joint_fatigue
  - thermal_cycling → capacitor_esr_drift
  - solder_joint_fatigue → watchdog_reset (via reset counter)
- Quality: 0.9 (calibrated lab equipment)

### Power Rail Monitor
- URL: http://testlab.local:8080/power
- Format: JSON
- Interval: 10 seconds
- Measures:
  - vcc_ripple_mv: Peak-to-peak voltage ripple on Vcc rail
  - esr_milliohms: Measured ESR of decoupling capacitors (LCR meter)
  - cap_temp_c: Capacitor surface temperature
  - rail_voltage_v: DC rail voltage
- Validates edges:
  - voltage_ripple → clock_jitter
  - capacitor_esr_drift → mcu_functional_failure
  - thermal_cycling → capacitor_esr_drift (ESR trend data)
- Quality: 0.85 (automated measurement, periodic calibration)

### Vibration Table DAQ
- URL: http://testlab.local:8080/vibration
- Format: JSON
- Interval: 100 milliseconds
- Measures:
  - accel_g_rms: RMS acceleration magnitude
  - freq_spectrum_hz: Dominant frequency components (array)
  - duration_hours: Accumulated vibration exposure time
  - resonance_detected: Boolean — PCB resonance mode active
- Validates edges:
  - vibration_exposure → solder_joint_fatigue
- Quality: 0.8 (DAQ system, noise floor considerations)

## Error Signal Definitions

### thermal_cycling → solder_joint_fatigue
- Prediction: Coffin-Manson model — cycles_to_failure = C * (delta_T)^(-n)
- Observation: X-ray inspection crack measurement (weekly)
- Error signal: epsilon = |predicted_cycles_to_failure - actual_cycles_to_crack|
- Normalization: Divide by expected lifetime (10,000 cycles)

### thermal_cycling → capacitor_esr_drift
- Prediction: Linear ESR increase model — ESR(t) = ESR_0 + k * cycles
- Observation: LCR meter ESR measurement (every 100 cycles)
- Error signal: epsilon = |predicted_esr - measured_esr| / ESR_0
- Normalization: Relative to initial ESR

### voltage_ripple → clock_jitter
- Prediction: Linear coupling model — jitter_ps = a * ripple_mv + b
- Observation: Oscilloscope TIE measurement
- Error signal: epsilon = |predicted_jitter - measured_jitter| / nominal_period
- Normalization: Relative to clock period

### vibration_exposure → solder_joint_fatigue
- Prediction: Steinberg fatigue model — cycles = f(displacement, frequency)
- Observation: Daisy-chain resistance monitoring + X-ray
- Error signal: epsilon = |predicted_resistance_change - measured|
- Normalization: Relative to initial resistance

### solder_joint_fatigue → watchdog_reset
- Prediction: Probabilistic model — P(reset) = f(crack_length, vibration_state)
- Observation: Watchdog reset counter log
- Error signal: epsilon = |predicted_reset_rate - actual_reset_rate|
- Normalization: Per hour

### capacitor_esr_drift → mcu_functional_failure
- Prediction: Threshold model — failure when ESR > critical_esr
- Observation: MCU functional test pass/fail
- Error signal: epsilon = |predicted_failure_probability - actual_failure_rate|
- Normalization: Binary outcome (0 or 1)

## Bradford Hill Criteria (applied to all edges)
- Strength: Measured via causal vector strength dimension
- Consistency: Tracked across thermal chamber runs
- Temporality: Stressor always precedes failure mode
- Biological gradient: More cycles/ripple → more degradation
- Plausibility: Known physics mechanisms for all edges
- Coherence: Consistent with IPC, JEDEC, AEC-Q standards
