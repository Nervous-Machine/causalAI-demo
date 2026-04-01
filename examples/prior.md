# MCU Reliability — Causal Prior
# Automotive ECU failure mode analysis

## Domain
Microcontroller reliability for automotive ECU applications.
Target: predict and detect failure modes from environmental stressors
and operational telemetry.

## Causal Relationships

### Environmental Stressors → Failure Modes

#### Thermal Cycling → Solder Joint Fatigue
- Mechanism: CTE mismatch between PCB substrate and BGA solder balls
  causes cyclic stress during temperature excursions (-40C to +125C).
- Initial certainty: Z=0.30 (LLM hypothesis from IPC-9701 standards)
- Vectors: strength=0.7, confidence=0.3, context=0.5, stability=0.4

#### Thermal Cycling → Capacitor ESR Drift
- Mechanism: Repeated thermal stress degrades MLCC dielectric,
  increasing equivalent series resistance over time.
- Initial certainty: Z=0.35 (datasheet reference: Murata technical note)
- Vectors: strength=0.6, confidence=0.4, context=0.4, stability=0.5

#### Voltage Ripple → Clock Jitter
- Mechanism: Power supply noise couples into PLL reference,
  causing cycle-to-cycle timing variation.
- Initial certainty: Z=0.40 (known mechanism, well-documented)
- Vectors: strength=0.8, confidence=0.5, context=0.3, stability=0.6

#### Vibration Exposure → Solder Joint Fatigue
- Mechanism: Mechanical resonance in PCB assembly causes
  high-cycle fatigue in solder joints, especially large BGA packages.
- Initial certainty: Z=0.25 (LLM hypothesis, limited field data)
- Vectors: strength=0.5, confidence=0.2, context=0.6, stability=0.3

### Failure Modes → Symptoms

#### Solder Joint Fatigue → Watchdog Reset
- Mechanism: Intermittent open connections cause transient
  MCU lockups, triggering hardware watchdog.
- Initial certainty: Z=0.30 (LLM hypothesis)
- Vectors: strength=0.6, confidence=0.3, context=0.5, stability=0.3

#### Capacitor ESR Drift → MCU Functional Failure
- Mechanism: Decoupling capacitor degradation allows voltage
  transients to violate MCU operating specs, causing logic errors.
- Initial certainty: Z=0.35 (datasheet reference)
- Vectors: strength=0.7, confidence=0.4, context=0.4, stability=0.4

## Semantic Relationships
- solder_joint_fatigue IS_A mechanical_failure
- capacitor_esr_drift IS_A electrical_degradation
- clock_jitter PART_OF timing_subsystem
- watchdog_reset RELATED_TO mcu_functional_failure

## Nodes
- thermal_cycling: type=stressor, unit=cycles, range=[-40C, +125C]
- voltage_ripple: type=stressor, unit=mV_pp, range=[0, 200]
- vibration_exposure: type=stressor, unit=g_rms, range=[0, 20]
- solder_joint_fatigue: type=failure_mode, metric=crack_length_um
- capacitor_esr_drift: type=failure_mode, metric=esr_milliohms
- clock_jitter: type=symptom, metric=jitter_ps_rms
- watchdog_reset: type=symptom, metric=resets_per_hour
- mcu_functional_failure: type=outcome, metric=binary
