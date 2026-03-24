# Space — Thermospheric Density Causal Prior

## Domain
Low Earth Orbit (LEO) thermospheric density modeling for satellite drag prediction.

## Context
Operational models (e.g., JB2008, NRLMSISE-00) predict bulk thermospheric density but lack driver-resolved attribution. They cannot tell an operator which driver caused a density spike, how long its influence persists, or how perturbations propagate across orbital shells. This prior seeds a voxel-level causal model that decomposes density into individual physical drivers.

## Causal Hypotheses

### Solar EUV Flux → Thermospheric Density
- Solar EUV heats the upper atmosphere, increasing scale height and density at satellite altitudes.
- F10.7 index is the standard proxy; S10.7 (EUV-specific) provides better temporal resolution.
- Influence is quasi-steady-state with ~1-day propagation lag from solar disk to thermospheric response.
- Expected weight: dominant driver under quiet geomagnetic conditions.

### Geomagnetic Activity → Thermospheric Density
- Geomagnetic storms (Kp, Dst) drive rapid density enhancements via Joule heating and particle precipitation.
- Response onset within 1–3 hours; recovery timescale 1–5 days depending on storm intensity.
- Density enhancement is latitude-dependent: strongest at high latitudes, propagating equatorward.
- Expected weight: dominant driver during storm conditions; secondary during quiet periods.

### Solar Wind Dynamic Pressure → Thermospheric Density
- Solar wind ram pressure modulates magnetospheric compression, indirectly affecting thermospheric energy input.
- Influence is weaker than direct EUV or geomagnetic pathways but provides early-warning signal (upstream).
- Lag structure: 30–90 minutes from L1 measurement to thermospheric response.
- Expected weight: low-to-moderate; primarily a leading indicator.

### Seasonal-Latitudinal Pattern → Thermospheric Density
- Annual and semi-annual variations driven by solar declination, thermospheric composition changes, and interhemispheric transport.
- Known semi-annual anomaly: density maxima near equinoxes, minima near solstices.
- Latitude-dependent: different voxels experience different seasonal amplitudes.
- Expected weight: moderate; provides baseline modulation.

### Joule Heating → Thermospheric Density
- High-latitude Joule heating from auroral electrojet currents produces localized density enhancements.
- Proxy: Hemispheric Power Index (HPI) or auroral electrojet index (AE).
- Propagation: equatorward via traveling atmospheric disturbances (TADs) on ~2–6 hour timescales.
- Expected weight: significant at high latitudes; attenuated but measurable at mid-latitudes.

## Expected Interactions
- EUV and geomagnetic activity are partially correlated (both solar-driven) but operate on different timescales.
- Joule heating is a sub-mechanism of geomagnetic activity; the model should learn whether it adds independent explanatory power.
- Seasonal pattern modulates the baseline that all other drivers perturb.

## Known Gaps
- Sub-voxel density gradients (within a single orbital shell) are not resolved at TLE cadence.
- Nightside density response may have different lag structure than dayside.
- Neutral wind effects on drag are not captured by density alone.
