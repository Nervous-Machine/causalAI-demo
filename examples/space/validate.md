# Space — Validation Endpoints

## Primary Density Sources

### GRACE-FO Accelerometer-Derived Density
- **Source**: GFZ Potsdam / NASA PO.DAAC
- **Cadence**: Sub-minute (10-second intervals, typically aggregated to 10-min or hourly)
- **Coverage**: Single orbital corridor (~490 km, 89° inclination)
- **Format**: CSV with columns [timestamp, latitude, longitude, altitude_km, density_kg_m3]
- **Quality**: Direct accelerometer measurement; no orbital inference required
- **Use**: High-cadence ground truth for temporal lag learning and driver attribution
- **Endpoint**: `https://podaac.jpl.nasa.gov/api/grace-fo/density`
- **Note**: Best available cadence for causal depth; limited spatial coverage

### NOAA/NCEI TLE Debris Catalog
- **Source**: Space-Track.org via NOAA/NCEI
- **Cadence**: ~1–3 day updates per object
- **Coverage**: Global LEO (156 voxels across altitude/latitude/longitude bins)
- **Format**: Two-Line Element sets → ballistic coefficient → inferred density
- **Quality**: Orbital inference introduces noise; daily cadence limits temporal resolution
- **Use**: Spatial breadth for global voxel coverage; architecture validation
- **Endpoint**: `https://space-track.org/basicspacedata/query/class/tle`
- **Auth**: Space-Track account required (free for approved users)
- **Note**: Proved the architecture (16% MAPE vs 85% JB2008) but cannot resolve lag structure

## Space Weather Driver Feeds

### Solar EUV Proxy (F10.7 / S10.7)
- **Source**: NRCan / NOAA SWPC
- **Cadence**: Daily (F10.7); hourly available for S10.7
- **Endpoint**: `https://services.swpc.noaa.gov/json/solar-cycle/observed-solar-cycle-indices.json`
- **Fields**: [date, f10.7_obs, f10.7_adj]

### Geomagnetic Indices (Kp, Dst)
- **Source**: GFZ Potsdam (Kp) / Kyoto WDC (Dst)
- **Cadence**: 3-hourly (Kp); hourly (Dst)
- **Kp endpoint**: `https://kp.gfz-potsdam.de/app/json`
- **Dst endpoint**: `https://wdc.kugi.kyoto-u.ac.jp/dst_realtime/`
- **Fields**: [timestamp, kp_value] / [timestamp, dst_nT]

### Solar Wind (DSCOVR at L1)
- **Source**: NOAA SWPC / DSCOVR
- **Cadence**: 1-minute
- **Endpoint**: `https://services.swpc.noaa.gov/products/solar-wind/plasma-1-day.json`
- **Fields**: [timestamp, density_p_cm3, speed_km_s, temperature_K]
- **Note**: ~30–60 min lead time before Earth impact

### Hemispheric Power Index (Joule Heating Proxy)
- **Source**: NOAA SWPC
- **Cadence**: ~5-minute
- **Endpoint**: `https://services.swpc.noaa.gov/json/ovation_aurora_latest.json`
- **Fields**: [timestamp, north_power_GW, south_power_GW]

### Seasonal-Latitudinal Baseline
- **Source**: Computed internally from epoch (day-of-year, solar declination)
- **Cadence**: Daily
- **Endpoint**: Internal computation, no external feed required
- **Fields**: [doy, solar_declination_deg, expected_seasonal_factor]

## Validation Strategy
1. **Phase 1 — Breadth**: Validate causal graph structure across 156 voxels using TLE debris catalog. Target: confirm driver directionality and relative weight ordering.
2. **Phase 2 — Depth**: Retrain identical pipeline on GRACE-FO corridor with sub-minute cadence. Target: resolve temporal lag structure and achieve >0.85 certainty per driver.
3. **Phase 3 — Prediction**: Use learned lag model to forecast density perturbation evolution from driver impulses. Validate against held-out GRACE-FO windows.
4. **Phase 4 — Fleet**: Ingest live LEO GPS drag residuals as GRACE-FO-comparable cadence source. Validate that fleet-learned causal vectors transfer across orbital corridors.
