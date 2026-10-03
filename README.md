# 3dofglider

A three-degree-of-freedom (point-mass) trajectory simulator for a guided glider, written in Python with NumPy. The vehicle is propagated in an Earth-centered inertial (ECI) frame over a rotating WGS-84 Earth, and the results are written to an interactive Plotly report.

## Features

- **Point-mass dynamics in ECI**, integrated with a fixed-step 4th-order Runge–Kutta (RK4) scheme
- **Rotating WGS-84 Earth**, with conversions between ECI, ECEF, NED and geodetic coordinates
- **Gravity**: central-body (point-mass) model
- **Atmosphere**: exponential density model (sea-level density 1.225 kg/m³, 8 km scale height)
- **Drag**: cone frontal-area model, using velocity relative to the rotating atmosphere
- **Guidance**: PD law that steers toward a fixed ECEF target point, with a configurable acceleration limit (5 g by default)
- **Ground impact detection**: stops the run when geodetic altitude drops below zero
- **Interactive HTML report** with latitude/longitude/altitude vs. time, ground track, ground-relative speed and a 3D trajectory

## Getting started

Tested with Python 3.14.

```bash
git clone git@github.com:oscarm123/3dofglider.git
cd 3dofglider

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
python main.py
```

The run takes a couple of seconds and writes `glider.html` next to `main.py`. Open it in a browser to view the plots. The page loads Plotly from a CDN, so viewing it needs an internet connection.

Example console output:

```text
Final ECI Position (m): [ 6.40564117e+06  4.62961623e+04 -1.63741064e+03]
Simulation took 1.54 seconds
Wrote updated trajectory to .../3dofglider/glider.html. Open it in your browser and refresh to see the new plots.
```

## Configuring a run

Scenario settings live in the `if __name__ == "__main__":` block of [`main.py`](main.py):

| Setting | Default | Description |
| --- | --- | --- |
| `dt` | `0.01` s | Integration step |
| `t0`, `tf` | `0`, `60` s | Start and end time |
| `lat0`, `lon0`, `h0` | 0°, 0°, 50 km | Initial geodetic position |
| `v_ned0` | `[400, 0, 0]` m/s | Initial Earth-relative velocity (north, east, down) |
| `ecef_target` | 0°, 0.15° E, 30 km | Guidance target point |

Vehicle parameters are passed to `Glider3DOF`:

| Parameter | Default | Description |
| --- | --- | --- |
| `mass` | `100` kg | Vehicle mass |
| `cone_radius` | `0.5` m | Base radius, used for the frontal area |
| `C_D` | `0.8` | Drag coefficient |
| `a_max` | `5 g` | Guidance acceleration limit |

The guidance gains `kp` and `kd` (both `1.2` by default) are arguments of `guidance_command` in [`helper_functions.py`](helper_functions.py).

## Project structure

| File | Contents |
| --- | --- |
| [`main.py`](main.py) | `Glider3DOF` vehicle model, RK4 integrator, propagation loop and example scenario |
| [`helper_functions.py`](helper_functions.py) | WGS-84 constants, frame transforms (ECI/ECEF/NED/geodetic) and the guidance law |
| [`trajectory_plotter_plotly.py`](trajectory_plotter_plotly.py) | Builds the Plotly figures and writes the HTML report |

## Model assumptions and limitations

- **Point mass only**: there's no attitude, so no rotational dynamics, moments or angle of attack.
- **No lift model**: the guidance acceleration is applied directly, as if from a thruster. A real glider can only steer by changing its lift vector (bank angle and angle of attack), so trajectories can be more aggressive than a glider could fly.
- **Simple environment**: point-mass gravity (no J2), a simple exponential atmosphere and no wind.
- **Earth rotation**: ECI and ECEF are aligned at `t = 0` and differ only by Earth's rotation (no precession, nutation or polar motion).
- **Flat-Earth plots**: the ground track and 3D plots use a local flat-Earth approximation, which is fine for short ranges.

## Roadmap

- [ ] Lift model with bank-angle / angle-of-attack guidance
- [ ] Standard atmosphere model (e.g. US Standard Atmosphere 1976)
- [ ] J2 gravity
- [ ] Adaptive integration with an impact event (e.g. `scipy.integrate.solve_ivp`)
- [ ] Unit tests for the frame transforms
