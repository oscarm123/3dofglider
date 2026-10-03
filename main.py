from datetime import datetime
from pathlib import Path

import numpy as np

from helper_functions import (G0, OMEGA_VEC, mu, ecef_to_eci, ecef_to_eci_vel,
                              eci_to_ecef, eci_to_ecef_vel, eci_to_geodetic,
                              geodetic_to_ecef, guidance_command, ned_to_ecef)
from trajectory_plotter_plotly import plot_states_plotly


# Glider model
class Glider3DOF:
    def __init__(self, mass, ecef_target, cone_radius=0.5, C_D=0.8, a_max=5 * G0):
        self.mass = mass
        self.target_ecef = ecef_target
        self.A_ref = np.pi * cone_radius**2  # frontal area of the cone
        self.C_D = C_D
        self.a_max = a_max

    def drag_force(self, r_eci, v_eci, alt):
        """Drag force in ECI (N), opposing the velocity relative to the rotating atmosphere."""
        # altitude-based density
        rho0 = 1.225           # kg/m^3 at sea level
        scale_height = 8000.0  # m
        rho = rho0 * np.exp(-alt / scale_height)

        v_rel = v_eci - np.cross(OMEGA_VEC, r_eci)
        return -0.5 * rho * self.C_D * self.A_ref * np.linalg.norm(v_rel) * v_rel

    def dynamics(self, t, state):
        r_eci = state[0:3]
        v_eci = state[3:6]

        _, _, alt = eci_to_geodetic(r_eci, t)
        F_eci = self.drag_force(r_eci, v_eci, alt)

        # Gravity in ECI
        a_gravity = -mu * r_eci / np.linalg.norm(r_eci)**3

        # Guidance command is computed in ECEF, so rotate it into ECI
        r_ecef = eci_to_ecef(r_eci, t)
        v_ecef = eci_to_ecef_vel(v_eci, r_eci, t)
        a_command_ecef = guidance_command(r_ecef, self.target_ecef, v_ecef, np.zeros(3),
                                          a_max=self.a_max)
        a_command = ecef_to_eci(a_command_ecef, t)

        # Total acceleration
        a_eci = a_gravity + F_eci / self.mass + a_command
        return np.hstack((v_eci, a_eci))

# Integrator (RK4)
def rk4_step(fun, t, y, dt):
    k1 = fun(t, y)
    k2 = fun(t + dt/2, y + dt/2 * k1)
    k3 = fun(t + dt/2, y + dt/2 * k2)
    k4 = fun(t + dt,   y + dt * k3)
    return y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

def propagate_eci(state0, t0, tf, dt, glider):
    n_steps = round((tf - t0) / dt)
    times = t0 + dt * np.arange(n_steps + 1)
    states = np.zeros((len(times), len(state0)))
    state = state0.copy()

    for i, t in enumerate(times):
        # record state
        states[i] = state

        # compute current geodetic altitude
        _, _, alt = eci_to_geodetic(state[0:3], t)

        if alt < 0:
            print(f"Simulation ended: glider hit the ground at t = {t:.2f} s (alt = {alt:.1f} m)")
            # truncate arrays up to and including this step
            return times[:i+1], states[:i+1]

        if i == n_steps:
            break

        # advance state
        state = rk4_step(glider.dynamics, t, state, dt)

    return times, states

# Example usage
if __name__ == "__main__":

    dt = 1e-2
    t0 = 0.0
    tf = 120.0

    # Target: equator, 0.15 deg E, 30 km
    ecef_target = geodetic_to_ecef(lat=np.deg2rad(0.0), lon=np.deg2rad(0.15), h=30e3)

    # Initial geodetic conditions (0 deg N, 0 deg E, 50 km)
    lat0, lon0, h0 = np.deg2rad(0.0), np.deg2rad(0.0), 50e3
    r_ecef0 = geodetic_to_ecef(lat0, lon0, h0)
    r_eci0 = ecef_to_eci(r_ecef0, t0)

    # Initial Earth-relative NED velocity (north 400 m/s, east 0, down 0)
    v_ned0 = np.array([400.0, 0.0, 0.0])
    v_ecef0 = ned_to_ecef(v_ned0, lat0, lon0)
    v_eci0 = ecef_to_eci_vel(v_ecef0, r_ecef0, t0)

    # State vector
    state0 = np.hstack((r_eci0, v_eci0))

    tick = datetime.now()
    glider = Glider3DOF(mass=100.0, ecef_target=ecef_target)
    times, states = propagate_eci(state0, t0, tf, dt, glider)
    diff = datetime.now() - tick
    # Print final ECI position
    print("Final ECI Position (m):", states[-1, :3])
    print("Simulation took " + str(diff.total_seconds()) + " seconds")

    # states is an (N,6) array: [x,y,z,vx,vy,vz]
    r_eci = states[:, :3]   # shape (N,3)
    v_eci = states[:, 3:]   # shape (N,3)
    r_ecef = np.vstack([eci_to_ecef(r, t) for r, t in zip(r_eci, times)])
    v_ecef = np.vstack([eci_to_ecef_vel(v, r, t)
                        for r, v, t in zip(r_eci, v_eci, times)])
    states_ecef = np.hstack((r_ecef, v_ecef))

    html_file = Path(__file__).parent / "glider.html"
    plot_states_plotly(times, states_ecef, html_file=html_file, fontsize=13,
                       target_ecef=ecef_target)
