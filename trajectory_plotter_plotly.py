import numpy as np
import plotly.offline as pyo
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from helper_functions import WGS84_A, ecef_to_geodetic

TRAJ_LINE = dict(width=4, color='#1f77b4')
TARGET_MARKER = dict(symbol='x', size=14, color='#d62728')
START_MARKER = dict(symbol='circle', size=10, color='#2ca02c')
END_MARKER = dict(symbol='square', size=10, color='#7f7f7f')

def plot_states_plotly(times, states, html_file="glider.html", fontsize=16, target_ecef=None):
    """Plot an ECEF trajectory, optionally with a fixed target point, to an HTML report.

    states are ECEF [x, y, z, vx, vy, vz] with shape (N, 6); target_ecef is a 3-vector (m).
    """
    figure_width = 1300
    figure_height = 600

    # --- 1) Build the figures ---
    # compute LLA and a flat-Earth track (East/North relative to the start point)
    lats, lons, alts = [], [], []
    for st in states:
        lat, lon, alt = ecef_to_geodetic(st[:3])
        lats.append(np.degrees(lat)); lons.append(np.degrees(lon)); alts.append(alt)
    lat0, lon0 = np.radians(lats[0]), np.radians(lons[0])

    def flat_earth(lat_deg, lon_deg):
        east = (np.radians(lon_deg) - lon0) * np.cos(lat0) * WGS84_A
        north = (np.radians(lat_deg) - lat0) * WGS84_A
        return east, north

    x, y = flat_earth(np.array(lats), np.array(lons))

    has_target = target_ecef is not None
    if has_target:
        t_lat, t_lon, t_alt = ecef_to_geodetic(target_ecef)
        t_lat, t_lon = np.degrees(t_lat), np.degrees(t_lon)
        t_x, t_y = flat_earth(t_lat, t_lon)
        dist = np.linalg.norm(states[:, :3] - target_ecef, axis=1)
        i_min = int(np.argmin(dist))

    # LLA vs Time
    fig1 = make_subplots(rows=3, cols=1, shared_xaxes='all',
                         subplot_titles=("Latitude (°)", "Longitude (°)", "Altitude (m)"),
                         vertical_spacing=0.06)
    for row, values in enumerate((lats, lons, alts), start=1):
        fig1.add_trace(go.Scatter(x=times, y=values, mode='lines', line=TRAJ_LINE,
                                  name='Glider', showlegend=(row == 1)), row=row, col=1)
    if has_target:
        for row, value in enumerate((t_lat, t_lon, t_alt), start=1):
            fig1.add_hline(y=value, line=dict(color=TARGET_MARKER['color'], dash='dash'),
                           row=row, col=1)
        fig1.add_trace(go.Scatter(x=[None], y=[None], mode='lines', name='Target',
                                  line=dict(color=TARGET_MARKER['color'], dash='dash')),
                       row=1, col=1)
    fig1.update_layout(width=figure_width, height=figure_height, title="LLA vs Time",
                       font=dict(size=fontsize))
    fig1.update_xaxes(title="Time (s)", row=3, col=1)

    # 2D ground track
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=x, y=y, mode='lines', line=TRAJ_LINE, name='Glider'))
    fig2.add_trace(go.Scatter(x=[x[0]], y=[y[0]], mode='markers', marker=START_MARKER, name='Start'))
    fig2.add_trace(go.Scatter(x=[x[-1]], y=[y[-1]], mode='markers', marker=END_MARKER, name='End'))
    if has_target:
        fig2.add_trace(go.Scatter(x=[t_x], y=[t_y], mode='markers', marker=TARGET_MARKER,
                                  name='Target'))
    fig2.update_layout(title="Flat-Earth Ground Track", xaxis_title="East (m)",
                       yaxis_title="North (m)", font=dict(size=fontsize),
                       width=figure_width, height=figure_height)
    fig2.update_yaxes(scaleanchor="x", scaleratio=1)

    # Earth-relative speed
    speed = np.linalg.vector_norm(states[:, 3:6], axis=1)
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=times, y=speed, mode='lines', line=TRAJ_LINE, name='Speed'))
    fig3.update_layout(title="Ground-Relative Speed", xaxis_title="Time (s)",
                       yaxis_title="Speed (m/s)", font=dict(size=fontsize),
                       width=figure_width, height=figure_height)

    # 3D trajectory
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter3d(x=x, y=y, z=alts, mode='lines', line=TRAJ_LINE, name='Glider'))
    fig4.add_trace(go.Scatter3d(x=[x[0]], y=[y[0]], z=[alts[0]], mode='markers',
                                marker=dict(START_MARKER, size=6), name='Start'))
    fig4.add_trace(go.Scatter3d(x=[x[-1]], y=[y[-1]], z=[alts[-1]], mode='markers',
                                marker=dict(END_MARKER, size=6), name='End'))
    if has_target:
        fig4.add_trace(go.Scatter3d(x=[t_x], y=[t_y], z=[t_alt], mode='markers',
                                    marker=dict(symbol='x', size=8, color=TARGET_MARKER['color']),
                                    name='Target'))
    fig4.update_layout(title="3D Flat-Earth Trajectory",
                       scene=dict(xaxis_title='East (m)',
                                  yaxis_title='North (m)',
                                  zaxis_title='Altitude (m)'),
                       font=dict(size=fontsize),
                       width=figure_width, height=figure_height)

    figs = [fig1, fig2, fig3, fig4]

    # Distance to target
    if has_target:
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=times, y=dist, mode='lines', line=TRAJ_LINE,
                                  name='Distance'))
        fig5.add_trace(go.Scatter(x=[times[i_min]], y=[dist[i_min]], mode='markers+text',
                                  marker=dict(TARGET_MARKER, symbol='circle', size=10),
                                  text=[f"min {dist[i_min]:.0f} m at {times[i_min]:.1f} s"],
                                  textposition='top center', name='Closest point'))
        fig5.update_layout(title="Distance to Target", xaxis_title="Time (s)",
                           yaxis_title="Distance (m)", font=dict(size=fontsize),
                           width=figure_width, height=figure_height)
        figs.append(fig5)

    # --- 2) Convert each to an HTML <div> snippet ---
    divs = [pyo.plot(fig, include_plotlyjs=('cdn' if i == 0 else False), output_type='div')
            for i, fig in enumerate(figs)]

    # --- 3) Combine & write a single HTML file (overwrites existing) ---
    html = ("<!DOCTYPE html>\n<html><head><meta charset=\"utf-8\"></head><body>\n"
            + "\n".join(divs)
            + "\n</body></html>")

    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"Wrote updated trajectory to {html_file}. Open it in your browser and refresh to see the new plots.")
