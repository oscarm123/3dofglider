import numpy as np

# Earth constants (WGS-84)
mu = 3.986004418e14            # Earth's gravitational parameter, m^3/s^2
Omega_e = 7.2921150e-5         # Earth's rotation rate, rad/s
WGS84_A = 6378137.0            # equatorial radius, m
WGS84_F = 1 / 298.257223563    # flattening
WGS84_E2 = WGS84_F * (2 - WGS84_F)  # first eccentricity squared
OMEGA_VEC = np.array([0.0, 0.0, Omega_e])
G0 = 9.80665                   # standard gravity, m/s^2


# Frame rotation about Z: expresses a vector given in a frame into a frame
# rotated by +angle about Z (passive rotation).
def frame_rot_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[  c,   s, 0.0],
                     [ -s,   c, 0.0],
                     [0.0, 0.0, 1.0]])

# ECI <-> ECEF
def eci_to_ecef(vec_eci, t):
    """Rotate a vector from ECI to ECEF at time t (s)."""
    return frame_rot_z(Omega_e * t) @ vec_eci

def ecef_to_eci(vec_ecef, t):
    """Rotate a vector from ECEF to ECI at time t (s)."""
    return frame_rot_z(Omega_e * t).T @ vec_ecef

def eci_to_ecef_vel(v_eci, r_eci, t):
    """Inertial velocity -> Earth-relative velocity expressed in ECEF."""
    return eci_to_ecef(v_eci - np.cross(OMEGA_VEC, r_eci), t)

def ecef_to_eci_vel(v_ecef, r_ecef, t):
    """Earth-relative ECEF velocity -> inertial velocity expressed in ECI."""
    return ecef_to_eci(v_ecef + np.cross(OMEGA_VEC, r_ecef), t)

# Geodetic <-> ECEF
def geodetic_to_ecef(lat, lon, h):
    """Convert geodetic lat, lon (rad) and altitude (m) to ECEF."""
    N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat)**2)
    x = (N + h) * np.cos(lat) * np.cos(lon)
    y = (N + h) * np.cos(lat) * np.sin(lon)
    z = (N * (1 - WGS84_E2) + h) * np.sin(lat)
    return np.array([x, y, z])

def ecef_to_geodetic(r_ecef, tol=1e-12, max_iter=10):
    """Convert ECEF position to geodetic lat, lon (rad) and altitude (m)."""
    x, y, z = r_ecef
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1 - WGS84_E2))  # initial guess
    for _ in range(max_iter):
        N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat)**2)
        lat_new = np.arctan2(z + WGS84_E2 * N * np.sin(lat), p)
        converged = abs(lat_new - lat) < tol
        lat = lat_new
        if converged:
            break
    N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat)**2)
    # Height formula that stays well-conditioned at both equator and poles
    alt = p * np.cos(lat) + z * np.sin(lat) - WGS84_A**2 / N
    return lat, lon, alt

def eci_to_geodetic(r_eci, t):
    """Convert ECI position at time t to geodetic lat, lon (rad) and altitude (m)."""
    return ecef_to_geodetic(eci_to_ecef(r_eci, t))

# NED <-> ECEF
def ned_to_ecef_matrix(lat, lon):
    """Rotation matrix taking NED vectors to ECEF (columns are N, E, D in ECEF)."""
    slat, clat = np.sin(lat), np.cos(lat)
    slon, clon = np.sin(lon), np.cos(lon)
    return np.array([
        [-slat*clon, -slon, -clat*clon],
        [-slat*slon,  clon, -clat*slon],
        [      clat,   0.0,      -slat]
    ])

def ned_to_ecef(vec_n, lat, lon):
    """Transform vector from NED to ECEF given geodetic lat, lon (rad)."""
    return ned_to_ecef_matrix(lat, lon) @ vec_n

def ecef_to_ned(vec_e, lat, lon):
    """Transform vector from ECEF to NED given geodetic lat, lon (rad)."""
    return ned_to_ecef_matrix(lat, lon).T @ vec_e

# Guidance
def guidance_command(ecef_chaser, ecef_target, ecef_v_chaser, ecef_v_target,
                     kp=1.2, kd=1.2, a_max=5 * G0):
    """PD guidance toward a target point, returned in ECEF and limited to a_max (m/s^2)."""
    a_command = kd * (ecef_v_target - ecef_v_chaser) + kp * (ecef_target - ecef_chaser)
    a_norm = np.linalg.norm(a_command)
    if a_norm > a_max:
        a_command *= a_max / a_norm
    return a_command
