import numpy as np

# -----------------------------------------------

def cartesian_to_spherical(x, y, z, degree=False):
    """Converts cartesian to spherical coordinates."""

    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    if degree:
        phi = np.rad2deg(phi)
        theta = np.rad2deg(theta)
    return r, phi, theta


def spherical_to_cartesian(r, phi, theta, degree=False):
    """Converts spherical to cartesian coordinates."""

    if degree:
        phi = np.deg2rad(phi)
        theta = np.deg2rad(theta)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return float(x), float(y), float(z)

def cartesian_to_cylindrical(x, y, z, degree=False):
    """Converts Cartesian to Cylindrical coordinates (r, phi, z)."""

    r = np.sqrt(x**2 + y**2)  # Radial distance
    phi = np.arctan2(y, x)    # Azimuthal angle
    if degree:
        phi = np.rad2deg(phi) # Convert phi to degrees if required
    return r, phi, z


def cylindrical_to_cartesian(r, phi, z, degree=False):
    """Converts Cylindrical to Cartesian coordinates (x, y, z)."""

    if degree:
        phi = np.deg2rad(phi) # Convert phi to radians if given in degrees
    x = r * np.cos(phi)       # X-coordinate
    y = r * np.sin(phi)       # Y-coordinate
    return float(x), float(y), float(z)  
