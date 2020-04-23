import numpy as np

def rotateFromLatLon(lons, lats, lonSouthPole=18, latSouthPole=-39.25):
    """This function applies a spherical rotation to a set of given latitude and
    longitude coordinates, yielding coordinates in the rotated system.

    Parameters
    ----------
    lons : list, numpy.ndarray
        A one-dimensional list of longitude coordinates

    lats : list, numpy.ndarray
        A one-dimensional list of latitude coordinates

    lonSouthPole : float
        The longitude of the rotated system's south pole

    latSouthPole : float
        The latitude of the rotated system's south pole
    """
    lons = np.radians(lons)
    lats = np.radians(lats)

    theta = np.radians(90+latSouthPole)  # south pole is at 18 deg longitude
    phi = np.radians(lonSouthPole)  # south pole is at -39.25 deg latitude

    x = np.cos(lons) * np.cos(lats)
    y = np.sin(lons) * np.cos(lats)
    z = np.sin(lats)

    x_new = np.cos(theta) * np.cos(phi) * x + np.cos(theta) * \
        np.sin(phi) * y + np.sin(theta) * z
    y_new = -np.sin(phi) * x + np.cos(phi) * y
    z_new = -np.sin(theta) * np.cos(phi) * x - np.sin(theta) * \
        np.sin(phi) * y + np.cos(theta) * z

    rlonCoords = np.degrees(np.arctan2(y_new, x_new))
    rlatCoords = np.degrees(np.arcsin(z_new))

    return rlonCoords, rlatCoords


def rotateToLatLon(rlons, rlats, lonSouthPole=18, latSouthPole=-39.25):
    """This function un-does a spherical rotation to a set of given latitude and
    longitude coordinates (in the rotated), yielding coordinates in the regular 
    longitude and latitude system.

    Parameters
    ----------
    rlons : list, numpy.ndarray
        A one-dimensional list of longitude coordinates in the rotated system

    rlats : list, numpy.ndarray
        A one-dimensional list of latitude coordinates in the rotated system

    lonSouthPole : float
        The longitude of the rotated system's south pole

    latSouthPole : float
        The latitude of the rotated system's south pole
    """
    rlons = np.radians(rlons)
    rlats = np.radians(rlats)

    theta = -np.radians(90+latSouthPole)  # south pole is at 18 deg longitude
    phi = -np.radians(lonSouthPole)  # south pole is at -39.25 deg latitude

    x = np.cos(rlons) * np.cos(rlats)
    y = np.sin(rlons) * np.cos(rlats)
    z = np.sin(rlats)

    x_new = np.cos(theta) * np.cos(phi) * x + np.sin(phi) * \
        y + np.sin(theta) * np.cos(phi) * z
    y_new = -np.cos(theta) * np.sin(phi) * x + np.cos(phi) * \
        y - np.sin(theta) * np.sin(phi) * z
    z_new = -np.sin(theta) * x + np.cos(theta) * z

    lonCoords = np.degrees(np.arctan2(y_new, x_new))
    latCoords = np.degrees(np.arcsin(z_new))

    return lonCoords, latCoords
