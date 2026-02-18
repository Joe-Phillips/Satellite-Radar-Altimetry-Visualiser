#----------------------------------------------------------------------
# Overview
#----------------------------------------------------------------------

"""
This script simulates radar altimetry waveforms from input topography
using a simplified geometric ray-casting approach.

The simulator works by casting rays from a satellite position down to
a surface defined by the input topography. Each ray finds its first
intersection with the surface, and the slant range of that intersection
is mapped to a waveform bin. Contributions are weighted by an antenna
gain pattern and accumulated to form the output waveform.

Two modes are provided:
  - 2D (SAR-analogous):  Rays are cast across a 1D cross-track profile,
                         collapsing the along-track dimension. This is
                         loosely analogous to SAR mode, where the
                         along-track footprint is Doppler-compressed,
                         but the comparison is approximate.
  - 3D (LRM-analogous):  Rays are cast across a 2D circular footprint
                         on a hexagonal grid. This is loosely analogous
                         to LRM, where the full beam-limited footprint
                         is illuminated. Again, the comparison is
                         approximate.

The 2D case produces reasonable waveform shapes and performs fairly
well for the purposes of this simulator. The 3D case inherits the same
geometric simplifications, but any limitations are compounded by the
extra spatial dimension. Most notably, the simulated waveform exhibits
a sharper drop in return power after the peak than would be seen in a
real LRM waveform, which typically has a longer and more gradual
trailing tail. This issue is present in the 2D case too but is much
less apparent there.

This is a geometric simulator only. It does not model:
  - Scattering behaviour based on surface type or roughness
    (e.g. subsurface or volume scattering, speckle)
  - Mode-specific instrument properties (LRM, SAR, SARIn, etc.)
  - Any aspect of on-board or ground processing

The waveform shape is determined entirely by surface geometry and the
antenna gain taper. The output is normalised to a maximum of 1.
"""


#----------------------------------------------------------------------
# Imports
#----------------------------------------------------------------------

import numpy as np
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.optimize import root_scalar
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math


#----------------------------------------------------------------------
# 2D Simulation (SAR-analogous)
#----------------------------------------------------------------------

def simulate_altimetry_waveform_2d(
    topography,
    range_window_height=239.8339664,
    sat_altitude=717000,
    footprint_width=15000,
    num_rays=512,
    num_bins=128,
    return_contribution_angle_scale=False,
):
    """Simulates a 2D radar altimetry waveform from a given topography profile.

    Models pulse returns from a satellite altimeter by casting rays across a
    footprint and finding their intersections with the surface. Contributions
    are weighted by an antenna gain pattern and accumulated into waveform bins.

    Args:
        topography (array-like): 1D array of surface heights, normalised to [0, 1]
            where 0.5 corresponds to the centre of the range window.
        range_window_height (float): Height of the range window in metres.
            Defaults to 239.8339664 (CryoSat-2).
        sat_altitude (float): Satellite altitude in metres. Defaults to 717000
            (CryoSat-2).
        footprint_width (float): Across-track width of the simulated footprint
            in metres. Defaults to 15000 (CryoSat-2).
        num_rays (int): Number of rays to cast across the footprint. Defaults
            to 512.
        num_bins (int): Number of bins in the output waveform. Defaults to 128.
        return_contribution_angle_scale (bool): If True, also returns the
            per-ray antenna gain scaling, useful for visualisation. Defaults
            to False.

    Returns:
        np.ndarray: Simulated waveform, normalised to a maximum of 1.
        tuple(np.ndarray, np.ndarray): If return_contribution_angle_scale is
            True, returns (simulated_waveform, contribution_angle_scale).

    To-do:
        Accept topography in metres and allow for truncation of values outside the range window.
    """

    #----------------------------------------------------------------------
    # Validate and scale topography
    #----------------------------------------------------------------------

    topography = np.asarray(topography)
    if len(np.shape(topography)) != 1:
        raise ValueError("Topography should be a 1D array representing the elevation profile across the footprint width")
    if np.min(topography) < 0 or np.max(topography) > 1:
        print("Warning: topography values should be in the range [0, 1] where 0.5 is the centre of the range window - values outside this range will be clipped")
        topography = np.clip(topography, 0, 1)

    # Convert normalised topography to metres
    topography = topography * range_window_height

    #----------------------------------------------------------------------
    # Build topography spline and range window geometry
    #----------------------------------------------------------------------

    topography_x = np.linspace(-footprint_width / 2, footprint_width / 2, len(topography))
    topography_spline = interp1d(topography_x, topography, fill_value="extrapolate", kind="slinear")

    # Compute the curved bottom edge of the range window across the swath
    range_window_bottom_x = np.linspace(-footprint_width / 2, footprint_width / 2, num_rays)
    range_window_bottom_x[np.argwhere(range_window_bottom_x == 0)] = 0.00001  # Avoid degenerate root at x=0
    range_window_bottom_z = sat_altitude - np.sqrt(sat_altitude**2 - np.square(range_window_bottom_x))

    #----------------------------------------------------------------------
    # Cast rays and find surface intersections
    #----------------------------------------------------------------------

    sat_coord = np.array([0, sat_altitude])
    return_bins = np.full(num_rays, np.nan)

    for ray in range(num_rays):
        ray_func = interp1d(
            [sat_coord[0], range_window_bottom_x[ray]],
            [sat_coord[1], range_window_bottom_z[ray]],
            fill_value="extrapolate",
        )

        # Find where the ray and topography surface intersect
        f = lambda x: topography_spline(x) - ray_func(x)
        try:
            root = root_scalar(
                f,
                x0=range_window_bottom_x[ray],
                bracket=[-footprint_width / 2, footprint_width / 2],
            ).root
            bisection = [root, topography_spline(root)]
        except Exception:
            continue

        # Map intersection distance from range window bottom to a waveform bin
        ray_vec = sat_coord - np.column_stack((range_window_bottom_x[ray], range_window_bottom_z[ray]))
        bisec_vec = sat_coord - bisection
        bisec_dist_from_rw_bottom = np.linalg.norm(ray_vec - bisec_vec)
        return_bins[ray] = np.rint((1 - bisec_dist_from_rw_bottom / range_window_height) * num_bins)

    #----------------------------------------------------------------------
    # Accumulate bin contributions weighted by antenna gain
    #----------------------------------------------------------------------

    return_bins = np.clip(np.array(return_bins, dtype="int"), 0, num_bins - 1)
    return_contributions = np.zeros((num_rays, num_bins))
    return_contributions[np.arange(num_rays), return_bins] = 1

    # Scale contributions by off-nadir antenna gain pattern
    # See https://www.sciencedirect.com/science/article/pii/S0273117705009348 Eq. 1
    theta = np.arctan2(
        np.abs(range_window_bottom_x),
        np.sqrt(sat_altitude**2 - range_window_bottom_x**2),
    )
    phi = 0
    contribution_angle_scale = np.exp(
        (-theta**2) * ((np.cos(phi) / 0.0133)**2 + (np.sin(phi) / 0.0148)**2)
    )
    return_contributions = return_contributions * contribution_angle_scale[:, np.newaxis]

    #----------------------------------------------------------------------
    # Sum, normalise and return
    #----------------------------------------------------------------------

    simulated_waveform = np.sum(return_contributions, axis=0)
    simulated_waveform = simulated_waveform / np.max(simulated_waveform)

    if return_contribution_angle_scale:
        return simulated_waveform, contribution_angle_scale
    return simulated_waveform


#----------------------------------------------------------------------
# 3D Simulation (LRM-analogous)
#----------------------------------------------------------------------

def get_range_window_bottom_3d(num_rays, footprint_radius, range_window_height, sat_altitude):
    """Simulates a 3D radar altimetry waveform from a given topography grid.

    Models pulse returns from a satellite altimeter by casting rays across a
    circular footprint and finding their intersections with the surface.
    Contributions are weighted by an antenna gain pattern and accumulated into
    waveform bins.

    Args:
        num_rays (int): Target number of rays (grid points) to generate within
            the circular footprint. The actual number may differ slightly due to
            hexagonal packing geometry.
        footprint_radius (float): Radius of the circular footprint in metres.
        range_window_height (float): Height of the range window in metres.
        sat_altitude (float): Satellite altitude in metres.

    Returns:
        tuple:
            - range_window_bottom (np.ndarray): Array of shape (N, 3) containing
              the (x, y, z) coordinates of each ray endpoint at the base of the
              range window, where N is the actual number of points generated.
            - num_points_outer_ring (int): Number of points in the outermost row
              of the hexagonal grid, useful for diagnostics.
    """

    # Calculate the area of the circular footprint
    area = np.pi * footprint_radius**2

    # Calculate the desired point spacing to achieve approximately num_rays points
    d = np.sqrt(area / num_rays)

    # Define hexagonal grid row spacing (sqrt(3)/2 * d for equilateral triangles)
    dy = np.sqrt(3) / 2 * d
    ny = int(np.floor(footprint_radius / dy))  # number of rows in each direction

    # Generate points on a hexagonal grid, then filter to the circular footprint
    xall, yall = [], []
    for i in range(-ny, ny + 1):
        y = dy * i  # y-coordinate for this row

        # The maximum x-extent of the circle at this y is sqrt(r^2 - y^2)
        max_x = np.sqrt(max(footprint_radius**2 - y**2, 0))

        # For even rows, place points symmetrically at integer multiples of d
        if i % 2 == 0:
            nx = int(np.floor(max_x / d))
            x = np.arange(-nx, nx + 1) * d
        # For odd rows, shift by half a spacing for the hexagonal offset
        else:
            nx = int(np.floor((max_x - d / 2) / d))
            x = np.arange(-nx - 0.5, nx + 0.5 + 1) * d

        # Discard any points that lie outside the circular footprint. The
        # half-spacing shift on odd rows can push a small number of points
        # just beyond the boundary even after the max_x correction above
        x = x[np.sqrt(x**2 + y**2) <= footprint_radius]

        xall.extend(x)
        yall.extend([y] * len(x))
        num_points_outer_ring = len(x)

    xy_in_footprint = np.column_stack((xall, yall))

    # If the hexagonal packing results in fewer points than requested, update num_rays to match the actual count
    if len(xy_in_footprint) != num_rays:
        # print(f"Note: Hexagonal packing resulted in {len(xy_in_footprint)} rays, which differs from the requested {num_rays}. Adjusting to match the actual count.")
        num_rays = len(xy_in_footprint)

    # Compute the curved base of the range window for each grid point
    range_window_bottom = np.zeros((num_rays, 3))
    for i, (rx, ry) in enumerate(xy_in_footprint):
        radial_distance = np.sqrt(rx**2 + ry**2)
        # Nudge exact-zero coordinates to avoid a degenerate ray through nadir
        if rx == 0:
            rx += 1e-5
        if ry == 0:
            ry += 1e-5
        range_window_bottom[i, 0:2] = [rx, ry]
        range_window_bottom[i, 2] = sat_altitude - np.sqrt(sat_altitude**2 - radial_distance**2)

    return range_window_bottom, num_points_outer_ring


def compute_intersection_3d(ray, sat_coord, range_window_bottom, topography_interpolator, root_bracket=[0, 1]):
    """Finds the intersection of a single ray with the topography surface.

    The ray is parametrised by t in [0, 1], where t=0 is the satellite position
    (nadir, [0, 0, sat_altitude]) and t=1 is the corresponding point at the base
    of the range window. A root finder locates the t at which the ray elevation
    equals the topography elevation.

    Args:
        ray (int): Index of the ray in range_window_bottom to process.
        sat_coord (np.ndarray): 3-element array [x, y, z] of the satellite position.
            Must be a numpy array to ensure correct arithmetic with range_window_bottom.
        range_window_bottom (np.ndarray): Array of shape (N, 3) containing the
            (x, y, z) coordinates of each ray's range window bottom endpoint,
            as returned by get_range_window_bottom_3d.
        topography_interpolator (RegularGridInterpolator): Interpolator for the
            2D topography surface, accepting (x, y) coordinate tuples in metres.
        root_bracket (list): Two-element list [t_min, t_max] bounding the
            parametric search interval. Defaults to [0, 1].

    Returns:
        np.ndarray: 3-element array [x, y, z] of the intersection point, or an
            array of NaN values if no valid intersection was found.
    """

    def intersection_function(t):
        """Returns ray_z - topo_z at parametric position t along the ray."""
        x = t * range_window_bottom[ray, 0]
        y = t * range_window_bottom[ray, 1]
        ray_z = sat_coord[2] + (range_window_bottom[ray, 2] - sat_coord[2]) * (x - sat_coord[0]) / (range_window_bottom[ray, 0] - sat_coord[0])
        z_topo = topography_interpolator((x, y))  # tuple call for RegularGridInterpolator
        return ray_z - z_topo

    # Find the parametric position of the intersection within the bracket
    try:
        solution = root_scalar(intersection_function, bracket=root_bracket)
    except Exception:
        return np.full(3, np.nan)

    # If the solver converged, evaluate and return the 3D intersection coordinates
    if solution.converged:
        t_intersect = solution.root
        intersection_x = t_intersect * range_window_bottom[ray, 0]
        intersection_y = t_intersect * range_window_bottom[ray, 1]
        intersection_z = topography_interpolator((intersection_x, intersection_y))  # tuple call, consistent with above
        return np.array([intersection_x, intersection_y, float(intersection_z)])

    return np.full(3, np.nan)


def simulate_altimetry_waveform_3d(
    topography,
    range_window_height=239.8339664,
    sat_altitude=717000,
    footprint_radius=7500,
    num_rays=2048,
    num_bins=128,
    return_contribution_angle_scale=False,
):
    """Simulates a 3D radar altimetry waveform from a given topography grid.

    Models pulse returns from a satellite altimeter by casting rays across a
    circular footprint and finding their intersections with the surface.
    Contributions are weighted by an antenna gain pattern and accumulated into
    waveform bins.

    Args:
        topography (array-like): 2D array of surface heights, normalised to [0, 1]
            where 0.5 corresponds to the centre of the range window. The array
            is assumed to span the full circular footprint diameter in both axes.
        range_window_height (float): Height of the range window in metres.
            Defaults to 239.8339664 (CryoSat-2).
        sat_altitude (float): Satellite altitude in metres. Defaults to 717000
            (CryoSat-2).
        footprint_radius (float): Radius of the circular footprint in metres.
            Defaults to 7500 (CryoSat-2 LRM).
        num_rays (int): Target number of rays to cast within the footprint.
            The actual count may differ slightly due to hexagonal packing.
            Defaults to 8192.
        num_bins (int): Number of bins in the output waveform. Defaults to 128.
        return_contribution_angle_scale (bool): If True, also returns the
            per-ray antenna gain scaling, useful for visualisation. Defaults
            to False.

    Returns:
        np.ndarray: Simulated waveform, normalised to a maximum of 1.
        tuple(np.ndarray, np.ndarray): If return_contribution_angle_scale is
            True, returns (simulated_waveform, contribution_angle_scale).

    To-do:
        Accept topography in metres and allow for truncation of values outside the range window.
    """

    #----------------------------------------------------------------------
    # Validate and scale topography
    #----------------------------------------------------------------------

    topography = np.asarray(topography)
    if len(np.shape(topography)) != 2:
        raise ValueError("Topography should be a 2D array representing the elevations within the circular footprint.")
    if np.min(topography) < 0 or np.max(topography) > 1:
        print("Warning: topography values should be in the range [0, 1] where 0.5 is the centre of the range window - values outside this range will be clipped")
        topography = np.clip(topography, 0, 1)

    # Convert normalised topography to metres
    topography = topography * range_window_height

    #----------------------------------------------------------------------
    # Build topography interpolator and range window geometry
    #----------------------------------------------------------------------

    # Build a regular grid spanning the footprint diameter in both axes
    nx, ny = np.shape(topography)
    x = np.linspace(-footprint_radius, footprint_radius, nx)
    y = np.linspace(-footprint_radius, footprint_radius, ny)
    topography_interpolator = RegularGridInterpolator((x, y), topography, bounds_error=False, method="slinear")

    # Compute the curved base of the range window on a hexagonal grid within the footprint
    range_window_bottom, _ = get_range_window_bottom_3d(num_rays, footprint_radius, range_window_height, sat_altitude)
    num_rays = len(range_window_bottom)  # update after hexagonal packing

    #----------------------------------------------------------------------
    # Cast rays and find surface intersections
    #----------------------------------------------------------------------

    sat_coord = np.array([0, 0, sat_altitude], dtype=float)
    return_bins = np.full(num_rays, np.nan)

    for ray in range(num_rays):

        # Find the 3D intersection of this ray with the topography surface
        intersection = compute_intersection_3d(ray, sat_coord, range_window_bottom, topography_interpolator, root_bracket=[0, 1])

        # Map intersection distance from range window bottom to a waveform bin
        if not np.isnan(intersection).any():
            ray_vec = sat_coord - range_window_bottom[ray]
            intersec_vec = sat_coord - intersection
            intersec_dist_from_rw_bottom = np.linalg.norm(ray_vec - intersec_vec)
            return_bins[ray] = np.rint(
                (1 - intersec_dist_from_rw_bottom / range_window_height) * num_bins
            )

    #----------------------------------------------------------------------
    # Accumulate bin contributions weighted by antenna gain
    #----------------------------------------------------------------------

    return_bins = np.clip(np.array(return_bins, dtype="int"), 0, num_bins - 1)
    return_contributions = np.zeros((num_rays, num_bins))
    return_contributions[np.arange(num_rays), return_bins] = 1

    # Scale contributions by off-nadir antenna gain pattern
    # See https://www.sciencedirect.com/science/article/pii/S0273117705009348 Eq. 1
    range_window_bottom_radius = np.sqrt(np.square(range_window_bottom[:, 0]) + np.square(range_window_bottom[:, 1]))
    theta = np.arctan(range_window_bottom_radius / np.sqrt(sat_altitude**2 - np.square(range_window_bottom_radius)))
    phi = np.arctan2(range_window_bottom[:, 1], range_window_bottom[:, 0])  # azimuth angle in the xy-plane
    contribution_angle_scale = np.exp(
        (-theta**2) * ((np.cos(phi) / 0.0133)**2 + (np.sin(phi) / 0.0148)**2)
    )
    return_contributions = return_contributions * contribution_angle_scale[:, np.newaxis]

    #----------------------------------------------------------------------
    # Sum, normalise and return
    #----------------------------------------------------------------------

    simulated_waveform = np.sum(return_contributions, axis=0)
    simulated_waveform /= np.max(simulated_waveform)

    if return_contribution_angle_scale:
        return simulated_waveform, contribution_angle_scale
    return simulated_waveform