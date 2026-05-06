# ----------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------

import numpy as np
from scipy.interpolate import interp1d, RegularGridInterpolator, splev, splrep
from scipy.optimize import root_scalar
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from simulate_altimetry_waveform import (
    simulate_altimetry_waveform_2d,
    simulate_altimetry_waveform_3d,
    get_range_window_bottom_3d,
    compute_intersection_3d,
)
import base64

# ----------------------------------------------------------------------
# Marker colour palette 
# ----------------------------------------------------------------------

colour_poca = "rgb(255,237,41)" # yellow
colour_nadir = "rgb(255,29,206)" # magenta
colour_le = "rgb(0,0,0)" # gray

# ----------------------------------------------------------------------
# Leading-edge detection
# ----------------------------------------------------------------------

def get_leading_edge(
    waveform,
    tracker_range,
    reference_bin_index,
    smoothing_window_width,
    range_bin_size,
    wf_oversampling_factor,
):
    """Detect the leading edge in a waveform and compute its range.

    Args:
        waveform (numpy.ndarray): Input waveform data.
        tracker_range (float): Tracker range value.
        reference_bin_index (int): Index of the reference bin.
        smoothing_window_width (int): Width of the smoothing window.
        range_bin_size (float): Distance between each range bin (m).
        wf_oversampling_factor (int): Waveform oversampling factor.

    Returns:
        tuple: (le_index_start, le_index_end, le_range_start, le_range_end).
            Index values are in the original (non-oversampled) waveform bin space.
            Range values are NaN when only indices are needed (tracker_range=0).
    """

    noise_threshold = 0.3 # reject waveform if mean noise exceeds this
    le_threshold_id = 0.05 # power must exceed noise by this to be a leading edge candidate
    le_threshold_dp = 0.2 # minimum normalised amplitude change to accept a leading edge

    le_index_start = np.nan
    le_index_end = np.nan
    le_range_start = np.nan
    le_range_end = np.nan

    while True:
        wf_norm = waveform / max(waveform)

        # Pseudo-Gaussian smoothing (3 passes of sliding-average)
        kernel = np.ones(smoothing_window_width) / smoothing_window_width
        wf_smooth = np.convolve(
            np.convolve(
                np.convolve(wf_norm, kernel, mode="same"),
                kernel, mode="same",
            ),
            kernel, mode="same",
        )

        wf_sorted = np.sort(wf_norm)
        wf_noise_mean = np.mean(wf_sorted[:6])
        if wf_noise_mean > noise_threshold:
            break

        # Oversample via spline
        oversampling_interval = 1 / wf_oversampling_factor
        bin_indices = np.arange(0, len(waveform), oversampling_interval)
        wf_interp = splev(bin_indices, splrep(range(len(waveform)), wf_smooth))
        wf_interp_d1 = np.gradient(wf_interp, oversampling_interval)

        le_index_prev = 0
        le_dp = 0
        while le_index_prev < len(bin_indices) - wf_oversampling_factor:
            candidates = np.where(
                (wf_interp > (wf_noise_mean + le_threshold_id))
                & (bin_indices > bin_indices[le_index_prev + wf_oversampling_factor])
            )
            if np.size(candidates) == 0:
                break
            le_idx = candidates[0][0]

            peaks = np.where((wf_interp_d1 <= 0) & (bin_indices > bin_indices[le_idx]))
            if np.size(peaks) == 0:
                break
            first_peak = peaks[0][0]
            le_dp = wf_interp[first_peak] - wf_interp[le_idx]
            le_index_prev = first_peak

            if le_dp > le_threshold_dp:
                le_index_start = le_idx / wf_oversampling_factor
                le_index_end   = first_peak / wf_oversampling_factor
                break

        if not np.isnan(le_index_start):
            le_range_start = tracker_range - (reference_bin_index - le_index_start) * range_bin_size
            le_range_end   = tracker_range + (le_index_end - reference_bin_index)   * range_bin_size
        break

    return le_index_start, le_index_end, le_range_start, le_range_end


# ----------------------------------------------------------------------
# 2D Plotting (SAR-analogous)
# ----------------------------------------------------------------------

def animate_altimetry_waveform_2d(
    topography,
    output_path=None,
    num_rays_to_display=25,
    range_window_top=1.0,
    range_window_bottom=0.0,
    show_poca=True,
    show_range_window=True,
    show_rays=True,
    wf_noise_amplitude=0.01,
    satellite_image_path="s3.png",
    show_nadir=True,
    show_leading_edge=True,
):
    """Generates an animated 2D altimetry waveform simulation using Plotly.

    Simulates the travel of radar pulses from a satellite towards a surface defined
    by the input topography, animating both the spatial pulse propagation and the
    corresponding waveform build-up.

    Args:
        topography (array-like): 1D array of surface heights, normalised to [0, 1].
        output_path (str, optional): If provided, saves the animation as an HTML file
            at this path (without extension). Defaults to None.
        num_rays_to_display (int): Number of rays to show in the spatial plot.
            Defaults to 25.
        range_window_top (float): Upper bound of the range window, in [0, 1].
            Defaults to 1.0.
        range_window_bottom (float): Lower bound of the range window, in [0, 1].
            Defaults to 0.0.
        show_poca (bool): Whether to mark the Point of Closest Approach.
            Defaults to True.
        show_range_window (bool): Whether to display the range window on both subplots.
            Defaults to True.
        show_rays (bool): Whether to display the individual ray paths. Defaults to True.
        wf_noise_amplitude (float): Amplitude of uniform noise added to the waveform,
            as a fraction of the peak waveform value. Defaults to 0.01.
        satellite_image_path (str, optional): Path to a satellite image to display at
            the ray origin. Silently skipped if not found. Defaults to "s3.png".
        show_nadir (bool): Whether to mark the nadir point (directly below the
            satellite) on both subplots. Defaults to True.
        show_leading_edge (bool): Whether to mark the leading-edge start and end on
            the waveform subplot. Defaults to True.

    Returns:
        plotly.graph_objects.Figure: The animated Plotly figure.
    """

    # ------------------------------------------------------------------
    # Constants
    # ------------------------------------------------------------------

    PLOT_HEIGHT = 800
    FPS = 24
    ANIMATION_LENGTH_S = 3
    NUM_FRAMES = FPS * ANIMATION_LENGTH_S
    NUM_WAVEFORM_BINS = 128
    PLOT_RAY_ORIGIN_HEIGHT = 3  # Controls pulse travel time and total animation length
    PLOT_ACROSS_TRACK_WIDTH = 2
    # The topography is scaled to [0.2, 0.8] so the simulator (defined on [0, 1])
    # has room for spillover beyond the topography extents.
    TOPO_SCALE_LOW = 0.2
    TOPO_SCALE_HIGH = 0.8
    TOPO_SCALE_RANGE = TOPO_SCALE_HIGH - TOPO_SCALE_LOW  # 0.6

    # ------------------------------------------------------------------
    # Validate and prepare inputs
    # ------------------------------------------------------------------

    topography = np.asarray(np.clip(topography, 0, 1), dtype=float)
    range_window_top = np.clip(range_window_top, 0.0, 1.0)
    range_window_bottom = np.clip(range_window_bottom, 0.0, range_window_top)
    range_window_scale = range_window_top - range_window_bottom
    range_window_centre = np.mean([range_window_top, range_window_bottom])

    topography_scaled = TOPO_SCALE_LOW + TOPO_SCALE_RANGE * topography

    # ------------------------------------------------------------------
    # Simulate waveform
    # ------------------------------------------------------------------

    waveform, contribution_angle_scale = simulate_altimetry_waveform_2d(
        topography_scaled, return_contribution_angle_scale=True
    )

    contribution_angle_scale = np.interp(
        np.linspace(0, 1, num_rays_to_display),
        np.linspace(0, 1, len(contribution_angle_scale)),
        contribution_angle_scale,
    )

    # ------------------------------------------------------------------
    # Detect leading edge (on the clean waveform, before noise is added)
    # ------------------------------------------------------------------

    le_detected = False
    le_index_start = le_index_end = np.nan

    if show_leading_edge:
        try:
            le_index_start, le_index_end, _, _ = get_leading_edge(
                waveform=waveform,
                tracker_range=0.0,
                reference_bin_index=NUM_WAVEFORM_BINS // 2,
                smoothing_window_width=5,
                range_bin_size=1.0,
                wf_oversampling_factor=10,
            )
            le_detected = not (np.isnan(le_index_start) or np.isnan(le_index_end))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Range window geometry in the spatial plot
    # ------------------------------------------------------------------

    # Curved bottom edge of the range window across the swath
    range_window_bottom_x = np.linspace(
        -PLOT_ACROSS_TRACK_WIDTH / 2, PLOT_ACROSS_TRACK_WIDTH / 2, num_rays_to_display
    )
    range_window_bottom_x[range_window_bottom_x == 0] = 1e-5  # Avoid degenerate ray at x=0

    range_window_bottom_z_at_nadir = range_window_centre - range_window_scale / 2
    off_nadir_offset = PLOT_RAY_ORIGIN_HEIGHT - np.sqrt(
        PLOT_RAY_ORIGIN_HEIGHT ** 2 - range_window_bottom_x ** 2
    )
    range_window_bottom_z = off_nadir_offset + range_window_bottom_z_at_nadir

    # Vertical offset between swath-edge and nadir range window positions
    edge_vs_nadir_offset_z = np.nanmax(range_window_bottom_z) - np.nanmin(range_window_bottom_z)

    # ------------------------------------------------------------------
    # Build and pad the full waveform array
    # ------------------------------------------------------------------

    # The 128-bin waveform maps normalised height h to bin via (1 - h) * 128, so
    # it spans the full PLOT_RAY_ORIGIN_HEIGHT of one-way travel time.  The leading
    # zeros represent the in-flight pulse before any surface return is received.
    waveform_distance_covered = float(PLOT_RAY_ORIGIN_HEIGHT)  # FIX: was * TOPO_SCALE_RANGE
    num_bins_in_full_waveform = int(np.rint(NUM_WAVEFORM_BINS * waveform_distance_covered))

    full_waveform = np.zeros(num_bins_in_full_waveform)
    full_waveform[-NUM_WAVEFORM_BINS:] = waveform

    # Offset of the 128-bin waveform within the full array (used for bin-index mapping)
    wf_offset = num_bins_in_full_waveform - NUM_WAVEFORM_BINS

    # Number of bins spanning the [0, 1] topography range (excluding spillover)
    num_waveform_bins_within_topo = int(np.rint(NUM_WAVEFORM_BINS * TOPO_SCALE_RANGE))

    # ------------------------------------------------------------------
    # Compute range window bin positions in the waveform
    # ------------------------------------------------------------------

    right_spillover_bins = int(np.rint(TOPO_SCALE_LOW * NUM_WAVEFORM_BINS))
    topo_bins_end = int(np.rint(num_bins_in_full_waveform - right_spillover_bins))
    topo_bins_start = topo_bins_end - num_waveform_bins_within_topo

    # Offset the window start by the edge-vs-nadir amount so that a topography
    # value of 1 aligns with the swath edge rather than nadir.
    edge_vs_nadir_bins = (
        edge_vs_nadir_offset_z / waveform_distance_covered
    ) * num_bins_in_full_waveform
    range_window_start_bin = (
        topo_bins_start + num_waveform_bins_within_topo * range_window_bottom + edge_vs_nadir_bins
    )
    # range_window_end_bin = topo_bins_start + num_waveform_bins_within_topo * range_window_top
    range_window_end_bin = (topo_bins_start + num_waveform_bins_within_topo * range_window_top + edge_vs_nadir_bins)

    # ------------------------------------------------------------------
    # Append blank tail and resample waveform to frame count
    # ------------------------------------------------------------------

    # Short blank tail for a cleaner visual ending
    tail_bins = int(np.rint(NUM_WAVEFORM_BINS * 0.2))
    full_waveform = np.concatenate((full_waveform, np.zeros(tail_bins)))
    num_bins_pre_resample = len(full_waveform)

    # Resample so bin count is an exact multiple of frame count (smooth animation)
    bins_per_frame = int(np.ceil(num_bins_pre_resample / NUM_FRAMES))
    num_wf_bins_resampled = bins_per_frame * NUM_FRAMES
    full_waveform = np.interp(
        np.linspace(0, 1, num_wf_bins_resampled),
        np.linspace(0, 1, num_bins_pre_resample),
        full_waveform,
    )

    # Rescale range window bin positions to match resampled waveform length
    _window_width = range_window_end_bin - range_window_start_bin
    range_window_start_bin = (
        range_window_start_bin / num_bins_pre_resample
    ) * num_wf_bins_resampled
    range_window_end_bin = range_window_start_bin + (
        _window_width / num_bins_pre_resample
    ) * num_wf_bins_resampled

    # ------------------------------------------------------------------
    # Map simulated waveform bin indices to resampled coordinates
    # ------------------------------------------------------------------

    def _to_wf_bin(sim_bin):
        """Map a bin index in the 128-bin simulated waveform to the resampled coordinate."""
        return (wf_offset + float(sim_bin)) / num_bins_pre_resample * num_wf_bins_resampled

    # Leading edge marker positions
    le_start_wf_bin = _to_wf_bin(le_index_start) if le_detected else None
    le_end_wf_bin = _to_wf_bin(le_index_end) if le_detected else None

    # POCA bin: coincides with leading edge start, or falls back to max-topography approximation
    if show_poca:
        if le_detected:
            poca_sim_bin = le_index_start + (le_index_end - le_index_start) * 0.75
        else:
            poca_sim_bin = (1.0 - float(np.max(topography_scaled))) * NUM_WAVEFORM_BINS
        poca_wf_bin = _to_wf_bin(poca_sim_bin)

    # Nadir bin: derived from the surface height at the centre of the topography profile
    if show_nadir:
        centre_idx = len(topography_scaled) // 2
        nadir_sim_bin = (1.0 - float(topography_scaled[centre_idx])) * NUM_WAVEFORM_BINS
        nadir_wf_bin = _to_wf_bin(nadir_sim_bin)

    # ------------------------------------------------------------------
    # Add noise to waveform
    # ------------------------------------------------------------------

    if wf_noise_amplitude > 0:
        full_waveform += np.random.uniform(
            low=0,
            high=wf_noise_amplitude * np.max(waveform),
            size=len(full_waveform),
        )

    # ------------------------------------------------------------------
    # Ray-topography intersections
    # ------------------------------------------------------------------

    plot_ray_origin = np.array([0.0, float(PLOT_RAY_ORIGIN_HEIGHT)])
    topography_x = np.linspace(-1, 1, len(topography))
    topography_spline = interp1d(topography_x, topography, fill_value="extrapolate", kind="slinear")

    plot_bisections = np.full((num_rays_to_display, 2), np.nan)
    for ray in range(num_rays_to_display):
        ray_func = interp1d(
            [plot_ray_origin[0], range_window_bottom_x[ray]],
            [plot_ray_origin[1], range_window_bottom_z[ray]],
            fill_value="extrapolate",
        )
        try:
            root = root_scalar(
                lambda x: topography_spline(x) - ray_func(x),
                x0=range_window_bottom_x[ray],
                bracket=[-PLOT_ACROSS_TRACK_WIDTH / 2, PLOT_ACROSS_TRACK_WIDTH / 2],
            ).root
            plot_bisections[ray] = [root, topography_spline(root)]
        except Exception:
            continue

    # ------------------------------------------------------------------
    # Pulse travel frames
    # ------------------------------------------------------------------

    plot_ray_vecs = plot_ray_origin - plot_bisections
    plot_ray_lengths = np.linalg.norm(plot_ray_vecs, axis=-1)
    ray_unit_vecs = plot_ray_vecs / plot_ray_lengths[:, np.newaxis]
    pulse_travel_distances = plot_ray_lengths * 2  # Round-trip distance

    # Pulse position is derived from waveform bin position so both subplots share one clock.
    # Each pre-resample bin = 1/NUM_WAVEFORM_BINS one-way distance units, so the round-trip
    # distance at bin b is (b / NUM_WAVEFORM_BINS) * 2.  After resampling, scale accordingly.
    current_bins = np.arange(NUM_FRAMES) * bins_per_frame
    dist_travelled = (
        current_bins / num_wf_bins_resampled
    ) * (num_bins_pre_resample / NUM_WAVEFORM_BINS) * 2  # FIX: was * PLOT_RAY_ORIGIN_HEIGHT * (1 / TOPO_SCALE_RANGE) * 2

    pulse_travel_frames = np.full((num_rays_to_display, NUM_FRAMES, 2), np.nan)
    for ray in range(num_rays_to_display):
        if np.isnan(plot_bisections[ray]).any():
            continue
        outgoing = dist_travelled <= plot_ray_lengths[ray]
        returning = (dist_travelled > plot_ray_lengths[ray]) & (
            dist_travelled <= pulse_travel_distances[ray]
        )
        pulse_travel_frames[ray, outgoing] = (
            plot_ray_origin - dist_travelled[outgoing, np.newaxis] * ray_unit_vecs[ray]
        )
        dist_back = dist_travelled[returning] - plot_ray_lengths[ray]
        pulse_travel_frames[ray, returning] = (
            plot_bisections[ray] + dist_back[:, np.newaxis] * ray_unit_vecs[ray]
        )

    # ------------------------------------------------------------------
    # POCA and nadir ray indices
    # ------------------------------------------------------------------

    valid_mask = ~np.isnan(plot_bisections).any(axis=1)

    if show_poca:
        distances = np.linalg.norm(
            plot_bisections[valid_mask] - plot_ray_origin, axis=1
        )
        poca_index = np.where(valid_mask)[0][np.argmin(distances)]
    else:
        poca_index = None

    if show_nadir:
        nadir_index = next(
            (idx for idx in np.argsort(np.abs(range_window_bottom_x)) if valid_mask[idx]),
            None,
        )
    else:
        nadir_index = None

    # ------------------------------------------------------------------
    # Ray colours (opacity scaled by antenna gain contribution)
    # ------------------------------------------------------------------

    ray_colour_power_factor = 2.5
    ray_alphas = contribution_angle_scale ** ray_colour_power_factor
    ray_alphas = (
        0.1 + (ray_alphas - ray_alphas.min()) * 0.9 / (ray_alphas.max() - ray_alphas.min() + 1e-12)
    )
    ray_colours_pulse = [
        f"rgba(255, 0, 0, {ray_alphas[ray]:.3f})" for ray in range(num_rays_to_display)
    ]

    # ------------------------------------------------------------------
    # Determine at which animation frame each waveform marker first appears
    # ------------------------------------------------------------------

    def _wf_bin_to_frame(wf_bin):
        """Return the first frame at which a given resampled waveform bin is visible."""
        return int(np.ceil(wf_bin / bins_per_frame)) if wf_bin is not None else None

    poca_reveal_frame = _wf_bin_to_frame(poca_wf_bin) if show_poca else None
    nadir_reveal_frame = _wf_bin_to_frame(nadir_wf_bin) if show_nadir else None
    le_start_reveal_frame = _wf_bin_to_frame(le_start_wf_bin) if (show_leading_edge and le_detected) else None
    le_end_reveal_frame = _wf_bin_to_frame(le_end_wf_bin) if (show_leading_edge and le_detected) else None

    # ------------------------------------------------------------------
    # Build static traces
    # ------------------------------------------------------------------

    fig = make_subplots(rows=1, cols=2, column_widths=[0.65, 0.35], horizontal_spacing=0.01)
    num_static_traces = 0

    # Topography surface and fill
    fig.add_trace(
        go.Scatter(
            x=topography_x, y=topography_spline(topography_x),
            mode="lines", line_shape="spline",
            showlegend=False, marker=dict(color="gray"),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=topography_x, y=[-edge_vs_nadir_offset_z] * len(topography_x),
            mode="lines", line=dict(width=0),
            fill="tonexty", showlegend=False, marker=dict(color="gray"),
        ),
        row=1, col=1,
    )
    num_static_traces += 2

    # Ray paths
    if show_rays:
        for ray in range(num_rays_to_display):
            fig.add_trace(
                go.Scatter(
                    x=[0, plot_bisections[ray, 0]],
                    y=[PLOT_RAY_ORIGIN_HEIGHT, plot_bisections[ray, 1]],
                    mode="lines", showlegend=False, name=f"Ray {ray}",
                    line=dict(color=f"rgba(255, 0, 0, {ray_alphas[ray]:.3f})", dash="dash"),
                ),
                row=1, col=1,
            )
        num_static_traces += num_rays_to_display

    # POCA spatial marker
    if show_poca:
        fig.add_trace(
            go.Scatter(
                x=[plot_bisections[poca_index, 0]], y=[plot_bisections[poca_index, 1]],
                mode="markers", showlegend=False, name="POCA",
                marker=dict(
                    color=colour_poca, size=12, symbol="star",
                    line=dict(width=0.5, color="rgb(0,0,1)"),
                ),
            ),
            row=1, col=1,
        )
        num_static_traces += 1

    # Nadir spatial marker
    if show_nadir and nadir_index is not None:
        fig.add_trace(
            go.Scatter(
                x=[plot_bisections[nadir_index, 0]], y=[plot_bisections[nadir_index, 1]],
                mode="markers", showlegend=False, name="Nadir",
                marker=dict(
                    color=colour_nadir, size=10, symbol="diamond",
                    line=dict(width=0.5, color="rgb(0,0,1)"),
                ),
            ),
            row=1, col=1,
        )
        num_static_traces += 1

    # Range window spatial overlay
    if show_range_window:
        fig.add_trace(
            go.Scatter(
                x=range_window_bottom_x, y=range_window_bottom_z,
                mode="lines", line_shape="spline", showlegend=False,
                name="Range Window Start (Plot 1)",
                fillcolor="rgba(100, 255, 100, 0.15)",
                marker=dict(color="rgba(0, 255, 0, 1)"), line=dict(dash="dash"),
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=range_window_bottom_x,
                y=range_window_bottom_z + range_window_scale - edge_vs_nadir_offset_z,
                mode="lines", line_shape="spline", showlegend=False,
                name="Range Window End (Plot 1)", fill="tonexty",
                fillcolor="rgba(100, 255, 100, 0.15)",
                marker=dict(color="rgba(0, 255, 0, 1)"), line=dict(dash="dash"),
            ),
            row=1, col=1,
        )
        num_static_traces += 2

    # ------------------------------------------------------------------
    # Animated traces (initialised to empty / off-screen)
    # ------------------------------------------------------------------
    # Ordering within the animated block determines z-order on the waveform subplot.
    # Traces added later render on top. Order:
    #   1. Range window lines (if shown) - background reference
    #   2. Waveform fill
    #   3. Pulse markers (spatial subplot)
    #   4. LE start line (if shown)
    #   5. LE end line (if shown)
    #   6. POCA waveform marker (if shown)
    #   7. Nadir waveform marker (if shown)

    _HIDDEN_X = [-1e9, -1e9]  # Off-screen position used to hide a marker before reveal

    animated_trace_indices = []

    def _add_animated(trace, row, col):
        fig.add_trace(trace, row=row, col=col)
        animated_trace_indices.append(len(fig.data) - 1)

    # Range window lines on the waveform subplot (redrawn each frame to avoid a
    # Plotly bug where filled traces disappear during animation)
    if show_range_window:
        _add_animated(
            go.Scatter(
                x=[range_window_start_bin, range_window_start_bin], y=[0, 1],
                mode="lines", showlegend=False, name="Range Window Start (Plot 2)",
                marker=dict(color="rgba(0, 255, 0, 1)"), line=dict(dash="dash"),
            ),
            row=1, col=2,
        )
        _add_animated(
            go.Scatter(
                x=[range_window_end_bin, range_window_end_bin], y=[0, 1],
                mode="lines", showlegend=False, name="Range Window End (Plot 2)",
                fill="tonextx", fillcolor="rgba(100, 255, 100, 0.15)",
                marker=dict(color="rgba(0, 255, 0, 1)"), line=dict(dash="dash"),
            ),
            row=1, col=2,
        )

    # Waveform fill
    _add_animated(
        go.Scatter(
            y=[waveform[0]], name="Waveform",
            fill="tozeroy", marker=dict(color="skyblue"), line=dict(width=1), showlegend=False,
        ),
        row=1, col=2,
    )

    # Pulse position markers (spatial subplot)
    _add_animated(
        go.Scatter(
            x=[0] * num_rays_to_display, y=[PLOT_RAY_ORIGIN_HEIGHT] * num_rays_to_display,
            mode="markers", showlegend=False, name="Pulses",
            marker=dict(color=ray_colours_pulse),
        ),
        row=1, col=1,
    )

    # Leading edge lines (waveform subplot) - rendered above the waveform fill
    if show_leading_edge and le_detected:
        _add_animated(
            go.Scatter(
                x=_HIDDEN_X, y=[0, 1],
                mode="lines", showlegend=False, name="LE Start",
                line=dict(color=colour_le, dash="dot", width=1.5), opacity=0.4,
            ),
            row=1, col=2,
        )
        _add_animated(
            go.Scatter(
                x=_HIDDEN_X, y=[0, 1],
                mode="lines", showlegend=False, name="LE End",
                line=dict(color=colour_le, dash="dot", width=1.5), opacity=0.4,
            ),
            row=1, col=2,
        )

    # POCA waveform marker - rendered above waveform fill and LE lines
    if show_poca:
        _add_animated(
            go.Scatter(
                x=_HIDDEN_X[:1], y=[waveform[int(np.rint(poca_sim_bin))]],
                mode="markers", showlegend=False, name="POCA",
                marker=dict(
                    color=colour_poca, size=12, symbol="star",
                    line=dict(width=0.5, color="rgb(0,0,1)"),
                ),
            ),
            row=1, col=2,
        )

    # Nadir waveform marker - rendered above everything else
    if show_nadir:
        _add_animated(
            go.Scatter(
                x=_HIDDEN_X[:1], y=[waveform[int(np.rint(nadir_sim_bin))]],
                mode="markers", showlegend=False, name="Nadir",
                marker=dict(
                    color=colour_nadir, size=12, symbol="diamond",
                    line=dict(width=0.5, color="rgb(0,0,1)"),
                ),
            ),
            row=1, col=2,
        )

    # ------------------------------------------------------------------
    # Build animation frames
    # ------------------------------------------------------------------

    # Identify the index of each animated trace within animated_trace_indices for
    # convenient per-frame updates.
    _ai = {name: i for i, name in enumerate(
        (["rw_start", "rw_end"] if show_range_window else [])
        + ["waveform", "pulses"]
        + (["le_start", "le_end"] if (show_leading_edge and le_detected) else [])
        + (["poca"] if show_poca else [])
        + (["nadir"] if show_nadir else [])
    )}

    frames = []
    for frame in range(NUM_FRAMES):
        current_bin = frame * bins_per_frame
        frame_traces = []

        if show_range_window:
            frame_traces.append(
                go.Scatter(x=[range_window_start_bin, range_window_start_bin], y=[0, 1])
            )
            frame_traces.append(
                go.Scatter(x=[range_window_end_bin, range_window_end_bin], y=[0, 1])
            )

        frame_traces.append(go.Scatter(y=full_waveform[:current_bin]))
        frame_traces.append(
            go.Scatter(
                x=pulse_travel_frames[:, frame, 0],
                y=pulse_travel_frames[:, frame, 1],
            )
        )

        if show_leading_edge and le_detected:
            # LE start: appear at its reveal frame, stay visible thereafter
            le_start_x = (
                [le_start_wf_bin, le_start_wf_bin]
                if frame >= le_start_reveal_frame
                else _HIDDEN_X
            )
            frame_traces.append(go.Scatter(x=le_start_x, y=[0, 1]))

            le_end_x = (
                [le_end_wf_bin, le_end_wf_bin]
                if frame >= le_end_reveal_frame
                else _HIDDEN_X
            )
            frame_traces.append(go.Scatter(x=le_end_x, y=[0, 1]))

        if show_poca:
            poca_x = [poca_wf_bin] if frame >= poca_reveal_frame else _HIDDEN_X[:1]
            frame_traces.append(go.Scatter(x=poca_x, y=[waveform[int(np.rint(poca_sim_bin))]]))

        if show_nadir:
            nadir_x = [nadir_wf_bin] if frame >= nadir_reveal_frame else _HIDDEN_X[:1]
            frame_traces.append(go.Scatter(x=nadir_x, y=[waveform[int(np.rint(nadir_sim_bin))]]))

        frames.append(
            dict(
                name=str(frame),
                data=frame_traces,
                traces=animated_trace_indices,
            )
        )

    fig.frames = frames

    # ------------------------------------------------------------------
    # Layout and controls
    # ------------------------------------------------------------------

    x_range = [-(PLOT_ACROSS_TRACK_WIDTH * 1.1) / 2, (PLOT_ACROSS_TRACK_WIDTH * 1.1) / 2]
    y_range = [-edge_vs_nadir_offset_z - 0.1, PLOT_RAY_ORIGIN_HEIGHT]

    # Satellite image at ray origin
    try:
        with open(satellite_image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()

        col1_paper_width = 0.65 - 0.01
        x_paper = (plot_ray_origin[0] - x_range[0]) / (x_range[1] - x_range[0]) * col1_paper_width
        y_paper = (plot_ray_origin[1] - y_range[0]) / (y_range[1] - y_range[0])

        fig.add_layout_image(dict(
            source=f"data:image/png;base64,{encoded}",
            xref="paper", yref="paper",
            x=x_paper, y=y_paper,
            sizex=0.125, sizey=0.125,
            xanchor="center", yanchor="middle",
            layer="above",
        ))
    except Exception:
        pass

    fig.update_layout(
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [
                            None,
                            {
                                "frame": {"duration": 1000 / FPS, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                        "label": "▶ Play",
                        "method": "animate",
                    },
                    {
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                        "label": "⏸ Pause",
                        "method": "animate",
                    },
                ],
                "bgcolor": "#4a7fb5",
                "font": {"color": "white", "size": 13},
                "bordercolor": "#4a7fb5",
                "borderwidth": 0,
                "direction": "left",
                "pad": {"r": 0, "t": 50},
                "showactive": False,
                "active": 0,
                "type": "buttons",
                "x": 0.5,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top",
            }
        ],
        dragmode=False,
        xaxis=dict(showgrid=False, visible=False, range=x_range, fixedrange=True),
        yaxis=dict(showgrid=False, visible=False, range=y_range, fixedrange=True),
        xaxis2=dict(
            showgrid=False, visible=False,
            range=[-5, num_wf_bins_resampled + 5], fixedrange=True,
        ),
        yaxis2=dict(showgrid=False, visible=False, range=[-0.25, 1.25], fixedrange=True),
        height=PLOT_HEIGHT,
        plot_bgcolor="rgba(0, 0, 0, 0)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
    )

    if output_path is not None:
        fig.write_html(f"{output_path}.html", auto_open=False, auto_play=False)

    return fig


# ----------------------------------------------------------------------
# 3D Plotting (LRM-analogous)
# ----------------------------------------------------------------------

def animate_altimetry_waveform_3d(
    topography,
    output_path=None,
    num_rays_to_display=25,
    range_window_top=1.0,
    range_window_bottom=0.0,
    show_poca=True,
    show_range_window=True,
    show_rays=True,
    wf_noise_amplitude=0.01,
    show_nadir=True,
    show_leading_edge=True,
):
    """Generates an animated 3D altimetry waveform simulation using Plotly.

    Simulates the travel of radar pulses from a satellite towards a surface defined
    by the input topography grid, animating both the spatial pulse propagation in 3D
    and the corresponding waveform build-up.

    Args:
        topography (array-like): 2D array of surface heights, normalised to [0, 1].
        output_path (str, optional): If provided, saves the animation as an HTML file
            at this path (without extension). Defaults to None.
        num_rays_to_display (int): Target number of rays to show in the spatial plot.
            The actual count may differ slightly due to hexagonal packing geometry.
            Defaults to 25.
        range_window_top (float): Upper bound of the range window, in [0, 1].
            Defaults to 1.0.
        range_window_bottom (float): Lower bound of the range window, in [0, 1].
            Defaults to 0.0.
        show_poca (bool): Whether to mark the Point of Closest Approach.
            Defaults to True.
        show_range_window (bool): Whether to display the range window on both subplots.
            Defaults to True.
        show_rays (bool): Whether to display the individual ray paths. Defaults to True.
        wf_noise_amplitude (float): Amplitude of uniform noise added to the waveform,
            as a fraction of the peak waveform value. Defaults to 0.01.
        show_nadir (bool): Whether to mark the nadir point (directly below the
            satellite) on both subplots. Defaults to True.
        show_leading_edge (bool): Whether to mark the leading-edge start and end on
            the waveform subplot. Defaults to True.

    Returns:
        plotly.graph_objects.Figure: The animated Plotly figure.
    """

    # ------------------------------------------------------------------
    # Constants
    # ------------------------------------------------------------------

    PLOT_HEIGHT = 800
    FPS = 24
    ANIMATION_LENGTH_S = 3
    NUM_FRAMES = FPS * ANIMATION_LENGTH_S
    NUM_WAVEFORM_BINS = 128
    PLOT_RAY_ORIGIN_HEIGHT = 3
    PLOT_FOOTPRINT_RADIUS = 1
    TOPO_SCALE_LOW = 0.2
    TOPO_SCALE_HIGH = 0.8
    TOPO_SCALE_RANGE = TOPO_SCALE_HIGH - TOPO_SCALE_LOW  # 0.6

    # ------------------------------------------------------------------
    # Validate and prepare inputs
    # ------------------------------------------------------------------

    topography = np.asarray(np.clip(topography, 0, 1), dtype=float)
    range_window_top = np.clip(range_window_top, 0.0, 1.0)
    range_window_bottom = np.clip(range_window_bottom, 0.0, range_window_top)
    range_window_scale = range_window_top - range_window_bottom
    range_window_centre = np.mean([range_window_top, range_window_bottom])

    topography_scaled = TOPO_SCALE_LOW + TOPO_SCALE_RANGE * topography

    # ------------------------------------------------------------------
    # Simulate waveform
    # ------------------------------------------------------------------

    waveform, contribution_angle_scale = simulate_altimetry_waveform_3d(
        topography_scaled, return_contribution_angle_scale=True
    )

    # ------------------------------------------------------------------
    # Detect leading edge (on the clean waveform, before noise is added)
    # ------------------------------------------------------------------

    le_detected = False
    le_index_start = le_index_end = np.nan

    if show_leading_edge:
        try:
            le_index_start, le_index_end, _, _ = get_leading_edge(
                waveform=waveform,
                tracker_range=0.0,
                reference_bin_index=NUM_WAVEFORM_BINS // 2,
                smoothing_window_width=5,
                range_bin_size=1.0,
                wf_oversampling_factor=10,
            )
            le_detected = not (np.isnan(le_index_start) or np.isnan(le_index_end))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Range window geometry in the spatial plot
    # ------------------------------------------------------------------

    plot_range_window_bottom, _ = get_range_window_bottom_3d(
        num_rays_to_display, PLOT_FOOTPRINT_RADIUS, PLOT_RAY_ORIGIN_HEIGHT
    )
    num_rays_to_display = len(plot_range_window_bottom)

    # Shift z so the nadir point of the range window bottom lands at the correct height
    range_window_bottom_z_at_nadir = range_window_centre - range_window_scale / 2
    plot_range_window_bottom[:, 2] += range_window_bottom_z_at_nadir

    edge_vs_nadir_offset_z = (
        np.nanmax(plot_range_window_bottom[:, 2]) - np.nanmin(plot_range_window_bottom[:, 2])
    )

    contribution_angle_scale = np.interp(
        np.linspace(0, 1, num_rays_to_display),
        np.linspace(0, 1, len(contribution_angle_scale)),
        contribution_angle_scale,
    )

    # ------------------------------------------------------------------
    # Build and pad the full waveform array
    # ------------------------------------------------------------------

    # The 128-bin waveform maps normalised height h to bin via (1 - h) * 128, so
    # it spans the full PLOT_RAY_ORIGIN_HEIGHT of one-way travel time.  The leading
    # zeros represent the in-flight pulse before any surface return is received.
    waveform_distance_covered = float(PLOT_RAY_ORIGIN_HEIGHT)  # FIX: was * TOPO_SCALE_RANGE
    num_bins_in_full_waveform = int(np.rint(NUM_WAVEFORM_BINS * waveform_distance_covered))

    full_waveform = np.zeros(num_bins_in_full_waveform)
    full_waveform[-NUM_WAVEFORM_BINS:] = waveform

    wf_offset = num_bins_in_full_waveform - NUM_WAVEFORM_BINS

    num_waveform_bins_within_topo = int(np.rint(NUM_WAVEFORM_BINS * TOPO_SCALE_RANGE))

    # ------------------------------------------------------------------
    # Compute range window bin positions in the waveform
    # ------------------------------------------------------------------

    right_spillover_bins = int(np.rint(TOPO_SCALE_LOW * NUM_WAVEFORM_BINS))
    topo_bins_end = int(np.rint(num_bins_in_full_waveform - right_spillover_bins))
    topo_bins_start = topo_bins_end - num_waveform_bins_within_topo

    edge_vs_nadir_bins = (
        edge_vs_nadir_offset_z / waveform_distance_covered
    ) * num_bins_in_full_waveform
    range_window_start_bin = (
        topo_bins_start + num_waveform_bins_within_topo * range_window_bottom + edge_vs_nadir_bins
    )
    #range_window_end_bin = topo_bins_start + num_waveform_bins_within_topo * range_window_top
    range_window_end_bin = (topo_bins_start + num_waveform_bins_within_topo * range_window_top + edge_vs_nadir_bins)
    
    # ------------------------------------------------------------------
    # Append blank tail and resample waveform to frame count
    # ------------------------------------------------------------------

    tail_bins = int(np.rint(NUM_WAVEFORM_BINS * 0.2))
    full_waveform = np.concatenate((full_waveform, np.zeros(tail_bins)))
    num_bins_pre_resample = len(full_waveform)

    bins_per_frame = int(np.ceil(num_bins_pre_resample / NUM_FRAMES))
    num_wf_bins_resampled = bins_per_frame * NUM_FRAMES
    full_waveform = np.interp(
        np.linspace(0, 1, num_wf_bins_resampled),
        np.linspace(0, 1, num_bins_pre_resample),
        full_waveform,
    )

    _window_width = range_window_end_bin - range_window_start_bin
    range_window_start_bin = (
        range_window_start_bin / num_bins_pre_resample
    ) * num_wf_bins_resampled
    range_window_end_bin = range_window_start_bin + (
        _window_width / num_bins_pre_resample
    ) * num_wf_bins_resampled

    # ------------------------------------------------------------------
    # Map simulated waveform bin indices to resampled coordinates
    # ------------------------------------------------------------------

    def _to_wf_bin(sim_bin):
        """Map a bin index in the 128-bin simulated waveform to the resampled coordinate."""
        return (wf_offset + float(sim_bin)) / num_bins_pre_resample * num_wf_bins_resampled

    le_start_wf_bin = _to_wf_bin(le_index_start) if le_detected else None
    le_end_wf_bin = _to_wf_bin(le_index_end) if le_detected else None

    if show_poca:
        if le_detected:
            poca_sim_bin = le_index_start + (le_index_end - le_index_start) / 2
        else:
            poca_sim_bin = (1.0 - float(np.max(topography_scaled))) * NUM_WAVEFORM_BINS
        poca_wf_bin = _to_wf_bin(poca_sim_bin)

    if show_nadir:
        cy, cx = topography_scaled.shape[0] // 2, topography_scaled.shape[1] // 2
        nadir_sim_bin = (1.0 - float(topography_scaled[cy, cx])) * NUM_WAVEFORM_BINS
        nadir_wf_bin = _to_wf_bin(nadir_sim_bin)

    # ------------------------------------------------------------------
    # Add noise to waveform
    # ------------------------------------------------------------------

    if wf_noise_amplitude > 0:
        full_waveform += np.random.uniform(
            low=0,
            high=wf_noise_amplitude * np.max(waveform),
            size=len(full_waveform),
        )

    # ------------------------------------------------------------------
    # Topography interpolator and ray-surface intersections
    # ------------------------------------------------------------------

    nx, ny = np.shape(topography)
    x = np.linspace(-PLOT_FOOTPRINT_RADIUS, PLOT_FOOTPRINT_RADIUS, nx)
    y = np.linspace(-PLOT_FOOTPRINT_RADIUS, PLOT_FOOTPRINT_RADIUS, ny)
    topography_interpolator = RegularGridInterpolator(
        (x, y), topography, bounds_error=False, method="slinear", fill_value=None,
    )

    plot_ray_origin = np.array([0.0, 0.0, float(PLOT_RAY_ORIGIN_HEIGHT)])
    plot_intersections = np.full((num_rays_to_display, 3), np.nan)
    for ray in range(num_rays_to_display):
        plot_intersections[ray] = compute_intersection_3d(
            ray, plot_ray_origin, plot_range_window_bottom, topography_interpolator,
            root_bracket=[0, 1.5],
        )

    # ------------------------------------------------------------------
    # Pulse travel frames
    # ------------------------------------------------------------------

    plot_ray_vecs = plot_ray_origin - plot_intersections
    plot_ray_lengths = np.linalg.norm(plot_ray_vecs, axis=-1)
    ray_unit_vecs = plot_ray_vecs / plot_ray_lengths[:, np.newaxis]
    pulse_travel_distances = plot_ray_lengths * 2

    # Pulse position is derived from waveform bin position so both subplots share one clock.
    # Each pre-resample bin = 1/NUM_WAVEFORM_BINS one-way distance units, so the round-trip
    # distance at bin b is (b / NUM_WAVEFORM_BINS) * 2.  After resampling, scale accordingly.
    current_bins = np.arange(NUM_FRAMES) * bins_per_frame
    dist_travelled = (
        current_bins / num_wf_bins_resampled
    ) * (num_bins_pre_resample / NUM_WAVEFORM_BINS) * 2  # FIX: was * PLOT_RAY_ORIGIN_HEIGHT * (1 / TOPO_SCALE_RANGE) * 2

    pulse_travel_frames = np.full((num_rays_to_display, NUM_FRAMES, 3), np.nan)
    for ray in range(num_rays_to_display):
        if np.isnan(plot_intersections[ray]).any():
            continue
        outgoing = dist_travelled <= plot_ray_lengths[ray]
        returning = (dist_travelled > plot_ray_lengths[ray]) & (
            dist_travelled <= pulse_travel_distances[ray]
        )
        pulse_travel_frames[ray, outgoing] = (
            plot_ray_origin - dist_travelled[outgoing, np.newaxis] * ray_unit_vecs[ray]
        )
        dist_back = dist_travelled[returning] - plot_ray_lengths[ray]
        pulse_travel_frames[ray, returning] = (
            plot_intersections[ray] + dist_back[:, np.newaxis] * ray_unit_vecs[ray]
        )

    # ------------------------------------------------------------------
    # POCA and nadir ray indices
    # ------------------------------------------------------------------

    valid_mask = ~np.isnan(plot_intersections).any(axis=1)

    if show_poca:
        distances = np.linalg.norm(plot_intersections[valid_mask] - plot_ray_origin, axis=1)
        poca_index = np.where(valid_mask)[0][np.argmin(distances)]
    else:
        poca_index = None

    if show_nadir:
        ray_xy_distances = np.sqrt(
            plot_range_window_bottom[:, 0] ** 2 + plot_range_window_bottom[:, 1] ** 2
        )
        nadir_index = next(
            (idx for idx in np.argsort(ray_xy_distances) if valid_mask[idx]), None
        )
    else:
        nadir_index = None

    # ------------------------------------------------------------------
    # Ray colours (opacity scaled by antenna gain contribution)
    # ------------------------------------------------------------------

    ray_colour_power_factor = 2.5
    ray_alphas = contribution_angle_scale ** ray_colour_power_factor
    ray_alphas = (
        0.1 + (ray_alphas - ray_alphas.min()) * 0.9 / (ray_alphas.max() - ray_alphas.min() + 1e-12)
    )
    ray_colours_pulse = [
        f"rgba(255, 0, 0, {ray_alphas[ray]:.3f})" for ray in range(num_rays_to_display)
    ]

    # ------------------------------------------------------------------
    # Determine at which animation frame each waveform marker first appears
    # ------------------------------------------------------------------

    def _wf_bin_to_frame(wf_bin):
        """Return the first frame at which a given resampled waveform bin is visible."""
        return int(np.ceil(wf_bin / bins_per_frame)) if wf_bin is not None else None

    poca_reveal_frame = _wf_bin_to_frame(poca_wf_bin) if show_poca else None
    nadir_reveal_frame = _wf_bin_to_frame(nadir_wf_bin) if show_nadir else None
    le_start_reveal_frame = _wf_bin_to_frame(le_start_wf_bin) if (show_leading_edge and le_detected) else None
    le_end_reveal_frame = _wf_bin_to_frame(le_end_wf_bin) if (show_leading_edge and le_detected) else None

    # ------------------------------------------------------------------
    # Build static traces
    # ------------------------------------------------------------------

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.65, 0.35],
        horizontal_spacing=0.01,
        specs=[[{"type": "scatter3d"}, {"type": "scatter"}]],
    )

    # Topography surface rendered as a mesh at the hexagonal ray grid points
    lighting_effects = dict(ambient=0.4, diffuse=0.5, roughness=0.9, specular=0.6, fresnel=0.2)
    topo_z = topography_interpolator(
        (plot_range_window_bottom[:, 0], plot_range_window_bottom[:, 1])
    )
    fig.add_trace(
        go.Mesh3d(
            x=plot_range_window_bottom[:, 0],
            y=plot_range_window_bottom[:, 1],
            z=topo_z,
            name="Topography", showlegend=False,
            color="gray", flatshading=True, lighting=lighting_effects,
        ),
        row=1, col=1,
    )

    # Ray paths (grouped by ring so traces sharing an off-nadir angle share opacity)
    if show_rays:
        dash_length = 0.04
        gap_length = 0.025

        ray_radii = np.sqrt(
            plot_range_window_bottom[:, 0] ** 2 + plot_range_window_bottom[:, 1] ** 2
        )
        d_hex = np.sqrt(np.pi * PLOT_FOOTPRINT_RADIUS ** 2 / num_rays_to_display)
        ring_ids = np.round(ray_radii / d_hex).astype(int)
        unique_rings = np.unique(ring_ids)

        for ring_id in unique_rings:
            ring_ray_indices = np.where(ring_ids == ring_id)[0]
            alpha = np.nanmean(ray_alphas[ring_ray_indices])

            rays_x, rays_y, rays_z = [], [], []
            for ray in ring_ray_indices:
                if np.isnan(plot_intersections[ray]).any():
                    continue
                end = plot_intersections[ray]
                ray_vec = end - plot_ray_origin
                ray_len = np.linalg.norm(ray_vec)
                ray_unit = ray_vec / ray_len

                t, drawing = 0.0, True
                while t < ray_len:
                    seg_len = dash_length if drawing else gap_length
                    t_end = min(t + seg_len, ray_len)
                    if drawing:
                        p0 = plot_ray_origin + t * ray_unit
                        p1 = plot_ray_origin + t_end * ray_unit
                        rays_x += [p0[0], p1[0], None]
                        rays_y += [p0[1], p1[1], None]
                        rays_z += [p0[2], p1[2], None]
                    t = t_end
                    drawing = not drawing

            fig.add_trace(
                go.Scatter3d(
                    x=rays_x, y=rays_y, z=rays_z,
                    mode="lines", showlegend=False, name=f"Ring {ring_id}",
                    line=dict(color=f"rgba(255, 0, 0, {alpha:.3f})"),
                ),
                row=1, col=1,
            )

    # POCA spatial marker
    if show_poca:
        fig.add_trace(
            go.Scatter3d(
                x=[plot_intersections[poca_index, 0]],
                y=[plot_intersections[poca_index, 1]],
                z=[plot_intersections[poca_index, 2] + 0.01],
                mode="markers", showlegend=False, name="POCA",
                marker=dict(color=colour_poca, size=5, symbol="diamond"),
            ),
            row=1, col=1,
        )

    # Nadir spatial marker
    if show_nadir and nadir_index is not None:
        fig.add_trace(
            go.Scatter3d(
                x=[plot_intersections[nadir_index, 0]],
                y=[plot_intersections[nadir_index, 1]],
                z=[plot_intersections[nadir_index, 2] + 0.01],
                mode="markers", showlegend=False, name="Nadir",
                marker=dict(color=colour_nadir, size=5, symbol="circle"),
            ),
            row=1, col=1,
        )

    # Range window spherical shell
    if show_range_window:
        z_at_range_window_bottom = range_window_bottom_z_at_nadir

        top_correction = edge_vs_nadir_offset_z
        for _ in range(20):
            _nadir_z_top = range_window_bottom_z_at_nadir + range_window_scale - top_correction
            _r_top = PLOT_FOOTPRINT_RADIUS * (
                (_nadir_z_top - PLOT_RAY_ORIGIN_HEIGHT)
                / (range_window_bottom_z_at_nadir - PLOT_RAY_ORIGIN_HEIGHT)
            )
            top_correction = PLOT_RAY_ORIGIN_HEIGHT - np.sqrt(
                max(PLOT_RAY_ORIGIN_HEIGHT ** 2 - _r_top ** 2, 0)
            )
        z_at_range_window_top = range_window_bottom_z_at_nadir + range_window_scale - top_correction

        circumference = 2 * np.pi * PLOT_FOOTPRINT_RADIUS
        d = np.sqrt(np.pi * PLOT_FOOTPRINT_RADIUS ** 2 / num_rays_to_display)
        num_points_outer_ring = int(np.round(circumference / d))
        num_latitude_rings = max(8, num_points_outer_ring // 4)
        theta = np.linspace(0, 2 * np.pi, num_points_outer_ring, endpoint=False)

        nadir_z_levels = np.linspace(z_at_range_window_bottom, z_at_range_window_top, num_latitude_rings)
        ring_radii = (
            PLOT_FOOTPRINT_RADIUS
            * (nadir_z_levels - PLOT_RAY_ORIGIN_HEIGHT)
            / (z_at_range_window_bottom - PLOT_RAY_ORIGIN_HEIGHT)
        )

        rings_x = np.outer(ring_radii, np.cos(theta))
        rings_y = np.outer(ring_radii, np.sin(theta))
        radial_distances = np.abs(ring_radii)[:, np.newaxis] * np.ones((1, num_points_outer_ring))
        curvature_offset = PLOT_RAY_ORIGIN_HEIGHT - np.sqrt(
            np.maximum(PLOT_RAY_ORIGIN_HEIGHT ** 2 - radial_distances ** 2, 0)
        )
        rings_z = nadir_z_levels[:, np.newaxis] + curvature_offset

        x_shell = rings_x.ravel()
        y_shell = rings_y.ravel()
        z_shell = rings_z.ravel()

        i_idx, j_idx, k_idx = [], [], []
        n = num_points_outer_ring
        for r in range(num_latitude_rings - 1):
            for p in range(n):
                p_next = (p + 1) % n
                v00 = r * n + p;        v01 = r * n + p_next
                v10 = (r + 1) * n + p;  v11 = (r + 1) * n + p_next
                i_idx.append(v00); j_idx.append(v01); k_idx.append(v10)
                i_idx.append(v01); j_idx.append(v11); k_idx.append(v10)

        for cap_nadir_z, cap_radius, outer_ring_start in [
            (z_at_range_window_bottom, ring_radii[0],                      0),
            (z_at_range_window_top,    ring_radii[num_latitude_rings - 1], (num_latitude_rings - 1) * n),
        ]:
            inner_rings = []
            for frac in (3/4, 1/2, 1/4):
                r_ring = cap_radius * frac
                curv = PLOT_RAY_ORIGIN_HEIGHT - np.sqrt(
                    max(PLOT_RAY_ORIGIN_HEIGHT ** 2 - r_ring ** 2, 0)
                )
                start = len(x_shell)
                x_shell = np.concatenate([x_shell, r_ring * np.cos(theta)])
                y_shell = np.concatenate([y_shell, r_ring * np.sin(theta)])
                z_shell = np.concatenate([z_shell, np.full(n, cap_nadir_z + curv)])
                inner_rings.append(start)

            cap_centre_idx = len(x_shell)
            x_shell = np.append(x_shell, 0.0)
            y_shell = np.append(y_shell, 0.0)
            z_shell = np.append(z_shell, cap_nadir_z)

            for ri in range(len(inner_rings := [outer_ring_start] + inner_rings) - 1):
                r_outer_start = inner_rings[ri]
                r_inner_start = inner_rings[ri + 1]
                for p in range(n):
                    p_next = (p + 1) % n
                    v_o0 = r_outer_start + p;  v_o1 = r_outer_start + p_next
                    v_i0 = r_inner_start + p;  v_i1 = r_inner_start + p_next
                    i_idx.append(v_o0); j_idx.append(v_o1); k_idx.append(v_i0)
                    i_idx.append(v_o1); j_idx.append(v_i1); k_idx.append(v_i0)

            innermost_start = inner_rings[-1]
            for p in range(n):
                i_idx.append(cap_centre_idx)
                j_idx.append(innermost_start + p)
                k_idx.append(innermost_start + (p + 1) % n)

        fig.add_trace(
            go.Mesh3d(
                x=x_shell, y=y_shell, z=z_shell,
                i=i_idx, j=j_idx, k=k_idx,
                color="green", opacity=0.05,
                name="Range Window (Plot 1)", showlegend=False,
            ),
            row=1, col=1,
        )

    # ------------------------------------------------------------------
    # Satellite mesh (static)
    # ------------------------------------------------------------------

    satellite_size = PLOT_FOOTPRINT_RADIUS * 0.15
    sat_x, sat_y, sat_z = 0.0, 0.0, float(PLOT_RAY_ORIGIN_HEIGHT)
    body_h  = satellite_size * 0.6
    body_w  = satellite_size * 0.35
    panel_w = satellite_size
    panel_h = satellite_size * 0.25

    bus_fill_colour      = "rgb(210, 215, 225)"
    bus_outline_colour   = "rgb(130, 135, 145)"
    panel_fill_colour    = "rgb(120, 145, 185)"
    panel_outline_colour = "rgb( 80, 100, 140)"

    def _rect_mesh(cx, cy, cz, hw, hh, axis):
        if axis == "x":
            x = [cx - hw, cx + hw, cx + hw, cx - hw]
            y = [cy,       cy,       cy,       cy     ]
            z = [cz - hh, cz - hh, cz + hh, cz + hh]
        else:
            x = [cx,       cx,       cx,       cx     ]
            y = [cy - hw, cy + hw, cy + hw, cy - hw]
            z = [cz - hh, cz - hh, cz + hh, cz + hh]
        return x, y, z, [0, 0], [1, 2], [2, 3]

    def _rect_outline(cx, cy, cz, hw, hh, axis):
        if axis == "x":
            x = [cx - hw, cx + hw, cx + hw, cx - hw, cx - hw, None]
            y = [cy,       cy,       cy,       cy,       cy,      None]
            z = [cz - hh, cz - hh, cz + hh, cz + hh, cz - hh, None]
        else:
            x = [cx,       cx,       cx,       cx,       cx,      None]
            y = [cy - hw, cy + hw, cy + hw, cy - hw, cy - hw, None]
            z = [cz - hh, cz - hh, cz + hh, cz + hh, cz - hh, None]
        return x, y, z

    for axis in ("x", "y"):
        left_cx   = sat_x - (body_w + panel_w) if axis == "x" else sat_x
        left_cy   = sat_y if axis == "x" else sat_y - (body_w + panel_w)
        right_cx  = sat_x + (body_w + panel_w) if axis == "x" else sat_x
        right_cy  = sat_y if axis == "x" else sat_y + (body_w + panel_w)

        shapes = [
            (sat_x,   sat_y,   sat_z, body_w,  body_h,  bus_fill_colour,   bus_outline_colour,   "Bus"),
            (left_cx, left_cy, sat_z, panel_w, panel_h, panel_fill_colour, panel_outline_colour, "Panel L"),
            (right_cx,right_cy,sat_z, panel_w, panel_h, panel_fill_colour, panel_outline_colour, "Panel R"),
        ]

        bus_out_x,   bus_out_y,   bus_out_z   = [], [], []
        panel_out_x, panel_out_y, panel_out_z = [], [], []

        for cx, cy, cz, hw, hh, fill_col, outline_col, label in shapes:
            fx, fy, fz, fi, fj, fk = _rect_mesh(cx, cy, cz, hw, hh, axis)
            fig.add_trace(
                go.Mesh3d(
                    x=fx, y=fy, z=fz, i=fi, j=fj, k=fk,
                    color=fill_col, opacity=1.0, flatshading=True,
                    lighting=dict(ambient=1.0, diffuse=0.0),
                    showlegend=False, name=f"Satellite {label} fill",
                ),
                row=1, col=1,
            )
            ox, oy, oz = _rect_outline(cx, cy, cz, hw, hh, axis)
            if label == "Bus":
                bus_out_x += ox; bus_out_y += oy; bus_out_z += oz
            else:
                panel_out_x += ox; panel_out_y += oy; panel_out_z += oz

        bus_out_x += [sat_x, sat_x, None]
        bus_out_y += [sat_y, sat_y, None]
        bus_out_z += [sat_z - body_h, sat_z - body_h - satellite_size * 0.4, None]

        for out_x, out_y, out_z, outline_col, name in [
            (bus_out_x,   bus_out_y,   bus_out_z,   bus_outline_colour,   "Satellite bus outline"),
            (panel_out_x, panel_out_y, panel_out_z, panel_outline_colour, "Satellite panel outline"),
        ]:
            fig.add_trace(
                go.Scatter3d(
                    x=out_x, y=out_y, z=out_z,
                    mode="lines", showlegend=False, name=name,
                    line=dict(color=outline_col, width=3),
                ),
                row=1, col=1,
            )

    # ------------------------------------------------------------------
    # Animated traces
    # ------------------------------------------------------------------
    # Ordering determines z-order on the waveform subplot (later = on top):
    #   1. Range window lines (if shown)
    #   2. Waveform fill
    #   3. Pulse markers (3D spatial subplot)
    #   4. LE start line (if shown)
    #   5. LE end line (if shown)
    #   6. POCA waveform marker (if shown)
    #   7. Nadir waveform marker (if shown)

    _HIDDEN_X = [-1e9, -1e9]

    animated_trace_indices = []

    def _add_animated(trace, row, col):
        fig.add_trace(trace, row=row, col=col)
        animated_trace_indices.append(len(fig.data) - 1)

    if show_range_window:
        _add_animated(
            go.Scatter(
                x=[range_window_start_bin, range_window_start_bin], y=[0, 1],
                mode="lines", showlegend=False, name="Range Window Start (Plot 2)",
                marker=dict(color="rgba(0, 255, 0, 1)"), line=dict(dash="dash"),
            ),
            row=1, col=2,
        )
        _add_animated(
            go.Scatter(
                x=[range_window_end_bin, range_window_end_bin], y=[0, 1],
                mode="lines", showlegend=False, name="Range Window End (Plot 2)",
                fill="tonextx", fillcolor="rgba(100, 255, 100, 0.15)",
                marker=dict(color="rgba(0, 255, 0, 1)"), line=dict(dash="dash"),
            ),
            row=1, col=2,
        )

    _add_animated(
        go.Scatter(
            y=[waveform[0]], name="Waveform",
            fill="tozeroy", marker=dict(color="skyblue"), line=dict(width=1), showlegend=False,
        ),
        row=1, col=2,
    )

    _add_animated(
        go.Scatter3d(
            x=[0] * num_rays_to_display,
            y=[0] * num_rays_to_display,
            z=[PLOT_RAY_ORIGIN_HEIGHT] * num_rays_to_display,
            mode="markers", showlegend=False, name="Pulses",
            marker=dict(color=ray_colours_pulse, size=2),
        ),
        row=1, col=1,
    )

    if show_leading_edge and le_detected:
        _add_animated(
            go.Scatter(
                x=_HIDDEN_X, y=[0, 1],
                mode="lines", showlegend=False, name="LE Start",
                line=dict(color=colour_le, dash="dot", width=1.5),
                opacity=0.4,
            ),
            row=1, col=2,
        )
        _add_animated(
            go.Scatter(
                x=_HIDDEN_X, y=[0, 1],
                mode="lines", showlegend=False, name="LE End",
                line=dict(color=colour_le, dash="dot", width=1.5),
                opacity=0.4,
            ),
            row=1, col=2,
        )

    if show_poca:
        _add_animated(
            go.Scatter(
                x=_HIDDEN_X[:1], y=[waveform[int(np.rint(poca_sim_bin))]/2],
                mode="markers", showlegend=False, name="POCA",
                marker=dict(
                    color=colour_poca, size=12, symbol="star",
                    line=dict(width=0.5, color="rgb(0,0,1)"),
                ),
            ),
            row=1, col=2,
        )

    if show_nadir:
        _add_animated(
            go.Scatter(
                x=_HIDDEN_X[:1], y=[waveform[int(np.rint(nadir_sim_bin))]],
                mode="markers", showlegend=False, name="Nadir",
                marker=dict(
                    color=colour_nadir, size=12, symbol="diamond",
                    line=dict(width=0.5, color="rgb(0,0,1)"),
                ),
            ),
            row=1, col=2,
        )

    # ------------------------------------------------------------------
    # Build animation frames
    # ------------------------------------------------------------------

    frames = []
    for frame in range(NUM_FRAMES):
        current_bin = frame * bins_per_frame
        frame_traces = []

        if show_range_window:
            frame_traces.append(
                go.Scatter(x=[range_window_start_bin, range_window_start_bin], y=[0, 1])
            )
            frame_traces.append(
                go.Scatter(x=[range_window_end_bin, range_window_end_bin], y=[0, 1])
            )

        frame_traces.append(go.Scatter(y=full_waveform[:current_bin]))
        frame_traces.append(
            go.Scatter3d(
                x=pulse_travel_frames[:, frame, 0],
                y=pulse_travel_frames[:, frame, 1],
                z=pulse_travel_frames[:, frame, 2],
            )
        )

        if show_leading_edge and le_detected:
            le_start_x = (
                [le_start_wf_bin, le_start_wf_bin] if frame >= le_start_reveal_frame else _HIDDEN_X
            )
            frame_traces.append(go.Scatter(x=le_start_x, y=[0, 1]))

            le_end_x = (
                [le_end_wf_bin, le_end_wf_bin] if frame >= le_end_reveal_frame else _HIDDEN_X
            )
            frame_traces.append(go.Scatter(x=le_end_x, y=[0, 1]))

        if show_poca:
            poca_x = [poca_wf_bin] if frame >= poca_reveal_frame else _HIDDEN_X[:1]
            frame_traces.append(go.Scatter(x=poca_x, y=[waveform[int(np.rint(poca_sim_bin))] / 2]))

        if show_nadir:
            nadir_x = [nadir_wf_bin] if frame >= nadir_reveal_frame else _HIDDEN_X[:1]
            frame_traces.append(go.Scatter(x=nadir_x, y=[waveform[int(np.rint(nadir_sim_bin))]]))

        frames.append(
            dict(
                name=str(frame),
                data=frame_traces,
                traces=animated_trace_indices,
            )
        )

    fig.frames = frames

    # ------------------------------------------------------------------
    # Layout and controls
    # ------------------------------------------------------------------

    fig.update_layout(
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [
                            None,
                            {
                                "frame": {"duration": 1000 / FPS, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                        "label": "▶ Play",
                        "method": "animate",
                    },
                    {
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                        "label": "⏸ Pause",
                        "method": "animate",
                    },
                ],
                "bgcolor": "#4a7fb5",
                "font": {"color": "white", "size": 13},
                "direction": "left",
                "pad": {"r": 0, "t": 50},
                "showactive": False,
                "type": "buttons",
                "x": 0.5,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top",
            }
        ],
        xaxis=dict(
            showgrid=False, visible=False,
            range=[-5, num_wf_bins_resampled + 5], fixedrange=True,
        ),
        yaxis=dict(showgrid=False, visible=False, range=[-0.25, 1.25], fixedrange=True),
        height=PLOT_HEIGHT,
        plot_bgcolor="rgba(0, 0, 0, 0)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        uirevision="animation",
    )

    if show_range_window:
        shell_z_min = float(np.nanmin(z_shell))
    else:
        shell_z_min = float(np.nanmin(topo_z))
    z_floor = min(float(np.nanmin(topo_z)), shell_z_min) - 0.1

    fig.update_scenes(
        xaxis=dict(showgrid=False, visible=False, range=[-PLOT_FOOTPRINT_RADIUS * 1.05, PLOT_FOOTPRINT_RADIUS * 1.05]),
        yaxis=dict(showgrid=False, visible=False, range=[-PLOT_FOOTPRINT_RADIUS * 1.05, PLOT_FOOTPRINT_RADIUS * 1.05]),
        zaxis=dict(showgrid=False, visible=False, range=[z_floor, sat_z + body_h + satellite_size * 0.05]),
        aspectratio=dict(x=1, y=1, z=1),
        camera=dict(
            eye=dict(x=1.1, y=1.1, z=0.5),
            center=dict(x=0, y=0, z=-0.1),
        ),
    )

    if output_path is not None:
        fig.write_html(f"{output_path}.html", auto_open=False, auto_play=False)

    return fig


# ----------------------------------------------------------------------
# Visual Tests
# ----------------------------------------------------------------------

if __name__ == "__main__":

    topography_2d = [0.1, 0.3, 0.8, 0.5, 0.2, 0.4, 0.9, 1, 1, 0.7, 0.3, 0.1, 0, 0]
    animate_altimetry_waveform_2d(
        topography_2d, output_path="2d_test",
        num_rays_to_display=50,
        range_window_top=1, range_window_bottom=0,
        show_poca=True, show_range_window=True, show_rays=True,
        wf_noise_amplitude=0.01,
    )

    topography_3d = np.tile(topography_2d, (len(topography_2d), 1))
    animate_altimetry_waveform_3d(
        topography_3d, output_path="3d_test",
        num_rays_to_display=75,
        range_window_top=1, range_window_bottom=0,
        show_poca=True, show_range_window=True, show_rays=True,
        wf_noise_amplitude=0.01,
    )