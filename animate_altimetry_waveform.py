#----------------------------------------------------------------------
# Imports
#----------------------------------------------------------------------

import numpy as np
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.optimize import root_scalar
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from simulate_altimetry_waveform import simulate_altimetry_waveform_2d, simulate_altimetry_waveform_3d, get_range_window_bottom_3d, compute_intersection_3d
import base64

#----------------------------------------------------------------------
# 2D Plotting (SAR-analogous)
#----------------------------------------------------------------------

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
        show_poca (bool): Whether to mark the Point of Closest Approach on the plot.
            Defaults to True.
        show_range_window (bool): Whether to display the range window on both subplots.
            Defaults to True.
        show_rays (bool): Whether to display the individual ray paths. Defaults to True.
        wf_noise_amplitude (float): Amplitude of uniform noise added to the waveform,
            as a fraction of the peak waveform value. Defaults to 0.01.
        satellite_image_path (str, optional): Path to a satellite image to display at
            the ray origin. Silently skipped if not found. Defaults to "s3.png".

    Returns:
        plotly.graph_objects.Figure: The animated Plotly figure.
    """

    #----------------------------------------------------------------------
    # Initialise parameters
    #----------------------------------------------------------------------

    plot_height = 800
    fps, animation_length = 24, 3
    num_frames, num_waveform_bins = fps * animation_length, 128

    topography = np.asarray(np.clip(topography, 0, 1), dtype="float")
    range_window_top = np.clip(range_window_top, 0.0, 1.0)
    range_window_bottom = np.clip(range_window_bottom, 0.0, range_window_top)
    range_window_scale = range_window_top - range_window_bottom
    range_window_offset = np.mean([range_window_top, range_window_bottom])

    #----------------------------------------------------------------------
    # Simulate waveform
    #----------------------------------------------------------------------

    # Scale topography to [0.2, 0.8] so the simulator (defined on [0, 1]) has
    # room for spillover beyond the topography extents
    topography_to_simulate = 0.2 + 0.6 * topography

    waveform, contribution_angle_scale = simulate_altimetry_waveform_2d(
        topography_to_simulate, return_contribution_angle_scale=True
    )

    # Interpolate contribution angle scale to match the number of displayed rays
    contribution_angle_scale = np.interp(
        np.linspace(0, 1, num_rays_to_display),
        np.linspace(0, 1, len(contribution_angle_scale)),
        contribution_angle_scale,
    )

    #----------------------------------------------------------------------
    # Compute range window geometry in the spatial plot
    #----------------------------------------------------------------------

    plot_ray_origin_height = 3  # Controls pulse travel time and total animation length
    plot_across_track_width = 2

    # Compute the curved bottom edge of the range window across the swath
    range_window_bottom_x = np.linspace(
        -plot_across_track_width / 2, plot_across_track_width / 2, num_rays_to_display
    )
    range_window_bottom_x[np.argwhere(range_window_bottom_x == 0)] = 0.00001  # Avoid degenerate ray at x=0
    range_window_bottom_z_at_nadir = range_window_offset - range_window_scale / 2
    offset_from_off_nadir_angle = plot_ray_origin_height - np.sqrt(
        plot_ray_origin_height**2 - range_window_bottom_x**2
    )
    range_window_bottom_z = offset_from_off_nadir_angle + range_window_bottom_z_at_nadir

    # Calculate and store the edge-vs-nadir offset for later
    edge_vs_nadir_offset_z = np.nanmax(range_window_bottom_z) - np.nanmin(range_window_bottom_z)

    #----------------------------------------------------------------------
    # Build the full waveform array
    #----------------------------------------------------------------------

    # The waveform covers the distance from the ray origin to the surface
    # The shrunken topography ([0.2, 0.8]) means the effective distance is 60% of the plot ray origin height
    waveform_distance_covered = plot_ray_origin_height * 0.6
    num_bins_in_full_waveform = int(np.rint(num_waveform_bins * waveform_distance_covered))

    # Place the simulated waveform at the end of a zero-padded array
    full_waveform = np.zeros(num_bins_in_full_waveform)
    full_waveform[-num_waveform_bins:] = waveform

    # Number of bins corresponding to the [0, 1] topography range (excluding spillover)
    num_waveform_bins_within_0_1_topography = int(np.rint(num_waveform_bins * 0.6))

    #----------------------------------------------------------------------
    # Compute range window bin positions in the waveform
    #----------------------------------------------------------------------

    # Right spillover is the simulated waveform bins from topo=1 (mapped to 0.8)
    # to the top of the simulator range (1.0)
    right_spillover_in_bins = int(np.rint(0.2 * num_waveform_bins))
    waveform_bins_within_0_1_topography_end = int(
        np.rint(num_bins_in_full_waveform - right_spillover_in_bins)
    )
    waveform_bins_within_0_1_topography_start = (
        waveform_bins_within_0_1_topography_end - num_waveform_bins_within_0_1_topography
    )

    # Calculate the range window bin positions, offsetting the start (top) by the edge-vs-nadir offset
    # so a topography value of 1 aligns with the swath edge rather than nadir
    edge_vs_nadir_offset_z_in_bins = (edge_vs_nadir_offset_z / waveform_distance_covered) * num_bins_in_full_waveform
    range_window_start_bin = waveform_bins_within_0_1_topography_start + num_waveform_bins_within_0_1_topography * range_window_bottom + edge_vs_nadir_offset_z_in_bins
    range_window_end_bin = waveform_bins_within_0_1_topography_start + num_waveform_bins_within_0_1_topography * range_window_top

    #----------------------------------------------------------------------
    # Pad waveform end and resample to fit animation frame count
    #----------------------------------------------------------------------

    # Append a short blank tail (20% of waveform length) for a cleaner visual ending
    bins_to_add_to_waveform_end = int(np.rint(num_waveform_bins * 0.2))
    full_waveform = np.concatenate((full_waveform, np.zeros(bins_to_add_to_waveform_end)))
    num_bins_in_full_waveform = len(full_waveform)

    # Resample so bin count is an exact multiple of frame count (ensures smooth animation)
    wf_bin_num_frames_ratio_ceil = int(np.ceil(num_bins_in_full_waveform / num_frames))
    num_wf_bins_to_sample_to = wf_bin_num_frames_ratio_ceil * num_frames
    full_waveform = np.interp(
        np.linspace(0, 1, num_wf_bins_to_sample_to),
        np.linspace(0, 1, num_bins_in_full_waveform),
        full_waveform,
    )

    # Rescale range window bin positions to match resampled waveform length
    _window_width = range_window_end_bin - range_window_start_bin
    range_window_start_bin = (range_window_start_bin / num_bins_in_full_waveform) * num_wf_bins_to_sample_to
    range_window_end_bin   = range_window_start_bin + (_window_width / num_bins_in_full_waveform) * num_wf_bins_to_sample_to
    num_bins_in_full_waveform = len(full_waveform)

    # Add uniform noise scaled to the waveform peak
    if wf_noise_amplitude > 0:
        noise = np.random.uniform(
            low=0,
            high=wf_noise_amplitude * np.max(waveform),
            size=num_bins_in_full_waveform,
        )
        full_waveform += noise

    #----------------------------------------------------------------------
    # Compute ray-topography intersections
    #----------------------------------------------------------------------

    plot_ray_origin_coord = np.array([0, plot_ray_origin_height], dtype=float)
    topography_x = np.linspace(-1, 1, len(topography))
    topography_spline = interp1d(topography_x, topography, fill_value="extrapolate", kind="slinear")

    plot_bisections = np.full((num_rays_to_display, 2), np.nan)
    for ray in range(num_rays_to_display):
        ray_func = interp1d(
            [plot_ray_origin_coord[0], range_window_bottom_x[ray]],
            [plot_ray_origin_coord[1], range_window_bottom_z[ray]],
            fill_value="extrapolate",
        )
        # Find where the ray and topography surface intersect
        f = lambda x: topography_spline(x) - ray_func(x)
        try:
            root = root_scalar(
                f,
                x0=range_window_bottom_x[ray],
                bracket=[-plot_across_track_width / 2, plot_across_track_width / 2],
            ).root
            plot_bisections[ray] = [root, topography_spline(root)]
        except Exception:
            continue

    #----------------------------------------------------------------------
    # Compute pulse travel frames
    #----------------------------------------------------------------------

    plot_ray_vecs = plot_ray_origin_coord - plot_bisections
    plot_ray_lengths = np.linalg.norm(plot_ray_vecs, axis=-1)
    ray_unit_vecs = plot_ray_vecs / plot_ray_lengths[:, np.newaxis]
    pulse_travel_distances = plot_ray_lengths * 2  # Total out-and-back distance per ray

    # Derive pulse position from waveform bin position, keeping both plots on a single clock
    # 1.2 factor in distance travelled accounts for the [0.2, 0.8] simulator topo scaling
    current_bins = np.arange(num_frames) * wf_bin_num_frames_ratio_ceil
    dist_travelled = (current_bins / num_wf_bins_to_sample_to) * plot_ray_origin_height * 1.2 * 2

    pulse_travel_frames = np.full((num_rays_to_display, num_frames, 2), np.nan)
    for ray in range(num_rays_to_display):
        if np.isnan(plot_bisections[ray]).any():
            continue

        outgoing = dist_travelled <= plot_ray_lengths[ray]
        returning = (dist_travelled > plot_ray_lengths[ray]) & (dist_travelled <= pulse_travel_distances[ray])

        # Outgoing: pulse travelling from origin towards surface
        pulse_travel_frames[ray, outgoing] = (
            plot_ray_origin_coord - dist_travelled[outgoing, np.newaxis] * ray_unit_vecs[ray]
        )

        # Returning: pulse travelling back from surface towards origin
        dist_back = dist_travelled[returning] - plot_ray_lengths[ray]
        pulse_travel_frames[ray, returning] = (
            plot_bisections[ray] + dist_back[:, np.newaxis] * ray_unit_vecs[ray]
        )
        # After return: pulse is invisible (remains NaN)

    #----------------------------------------------------------------------
    # Determine POCA
    #----------------------------------------------------------------------

    valid_mask = ~np.isnan(plot_bisections).any(axis=1)
    distances = np.linalg.norm(plot_bisections[valid_mask] - plot_ray_origin_coord, axis=1)
    reduced_index = np.argmin(distances) if show_poca else None
    poca_index = np.where(valid_mask)[0][reduced_index] if show_poca else None

    #----------------------------------------------------------------------
    # Define ray and pulse colours
    #----------------------------------------------------------------------

    # Scale opacity by the antenna gain contribution at each ray angle
    ray_colour_power_factor = 2.5  # Accentuates the opacity falloff with angle
    ray_colours_pulse = [
        f"rgba(255, 0, 0, {contribution_angle_scale[ray] ** ray_colour_power_factor})"
        for ray in range(num_rays_to_display)
    ]

    #----------------------------------------------------------------------
    # Build static plot traces
    #----------------------------------------------------------------------

    fig = make_subplots(rows=1, cols=2, column_widths=[0.65, 0.35], horizontal_spacing=0.01)

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
    num_traces_before_animations = 2

    # Rays paths 
    if show_rays:
        for ray in range(num_rays_to_display):
            alpha = max(0.04, contribution_angle_scale[ray] ** ray_colour_power_factor / 2.5)
            fig.add_trace(
                go.Scatter(
                    x=[0, plot_bisections[ray, 0]],
                    y=[plot_ray_origin_height, plot_bisections[ray, 1]],
                    mode="lines", showlegend=False, name=f"Ray {ray}",
                    line=dict(color=f"rgba(255, 0, 0, {alpha:.3f})", dash="dash"),
                ),
                row=1, col=1,
            )
            num_traces_before_animations += 1

    # POCA marker
    if show_poca:
        fig.add_trace(
            go.Scatter(
                x=[plot_bisections[poca_index, 0]], y=[plot_bisections[poca_index, 1]],
                mode="markers", showlegend=False, name="POCA",
                marker=dict(color="yellow", size=12, symbol="star", line=dict(width=1.5, color="orange")),
            ),
            row=1, col=1,
        )
        num_traces_before_animations += 1

    # Range window overlay on the spatial plot
    # Offset range window top so topography of 1 aligns with the swath edge rather than nadir
    if show_range_window:
        fig.add_trace(
            go.Scatter(
                x=range_window_bottom_x, y=range_window_bottom_z,
                mode="lines", line_shape="spline", showlegend=False,
                name="Range Window Start (Plot 1)", fillcolor="rgba(100, 255, 100, 0.15)",
                marker=dict(color="rgba(0, 255, 0, 1)"), line=dict(dash="dash"),
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=range_window_bottom_x, y=range_window_bottom_z + range_window_scale - edge_vs_nadir_offset_z,
                mode="lines", line_shape="spline", showlegend=False,
                name="Range Window End (Plot 1)", fill="tonexty",
                fillcolor="rgba(100, 255, 100, 0.15)",
                marker=dict(color="rgba(0, 255, 0, 1)"), line=dict(dash="dash"),
            ),
            row=1, col=1,
        )
        num_traces_before_animations += 2

        # Range window bounds on the waveform plot
        fig.add_trace(
            go.Scatter(
                x=[range_window_start_bin, range_window_start_bin], y=[0, 1],
                mode="lines", showlegend=False, name="Range Window Start (Plot 2)",
                marker=dict(color="rgba(0, 255, 0, 1)"), line=dict(dash="dash"),
            ),
            row=1, col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=[range_window_end_bin, range_window_end_bin], y=[0, 1],
                mode="lines", showlegend=False, name="Range Window End (Plot 2)",
                fill="tonextx", fillcolor="rgba(100, 255, 100, 0.15)",
                marker=dict(color="rgba(0, 255, 0, 1)"), line=dict(dash="dash"),
            ),
            row=1, col=2,
        )

    # Waveform trace (initialised with the first sample)
    fig.add_trace(
        go.Scatter(
            y=[waveform[0]], name="Waveform",
            fill="tozeroy", marker=dict(color="skyblue"), line=dict(width=1), showlegend=False,
        ),
        row=1, col=2,
    )

    # Pulse position markers (one per ray)
    fig.add_trace(
        go.Scatter(
            x=[0] * num_rays_to_display, y=[plot_ray_origin_height] * num_rays_to_display,
            mode="markers", showlegend=False, name="Pulses",
            marker=dict(color=ray_colours_pulse),
        ),
        row=1, col=1,
    )

    #----------------------------------------------------------------------
    # Build animation frames
    #----------------------------------------------------------------------

    # The range window lines on the waveform plot are redrawn each frame to
    # work around a Plotly bug where filled traces can disappear during animation
    waveform_range_window_start_bin_data = [
        go.Scatter(x=[range_window_start_bin, range_window_start_bin], y=[0, 1])
        for _ in range(num_frames)
    ]
    waveform_range_window_end_bin_data = [
        go.Scatter(x=[range_window_end_bin, range_window_end_bin], y=[0, 1])
        for _ in range(num_frames)
    ]
    waveform_frame_data = [
        go.Scatter(y=full_waveform[0 : frame * wf_bin_num_frames_ratio_ceil])
        for frame in range(num_frames)
    ]
    pulse_frame_data = [
        go.Scatter(x=pulse_travel_frames[:, frame, 0], y=pulse_travel_frames[:, frame, 1])
        for frame in range(num_frames)
    ]

    if show_range_window:
        frame_data = [
            [waveform_range_window_start_bin_data[frame]]
            + [waveform_range_window_end_bin_data[frame]]
            + [waveform_frame_data[frame]]
            + [pulse_frame_data[frame]]
            for frame in range(num_frames)
        ]
        range_window_animation_traces = 2
    else:
        frame_data = [
            [waveform_frame_data[frame]] + [pulse_frame_data[frame]]
            for frame in range(num_frames)
        ]
        range_window_animation_traces = 0

    frames = [
        dict(
            name=str(frame),
            data=frame_data[frame],
            traces=np.arange(
                num_traces_before_animations,
                num_traces_before_animations + 2 + range_window_animation_traces,
            ),
        )
        for frame in range(num_frames)
    ]

    #----------------------------------------------------------------------
    # Finalise layout and controls
    #----------------------------------------------------------------------

    fig.frames = frames
    updatemenus = [
        {
            "buttons": [
                {
                    "args": [
                        None,
                        {
                            "frame": {"duration": 1000 / fps, "redraw": True},
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
                    "label": "⏸Pause",
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
    ]

    # Define x and y ranges for the spatial plot (plot 1)
    x_range = [-(plot_across_track_width * 1.1) / 2, (plot_across_track_width * 1.1) / 2]
    y_range = [-edge_vs_nadir_offset_z - 0.1, plot_ray_origin_height]

    # Add satellite image at ray origin if found - stored as s3.svg
    # https://upload.wikimedia.org/wikipedia/commons/f/f7/Sentinel-3_spacecraft_model.svg
    # (Reduced in size and edited to have antenna centered at x=0)
    try:
        with open(satellite_image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()

        col1_paper_width = 0.65 - 0.01

        x_paper = (plot_ray_origin_coord[0] - x_range[0]) / (x_range[1] - x_range[0]) * col1_paper_width
        y_paper = (plot_ray_origin_coord[1] - y_range[0]) / (y_range[1] - y_range[0])

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
        updatemenus=updatemenus,
        dragmode=False,
        xaxis=dict(
            showgrid=False, visible=False,
            range=x_range, fixedrange=True,
        ),
        yaxis=dict(
            showgrid=False, visible=False,
            range=y_range, fixedrange=True,
        ),
        xaxis2=dict(
            showgrid=False, visible=False,
            range=[-5, num_wf_bins_to_sample_to + 5], fixedrange=True,
        ),
        yaxis2=dict(
            showgrid=False, visible=False,
            range=[-0.25, 1.25], fixedrange=True,
        ),
        height=plot_height,
        plot_bgcolor="rgba(0, 0, 0, 0)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
    )

    if output_path is not None:
        fig.write_html(f"{output_path}.html", auto_open=False, auto_play=False)

    return fig


#----------------------------------------------------------------------
# 3D Plotting (LRM-analogous)
#----------------------------------------------------------------------

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
        show_poca (bool): Whether to mark the Point of Closest Approach on the plot.
            Defaults to True.
        show_range_window (bool): Whether to display the range window on both subplots.
            Defaults to True.
        show_rays (bool): Whether to display the individual ray paths. Defaults to True.
        wf_noise_amplitude (float): Amplitude of uniform noise added to the waveform,
            as a fraction of the peak waveform value. Defaults to 0.01.

    Returns:
        plotly.graph_objects.Figure: The animated Plotly figure.
    """

    #----------------------------------------------------------------------
    # Initialise parameters
    #----------------------------------------------------------------------

    plot_height = 800
    fps, animation_length = 24, 3
    num_frames, num_waveform_bins = fps * animation_length, 128

    topography = np.asarray(np.clip(topography, 0, 1), dtype="float")
    range_window_top = np.clip(range_window_top, 0.0, 1.0)
    range_window_bottom = np.clip(range_window_bottom, 0.0, range_window_top)
    range_window_scale = range_window_top - range_window_bottom
    range_window_offset = np.mean([range_window_top, range_window_bottom])

    #----------------------------------------------------------------------
    # Simulate waveform
    #----------------------------------------------------------------------

    # Scale topography to [0.2, 0.8] so the simulator has room for spillover
    # beyond the topography extents
    topography_to_simulate = 0.2 + 0.6 * topography

    waveform, contribution_angle_scale = simulate_altimetry_waveform_3d(
        topography_to_simulate, return_contribution_angle_scale=True
    )

    #----------------------------------------------------------------------
    # Compute range window geometry in the spatial plot
    #----------------------------------------------------------------------

    plot_ray_origin_height = 3
    plot_footprint_radius = 1

    # Build the hexagonal ray grid at the base of the range window for the plot
    plot_range_window_bottom, _ = get_range_window_bottom_3d(
        num_rays_to_display, plot_footprint_radius, range_window_scale, plot_ray_origin_height
    )
    num_rays_to_display = len(plot_range_window_bottom)

    # Shift z values so the nadir point of the range window bottom lands at
    # range_window_offset - range_window_scale/2
    range_window_bottom_z_at_nadir = range_window_offset - range_window_scale / 2
    plot_range_window_bottom[:, 2] += range_window_bottom_z_at_nadir

    # Edge-vs-nadir z offset: the extra depth at the footprint edge due to
    # the curved range window bottom
    edge_vs_nadir_offset_z = np.nanmax(plot_range_window_bottom[:, 2]) - np.nanmin(plot_range_window_bottom[:, 2])

    # Interpolate contribution angle scale to match the number of displayed rays
    contribution_angle_scale = np.interp(
        np.linspace(0, 1, num_rays_to_display),
        np.linspace(0, 1, len(contribution_angle_scale)),
        contribution_angle_scale,
    )

    #----------------------------------------------------------------------
    # Build the full waveform array
    #----------------------------------------------------------------------

    # The [0.2, 0.8] topography scaling means the waveform covers 60% of the
    # plot ray origin height
    waveform_distance_covered = plot_ray_origin_height * 0.6
    num_bins_in_full_waveform = int(np.rint(num_waveform_bins * waveform_distance_covered))

    full_waveform = np.zeros(num_bins_in_full_waveform)
    full_waveform[-num_waveform_bins:] = waveform

    num_waveform_bins_within_0_1_topography = int(np.rint(num_waveform_bins * 0.6))

    #----------------------------------------------------------------------
    # Compute range window bin positions in the waveform
    #----------------------------------------------------------------------

    # Right spillover: bins from topo=1 (mapped to 0.8) to the top of the
    # simulator range (1.0)
    right_spillover_in_bins = int(np.rint(0.2 * num_waveform_bins))
    waveform_bins_within_0_1_topography_end = int(
        np.rint(num_bins_in_full_waveform - right_spillover_in_bins)
    )
    waveform_bins_within_0_1_topography_start = (
        waveform_bins_within_0_1_topography_end - num_waveform_bins_within_0_1_topography
    )

    # Offset the start bin by the edge-vs-nadir shift so a topography value of
    # 1 aligns with the footprint edge rather than nadir
    edge_vs_nadir_offset_z_in_bins = (edge_vs_nadir_offset_z / waveform_distance_covered) * num_bins_in_full_waveform
    range_window_start_bin = (
        waveform_bins_within_0_1_topography_start
        + num_waveform_bins_within_0_1_topography * range_window_bottom
        + edge_vs_nadir_offset_z_in_bins
    )
    range_window_end_bin = (
        waveform_bins_within_0_1_topography_start
        + num_waveform_bins_within_0_1_topography * range_window_top
    )

    #----------------------------------------------------------------------
    # Pad waveform end and resample to fit animation frame count
    #----------------------------------------------------------------------

    # Append a short blank tail for a cleaner visual ending
    bins_to_add_to_waveform_end = int(np.rint(num_waveform_bins * 0.2))
    full_waveform = np.concatenate((full_waveform, np.zeros(bins_to_add_to_waveform_end)))
    num_bins_in_full_waveform = len(full_waveform)

    # Resample so bin count is an exact multiple of frame count
    wf_bin_num_frames_ratio_ceil = int(np.ceil(num_bins_in_full_waveform / num_frames))
    num_wf_bins_to_sample_to = wf_bin_num_frames_ratio_ceil * num_frames
    full_waveform = np.interp(
        np.linspace(0, 1, num_wf_bins_to_sample_to),
        np.linspace(0, 1, num_bins_in_full_waveform),
        full_waveform,
    )

    # Rescale range window bin positions to match resampled waveform length
    _window_width = range_window_end_bin - range_window_start_bin
    range_window_start_bin = (range_window_start_bin / num_bins_in_full_waveform) * num_wf_bins_to_sample_to
    range_window_end_bin   = range_window_start_bin + (_window_width / num_bins_in_full_waveform) * num_wf_bins_to_sample_to
    num_bins_in_full_waveform = len(full_waveform)

    if wf_noise_amplitude > 0:
        noise = np.random.uniform(
            low=0,
            high=wf_noise_amplitude * np.max(waveform),
            size=num_bins_in_full_waveform,
        )
        full_waveform += noise

    #----------------------------------------------------------------------
    # Build topography interpolator and compute ray-surface intersections
    #----------------------------------------------------------------------

    nx, ny = np.shape(topography)
    x = np.linspace(-plot_footprint_radius, plot_footprint_radius, nx)
    y = np.linspace(-plot_footprint_radius, plot_footprint_radius, ny)
    topography_interpolator = RegularGridInterpolator(
        (x, y), topography, bounds_error=False, method="slinear",
        fill_value=None 
    )

    plot_ray_origin_coord = np.array([0, 0, plot_ray_origin_height], dtype=float)
    plot_intersections = np.full((num_rays_to_display, 3), np.nan)
    for ray in range(num_rays_to_display):
        plot_intersections[ray] = compute_intersection_3d(
            ray, plot_ray_origin_coord, plot_range_window_bottom, topography_interpolator, root_bracket=[0, 1.5]
        )

    #----------------------------------------------------------------------
    # Compute pulse travel frames
    #----------------------------------------------------------------------

    plot_ray_vecs = plot_ray_origin_coord - plot_intersections
    plot_ray_lengths = np.linalg.norm(plot_ray_vecs, axis=-1)
    ray_unit_vecs = plot_ray_vecs / plot_ray_lengths[:, np.newaxis]
    pulse_travel_distances = plot_ray_lengths * 2

    # Derive pulse position from waveform bin position, keeping both plots on a single clock
    # The 1.2 factor accounts for the [0.2, 0.8] topography scaling
    current_bins = np.arange(num_frames) * wf_bin_num_frames_ratio_ceil
    dist_travelled = (current_bins / num_wf_bins_to_sample_to) * plot_ray_origin_height * 1.2 * 2

    pulse_travel_frames = np.full((num_rays_to_display, num_frames, 3), np.nan)
    for ray in range(num_rays_to_display):
        if np.isnan(plot_intersections[ray]).any():
            continue

        outgoing = dist_travelled <= plot_ray_lengths[ray]
        returning = (dist_travelled > plot_ray_lengths[ray]) & (dist_travelled <= pulse_travel_distances[ray])

        # Outgoing: pulse travelling from origin towards surface
        pulse_travel_frames[ray, outgoing] = (
            plot_ray_origin_coord - dist_travelled[outgoing, np.newaxis] * ray_unit_vecs[ray]
        )

        # Returning: pulse travelling back from surface towards origin
        dist_back = dist_travelled[returning] - plot_ray_lengths[ray]
        pulse_travel_frames[ray, returning] = (
            plot_intersections[ray] + dist_back[:, np.newaxis] * ray_unit_vecs[ray]
        )
        # After return: pulse is invisible (remains NaN)

    #----------------------------------------------------------------------
    # Determine POCA
    #----------------------------------------------------------------------

    valid_mask = ~np.isnan(plot_intersections).any(axis=1)
    distances = np.linalg.norm(plot_intersections[valid_mask] - plot_ray_origin_coord, axis=1)
    reduced_index = np.argmin(distances) if show_poca else None
    poca_index = np.where(valid_mask)[0][reduced_index] if show_poca else None

    #----------------------------------------------------------------------
    # Define ray and pulse colours
    #----------------------------------------------------------------------

    ray_colour_power_factor = 2.5
    ray_colours_pulse = [
        f"rgba(255, 0, 0, {contribution_angle_scale[ray] ** ray_colour_power_factor})"
        for ray in range(num_rays_to_display)
    ]

    #----------------------------------------------------------------------
    # Build static plot traces
    #----------------------------------------------------------------------

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
    num_traces_before_animations = 1

    # Once trace per ring as rays on the same ring share the same off-nadir angle and antenna gain
    if show_rays:
        dash_length = 0.04
        gap_length  = 0.025

        ray_radii = np.sqrt(
            plot_range_window_bottom[:, 0] ** 2 + plot_range_window_bottom[:, 1] ** 2
        )

        # Round to the nearest hex grid spacing to cluster rays into rings
        area = np.pi * plot_footprint_radius ** 2
        d_hex = np.sqrt(area / num_rays_to_display)
        ring_ids = np.round(ray_radii / d_hex).astype(int)
        unique_rings = np.unique(ring_ids)

        origin = np.array([0.0, 0.0, plot_ray_origin_height])

        for ring_id in unique_rings:
            ring_mask = ring_ids == ring_id
            ring_ray_indices = np.where(ring_mask)[0]

            ring_scale = float(np.mean(contribution_angle_scale[ring_ray_indices]))
            alpha = max(0.04, ring_scale ** ray_colour_power_factor / 2.5)

            rays_x, rays_y, rays_z = [], [], []
            for ray in ring_ray_indices:
                if np.isnan(plot_intersections[ray]).any():
                    continue
                end      = plot_intersections[ray]
                ray_vec  = end - origin
                ray_len  = np.linalg.norm(ray_vec)
                ray_unit = ray_vec / ray_len

                t = 0.0
                drawing = True
                while t < ray_len:
                    seg_len = dash_length if drawing else gap_length
                    t_end   = min(t + seg_len, ray_len)
                    if drawing:
                        p0 = origin + t     * ray_unit
                        p1 = origin + t_end * ray_unit
                        rays_x += [p0[0], p1[0], None]
                        rays_y += [p0[1], p1[1], None]
                        rays_z += [p0[2], p1[2], None]
                    t       = t_end
                    drawing = not drawing

            fig.add_trace(
                go.Scatter3d(
                    x=rays_x, y=rays_y, z=rays_z,
                    mode="lines", showlegend=False, name=f"Ring {ring_id}",
                    line=dict(color=f"rgba(255, 0, 0, {alpha:.3f})"),
                ),
                row=1, col=1,
            )
            num_traces_before_animations += 1

    # POCA marker
    if show_poca:
        fig.add_trace(
            go.Scatter3d(
                x=[plot_intersections[poca_index, 0]],
                y=[plot_intersections[poca_index, 1]],
                z=[plot_intersections[poca_index, 2] + 0.01],
                mode="markers", showlegend=False, name="POCA",
                marker=dict(color="yellow", size=4, symbol="x"),
            ),
            row=1, col=1,
        )
        num_traces_before_animations += 1

    # Range window visualised as a curved spherical shell segment 
    if show_range_window:
        z_at_range_window_bottom = range_window_bottom_z_at_nadir

        # Solve for the top nadir z-level iteratively: the outermost ring of
        # the top cap must sit at range_window_bottom_z_at_nadir + range_window_scale
        # after wavefront curvature is applied
        top_correction = edge_vs_nadir_offset_z
        for _ in range(20):
            _nadir_z_top = range_window_bottom_z_at_nadir + range_window_scale - top_correction
            _r_top = plot_footprint_radius * (_nadir_z_top - plot_ray_origin_height) / (range_window_bottom_z_at_nadir - plot_ray_origin_height)
            top_correction = plot_ray_origin_height - np.sqrt(max(plot_ray_origin_height ** 2 - _r_top ** 2, 0))
        z_at_range_window_top = range_window_bottom_z_at_nadir + range_window_scale - top_correction

        circumference = 2 * np.pi * plot_footprint_radius
        area = np.pi * plot_footprint_radius ** 2
        d = np.sqrt(area / num_rays_to_display)
        num_points_outer_ring = int(np.round(circumference / d))
        num_latitude_rings = max(8, num_points_outer_ring // 4)
        theta = np.linspace(0, 2 * np.pi, num_points_outer_ring, endpoint=False)

        # Each latitude ring is linearly interpolated in nadir z between bottom and top
        # Ring radius scales with distance from the satellite, and each point's actual z is raised by the wavefront curvature offset
        nadir_z_levels = np.linspace(z_at_range_window_bottom, z_at_range_window_top, num_latitude_rings)
        ring_radii = (
            plot_footprint_radius
            * (nadir_z_levels - plot_ray_origin_height)
            / (z_at_range_window_bottom - plot_ray_origin_height)
        )

        rings_x = np.outer(ring_radii, np.cos(theta))
        rings_y = np.outer(ring_radii, np.sin(theta))
        radial_distances = np.abs(ring_radii)[:, np.newaxis] * np.ones((1, num_points_outer_ring))
        curvature_offset = plot_ray_origin_height - np.sqrt(
            np.maximum(plot_ray_origin_height ** 2 - radial_distances ** 2, 0)
        )
        rings_z = nadir_z_levels[:, np.newaxis] + curvature_offset

        x_shell = rings_x.ravel()
        y_shell = rings_y.ravel()
        z_shell = rings_z.ravel()

        # Triangulate the shell by connecting adjacent rings with quads split diagonally
        i_idx, j_idx, k_idx = [], [], []
        n = num_points_outer_ring
        for r in range(num_latitude_rings - 1):
            for p in range(n):
                p_next = (p + 1) % n
                v00 = r * n + p
                v01 = r * n + p_next
                v10 = (r + 1) * n + p
                v11 = (r + 1) * n + p_next
                i_idx.append(v00); j_idx.append(v01); k_idx.append(v10)
                i_idx.append(v01); j_idx.append(v11); k_idx.append(v10)

        # Cap disks at bottom and top
        # Three intermediate concentric rings at 3/4, 1/2, and 1/4 of the cap radius fill each disk, 
        # with a fan from the innermost ring to a centre point
        for cap_nadir_z, cap_radius, outer_ring_start in [
            (z_at_range_window_bottom, ring_radii[0],                    0                          ),
            (z_at_range_window_top,    ring_radii[num_latitude_rings-1], (num_latitude_rings - 1) * n),
        ]:
            inner_rings = []
            for frac in (3/4, 1/2, 1/4):
                r_ring = cap_radius * frac
                curv   = plot_ray_origin_height - np.sqrt(
                    max(plot_ray_origin_height ** 2 - r_ring ** 2, 0)
                )
                ring_x = r_ring * np.cos(theta)
                ring_y = r_ring * np.sin(theta)
                ring_z = np.full(n, cap_nadir_z + curv)
                start  = len(x_shell)
                x_shell = np.concatenate([x_shell, ring_x])
                y_shell = np.concatenate([y_shell, ring_y])
                z_shell = np.concatenate([z_shell, ring_z])
                inner_rings.append(start)

            cap_centre_idx = len(x_shell)
            x_shell = np.append(x_shell, 0.0)
            y_shell = np.append(y_shell, 0.0)
            z_shell = np.append(z_shell, cap_nadir_z)

            rings_in_order = [outer_ring_start] + inner_rings
            for ri in range(len(rings_in_order) - 1):
                r_outer_start = rings_in_order[ri]
                r_inner_start = rings_in_order[ri + 1]
                for p in range(n):
                    p_next = (p + 1) % n
                    v_o0 = r_outer_start + p;      v_o1 = r_outer_start + p_next
                    v_i0 = r_inner_start + p;      v_i1 = r_inner_start + p_next
                    i_idx.append(v_o0); j_idx.append(v_o1); k_idx.append(v_i0)
                    i_idx.append(v_o1); j_idx.append(v_i1); k_idx.append(v_i0)

            innermost_start = rings_in_order[-1]
            for p in range(n):
                p_next = (p + 1) % n
                i_idx.append(cap_centre_idx)
                j_idx.append(innermost_start + p)
                k_idx.append(innermost_start + p_next)

        fig.add_trace(
            go.Mesh3d(
                x=x_shell, y=y_shell, z=z_shell,
                i=i_idx, j=j_idx, k=k_idx,
                color="green", opacity=0.05,
                name="Range Window (Plot 1)", showlegend=False,
            ),
            row=1, col=1,
        )
        num_traces_before_animations += 1

        # Range window bounds on the waveform plot
        fig.add_trace(
            go.Scatter(
                x=[range_window_start_bin, range_window_start_bin], y=[0, 1],
                mode="lines", showlegend=False, name="Range Window Start (Plot 2)",
                marker=dict(color="rgba(0, 255, 0, 1)"), line=dict(dash="dash"),
            ),
            row=1, col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=[range_window_end_bin, range_window_end_bin], y=[0, 1],
                mode="lines", showlegend=False, name="Range Window End (Plot 2)",
                fill="tonextx", fillcolor="rgba(100, 255, 100, 0.15)",
                marker=dict(color="rgba(0, 255, 0, 1)"), line=dict(dash="dash"),
            ),
            row=1, col=2,
        )
        num_traces_before_animations += 2

    # Waveform trace
    waveform_trace_index = num_traces_before_animations
    fig.add_trace(
        go.Scatter(
            y=[waveform[0]], name="Waveform",
            fill="tozeroy", marker=dict(color="skyblue"), line=dict(width=1), showlegend=False,
        ),
        row=1, col=2,
    )
    num_traces_before_animations += 1

    # Pulse position markers, one per ray
    pulse_trace_index = num_traces_before_animations
    fig.add_trace(
        go.Scatter3d(
            x=[0] * num_rays_to_display,
            y=[0] * num_rays_to_display,
            z=[plot_ray_origin_height] * num_rays_to_display,
            mode="markers", showlegend=False, name="Pulses",
            marker=dict(color=ray_colours_pulse, size=2),
        ),
        row=1, col=1,
    )
    num_traces_before_animations += 1

    #----------------------------------------------------------------------
    # Build animation frames
    #----------------------------------------------------------------------

    # The range window lines on the waveform plot are redrawn each frame to
    # work around a Plotly bug where filled traces can disappear during animation
    waveform_frame_data = [
        go.Scatter(y=full_waveform[0 : frame * wf_bin_num_frames_ratio_ceil])
        for frame in range(num_frames)
    ]
    pulse_frame_data = [
        go.Scatter3d(
            x=pulse_travel_frames[:, frame, 0],
            y=pulse_travel_frames[:, frame, 1],
            z=pulse_travel_frames[:, frame, 2],
        )
        for frame in range(num_frames)
    ]

    if show_range_window:
        rw_start_trace_index = waveform_trace_index - 2
        rw_end_trace_index   = waveform_trace_index - 1

        waveform_range_window_start_bin_data = [
            go.Scatter(x=[range_window_start_bin, range_window_start_bin], y=[0, 1])
            for _ in range(num_frames)
        ]
        waveform_range_window_end_bin_data = [
            go.Scatter(x=[range_window_end_bin, range_window_end_bin], y=[0, 1])
            for _ in range(num_frames)
        ]

        frame_data = [
            [
                waveform_range_window_start_bin_data[frame],
                waveform_range_window_end_bin_data[frame],
                waveform_frame_data[frame],
                pulse_frame_data[frame],
            ]
            for frame in range(num_frames)
        ]
        animated_trace_indices = [
            rw_start_trace_index,
            rw_end_trace_index,
            waveform_trace_index,
            pulse_trace_index,
        ]
    else:
        frame_data = [
            [waveform_frame_data[frame], pulse_frame_data[frame]]
            for frame in range(num_frames)
        ]
        animated_trace_indices = [waveform_trace_index, pulse_trace_index]

    frames = [
        dict(
            name=str(frame),
            data=frame_data[frame],
            traces=animated_trace_indices,
        )
        for frame in range(num_frames)
    ]

    #----------------------------------------------------------------------
    # Finalise layout and controls
    #----------------------------------------------------------------------

    fig.frames = frames
    updatemenus = [
        {
            "buttons": [
                {
                    "args": [
                        None,
                        {
                            "frame": {"duration": 1000 / fps, "redraw": True},
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
    ]

    fig.update_layout(
        updatemenus=updatemenus,
        xaxis=dict(showgrid=False, visible=False, range=[-5, num_wf_bins_to_sample_to + 5], fixedrange=True),
        yaxis=dict(showgrid=False, visible=False, range=[-0.25, 1.25], fixedrange=True),
        height=plot_height,
        plot_bgcolor="rgba(0, 0, 0, 0)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
    )

    #----------------------------------------------------------------------
    # Satellite marker (static)
    #----------------------------------------------------------------------

    # The satellite is represented as filled quads (go.Mesh3d) with a darker
    # outline (go.Scatter3d) for the bus body and two solar panels 
    # Both an xz-plane and a yz-plane version are drawn so the marker is visible from any view
    satellite_size = plot_footprint_radius * 0.15
    sat_x, sat_y, sat_z = 0.0, 0.0, float(plot_ray_origin_height)
    body_h  = satellite_size * 0.6
    body_w  = satellite_size * 0.35
    panel_w = satellite_size
    panel_h = satellite_size * 0.25

    bus_fill_colour      = "rgb(210, 215, 225)"
    bus_outline_colour   = "rgb(130, 135, 145)"
    panel_fill_colour    = "rgb(120, 145, 185)"
    panel_outline_colour = "rgb( 80, 100, 140)"

    def _rect_mesh(cx, cy, cz, hw, hh, axis):
        """Return (x, y, z, i, j, k) for a filled rectangle as two triangles.

        The rectangle is centred at (cx, cy, cz) with half-width hw and
        half-height hh, lying in the xz-plane (axis='x') or yz-plane (axis='y').
        """
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
        """Return (x, y, z) for the closed outline of the same rectangle."""
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
        left_centre   = sat_x - (body_w + panel_w) if axis == "x" else sat_x
        left_centre_y = sat_y if axis == "x" else sat_y - (body_w + panel_w)
        right_centre   = sat_x + (body_w + panel_w) if axis == "x" else sat_x
        right_centre_y = sat_y if axis == "x" else sat_y + (body_w + panel_w)

        shapes = [
            (sat_x,        sat_y,         sat_z, body_w,  body_h,  bus_fill_colour,   bus_outline_colour,   "Bus"),
            (left_centre,  left_centre_y, sat_z, panel_w, panel_h, panel_fill_colour, panel_outline_colour, "Panel L"),
            (right_centre, right_centre_y,sat_z, panel_w, panel_h, panel_fill_colour, panel_outline_colour, "Panel R"),
        ]

        bus_out_x,   bus_out_y,   bus_out_z   = [], [], []
        panel_out_x, panel_out_y, panel_out_z = [], [], []

        for (cx, cy, cz, hw, hh, fill_col, outline_col, label) in shapes:
            fx, fy, fz, fi, fj, fk = _rect_mesh(cx, cy, cz, hw, hh, axis)
            fig.add_trace(
                go.Mesh3d(
                    x=fx, y=fy, z=fz,
                    i=fi, j=fj, k=fk,
                    color=fill_col, opacity=1.0,
                    flatshading=True,
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

        # Antenna stub
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

    if show_range_window:
        shell_z_min = float(np.nanmin(z_shell))
    else:
        shell_z_min = float(np.nanmin(topo_z))
    z_floor = min(float(np.nanmin(topo_z)), shell_z_min) - 0.1

    camera_eye_x = 1.1   
    camera_eye_y = 1.1
    camera_eye_z = 0.5     
    camera_center_z = -0.1 
    fig.update_scenes(
        xaxis=dict(showgrid=False, visible=False, range=[-plot_footprint_radius * 1.05, plot_footprint_radius * 1.05]),
        yaxis=dict(showgrid=False, visible=False, range=[-plot_footprint_radius * 1.05, plot_footprint_radius * 1.05]),
        zaxis=dict(showgrid=False, visible=False, range=[z_floor, sat_z + body_h + satellite_size * 0.05]),
        aspectratio=dict(x=1, y=1, z=1),
        camera=dict(
            eye=dict(x=camera_eye_x, y=camera_eye_y, z=camera_eye_z),
            center=dict(x=0, y=0, z=camera_center_z),
        ),
    )
    fig.update_layout(uirevision="animation")

    if output_path is not None:
        fig.write_html(f"{output_path}.html", auto_open=False, auto_play=False)

    return fig


#----------------------------------------------------------------------
# Visual Tests
#----------------------------------------------------------------------

if __name__ == "__main__":

    topography_2d = [0.1,0.3,0.8,0.5,0.2,0.4,0.9,1,1,0.7,0.3,0.1,0,0]
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