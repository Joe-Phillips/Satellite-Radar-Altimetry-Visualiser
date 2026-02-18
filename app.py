# ----------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------

import streamlit as st
import numpy as np
from animate_altimetry_waveform import animate_altimetry_waveform_2d, animate_altimetry_waveform_3d

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------

st.set_page_config(
    page_title="Satellite Radar Altimetry Visualiser",
    page_icon="🛰️",
    layout="wide",
)

# ----------------------------------------------------------------------
# CSS
# ----------------------------------------------------------------------

st.markdown(
    """
    <style>
        /* ── Global ── */
        [data-testid="stHeader"]            { display: none !important; }
        hr                                  { margin: 0.75rem 0 !important; }
        html, body,
        [data-testid="stAppViewContainer"],
        [data-testid="stApp"]               { background-color: white !important; color: black !important; }

        /* ── Sidebar ── */
        [data-testid="stSidebar"]           { min-width: 250px !important; max-width: 250px !important;
                                              width: 250px !important; background-color: #f0f4f8 !important; }
        [data-testid="stSidebarContent"]    { padding: 1rem 0.85rem !important; }
        [data-testid="stSidebar"],
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] div,
        [data-testid="stSidebar"] a,
        [data-testid="stSidebar"] li,
        [data-testid="stSidebar"] label     { color: #1a1a1a !important; }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3        { color: #1a3a5c !important; font-size: 0.95rem !important;
                                              text-transform: uppercase; letter-spacing: 0.06em;
                                              margin-top: 1.2rem !important; margin-bottom: 0.2rem !important; }
        [data-testid="stSidebar"] hr        { border-color: #c8d6e5 !important; }
        [data-testid="stSidebar"] [data-testid="stAlert"],
        [data-testid="stSidebar"] [data-baseweb="notification"] {
                                              background-color: white !important;
                                              border: 1px solid #dde6ef !important;
                                              border-radius: 8px !important; color: #222 !important; }
        [data-testid="stSidebar"] [data-testid="stAlert"] p,
        [data-testid="stSidebar"] [data-baseweb="notification"] p
                                            { color: #222 !important; font-size: 0.88rem !important; }
        [data-testid="stSidebar"] img       { display: block; margin: 0.5rem auto 0.25rem; max-width: 80% !important; }
        [data-testid="stSidebar"] p img     { display: inline !important; margin: 0 !important; max-width: unset !important; }

        /* ── Expanders ── */
        [data-testid="stExpander"]          { border: 1px solid #dde6ef !important; border-radius: 6px !important; }
        [data-testid="stExpander"] summary  { background-color: #f0f4f8 !important; border-radius: 6px !important; }
        [data-testid="stExpander"] summary p,
        [data-testid="stExpander"] summary span,
        [data-testid="stExpander"] summary svg
                                            { color: #1a3a5c !important; fill: #1a3a5c !important; }
        [data-testid="stExpander"] > div    { background-color: #fafbfc !important; }

        /* ── Widget labels ── */
        label,
        [data-testid="stWidgetLabel"],
        [data-testid="stWidgetLabel"] p,
        .stSlider p, .stCheckbox p,
        .stMarkdown p                       { color: black !important; }

        /* ── Dropdowns ── */
        .stSelectbox [data-baseweb="select"] input
                                            { cursor: pointer !important; caret-color: transparent !important; }
        .stSelectbox [data-baseweb="select"],
        .stSelectbox [data-baseweb="select"] *
                                            { cursor: pointer !important; }
        .stSelectbox div[data-baseweb="select"],
        .stSelectbox div[data-baseweb="select"] *,
        [data-baseweb="popover"] *,
        [data-baseweb="menu"] *             { color: black !important; background-color: white !important; }

        /* ── Text input ── */
        .stTextInput input                  { color: #111111 !important; background-color: #ffffff !important; }

        /* ── Plotly Play/Pause buttons — matching primaryColor ── */
        .updatemenu-item-rect               { fill: #4a7fb5 !important; stroke: #4a7fb5 !important; stroke-width: 0 !important; }
        .updatemenu-item-text               { fill: #ffffff !important; font-weight: 700 !important;
                                              font-size: 12px !important; letter-spacing: 0.05em !important; }
        .updatemenu-item-rect:hover         { fill: #3a6a99 !important; stroke: #3a6a99 !important; }
        g.updatemenu rect.bg                { fill: transparent !important; stroke: none !important; }

        /* ── Fixed-width action buttons (Run / New / Generate) ── */
        [data-testid="stButton"] button     { width: fit-content !important; min-width: unset !important;
                                              max-width: unset !important; white-space: nowrap !important; }

        /* ── Alert text ── */
        [data-testid="stAlert"] p,
        div[data-testid="stAlert"]          { color: #4a3800 !important; }

        /* ── Parameters heading ── */
        .params-heading                     { font-size: 2.1rem !important; font-weight: 700 !important;
                                              margin: 2rem 0 0.9rem 0 !important; color: #111111 !important;
                                              display: block !important; }

        /* ── Intro callout box ── */
        .callout-box                        { background-color: #f0f4f8; border-left: 4px solid #1a3a5c;
                                              padding: 0.85rem 1.1rem; border-radius: 0 6px 6px 0;
                                              margin: 0.75rem 0 1rem 0; color: #1a1a1a;
                                              font-size: 0.97rem; line-height: 1.6; }

    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Noise helpers — two-layer: rolling (low-freq) + gentle peaky (mid-freq)
# ----------------------------------------------------------------------

def _noise_1d(n, rng, roll_amp=0.05, peak_amp=0.012):
    x = np.linspace(0, 2 * np.pi, n)
    rolling = roll_amp * np.sin(float(rng.uniform(0.5, 1.5)) * x + float(rng.uniform(0, 2 * np.pi)))
    peaky   = peak_amp * np.sin(float(rng.uniform(2.5, 5.0)) * x + float(rng.uniform(0, 2 * np.pi)))
    return rolling + peaky

def _noise_2d(shape, rng, roll_amp=0.05, peak_amp=0.012):
    ny, nx = shape
    xx, yy = np.meshgrid(np.linspace(0, 2 * np.pi, nx), np.linspace(0, 2 * np.pi, ny))
    rolling = roll_amp * (
        np.sin(float(rng.uniform(0.5, 1.5)) * xx + float(rng.uniform(0, 2 * np.pi))) +
        np.sin(float(rng.uniform(0.5, 1.5)) * yy + float(rng.uniform(0, 2 * np.pi)))
    ) / 2
    peaky = peak_amp * (
        np.sin(float(rng.uniform(2.5, 5.0)) * xx + float(rng.uniform(0, 2 * np.pi))) +
        np.sin(float(rng.uniform(2.5, 5.0)) * yy + float(rng.uniform(0, 2 * np.pi)))
    ) / 2
    return rolling + peaky

# ----------------------------------------------------------------------
# 2D preset generators
# ----------------------------------------------------------------------

def _gen_flat_2d():
    rng = np.random.default_rng()
    n   = int(rng.integers(14, 20))
    return np.full(n, float(rng.uniform(0.1, 0.9))) + _noise_1d(n, rng, roll_amp=0.018, peak_amp=0.005)

def _gen_sloped_2d():
    rng    = np.random.default_rng()
    n      = int(rng.integers(12, 18))
    slope  = float(rng.uniform(0.35, 0.65)) * float(rng.choice([-1, 1]))
    center = float(rng.uniform(0.35, 0.65))
    return center + slope * (np.linspace(0, 1, n) - 0.5) + _noise_1d(n, rng, roll_amp=0.04, peak_amp=0.010)

def _gen_peaked_2d():
    rng    = np.random.default_rng()
    n      = int(rng.integers(14, 20))
    pos    = float(rng.uniform(0.2, 0.8))
    peak_h = float(rng.uniform(0.55, 0.85))
    base   = float(rng.uniform(0.08, 0.32))
    width  = float(rng.uniform(0.10, 0.38))
    x      = np.linspace(0, 1, n)
    return base + (peak_h - base) * np.exp(-0.5 * ((x - pos) / width) ** 2) + _noise_1d(n, rng, roll_amp=0.04, peak_amp=0.012)

def _gen_valley_2d():
    rng      = np.random.default_rng()
    n        = int(rng.integers(14, 20))
    pos      = float(rng.uniform(0.2, 0.8))
    valley_h = float(rng.uniform(0.08, 0.32))
    top      = float(rng.uniform(0.60, 0.88))
    width    = float(rng.uniform(0.10, 0.38))
    x        = np.linspace(0, 1, n)
    return top - (top - valley_h) * np.exp(-0.5 * ((x - pos) / width) ** 2) + _noise_1d(n, rng, roll_amp=0.04, peak_amp=0.010)

def _gen_rough_2d():
    rng   = np.random.default_rng()
    n     = int(rng.integers(16, 22))
    amp   = float(rng.uniform(0.15, 0.38))
    x     = np.linspace(0, 2 * np.pi * int(rng.integers(2, 5)), n)
    vals  = 0.5 + amp * np.sin(float(rng.uniform(0.6, 1.5)) * x + float(rng.uniform(0, 2 * np.pi)))
    vals += rng.normal(0, 0.03, n)
    return vals + _noise_1d(n, rng, roll_amp=0.04, peak_amp=0.015)

PRESET_2D_GENERATORS = {
    "Flat":   _gen_flat_2d,
    "Sloped": _gen_sloped_2d,
    "Peaked": _gen_peaked_2d,
    "Valley": _gen_valley_2d,
    "Rough":  _gen_rough_2d,
}

# ----------------------------------------------------------------------
# 3D preset generators
# ----------------------------------------------------------------------

def _gen_flat_3d():
    rng = np.random.default_rng()
    n   = 16
    return np.full((n, n), float(rng.uniform(0.1, 0.9))) + _noise_2d((n, n), rng, roll_amp=0.018, peak_amp=0.005)

def _gen_sloped_3d():
    rng    = np.random.default_rng()
    n      = 14
    slope  = float(rng.uniform(0.35, 0.65)) * float(rng.choice([-1, 1]))
    center = float(rng.uniform(0.35, 0.65))
    row    = center + slope * (np.linspace(0, 1, n) - 0.5)
    return np.tile(row, (n, 1)) + _noise_2d((n, n), rng, roll_amp=0.04, peak_amp=0.010)

def _gen_peaked_3d():
    rng    = np.random.default_rng()
    n      = 16
    x      = np.linspace(-1, 1, n)
    xx, yy = np.meshgrid(x, x)
    px, py = float(rng.uniform(-0.4, 0.4)), float(rng.uniform(-0.4, 0.4))
    peak_h = float(rng.uniform(0.55, 0.85))
    base   = float(rng.uniform(0.08, 0.28))
    width  = float(rng.uniform(0.3, 0.8))
    vals   = base + (peak_h - base) * np.exp(-((xx - px) ** 2 + (yy - py) ** 2) / (2 * width ** 2))
    return vals + _noise_2d((n, n), rng, roll_amp=0.04, peak_amp=0.012)

def _gen_valley_3d():
    rng      = np.random.default_rng()
    n        = 16
    x        = np.linspace(-1, 1, n)
    xx, yy   = np.meshgrid(x, x)
    vx, vy   = float(rng.uniform(-0.4, 0.4)), float(rng.uniform(-0.4, 0.4))
    valley_h = float(rng.uniform(0.08, 0.28))
    top      = float(rng.uniform(0.60, 0.88))
    width    = float(rng.uniform(0.3, 0.8))
    vals     = top - (top - valley_h) * np.exp(-((xx - vx) ** 2 + (yy - vy) ** 2) / (2 * width ** 2))
    return vals + _noise_2d((n, n), rng, roll_amp=0.04, peak_amp=0.010)

def _gen_sinusoidal_3d(n=14):
    rng    = np.random.default_rng()
    x      = np.linspace(0, 2 * np.pi, n)
    xx, yy = np.meshgrid(x, x)
    z      = np.zeros((n, n))
    for _ in range(int(rng.integers(2, 5))):
        z += np.sin(int(rng.integers(1, 4)) * xx + int(rng.integers(1, 4)) * yy + float(rng.uniform(0, 2 * np.pi)))
    z -= z.min()
    return z / z.max() if z.max() > 0 else np.full_like(z, 0.5)

def _gen_rough_3d():
    rng = np.random.default_rng()
    z   = _gen_sinusoidal_3d(n=14)
    z   = 0.1 + 0.8 * z + rng.normal(0, 0.15, z.shape)
    z  -= z.min()
    z   = 0.1 + 0.8 * z / z.max() if z.max() > 0 else np.full_like(z, 0.5)
    return z + _noise_2d(z.shape, rng, roll_amp=0.03, peak_amp=0.012)

def _apply_roughness(base, roughness):
    return np.clip(0.5 + (base - 0.5) * roughness, 0.0, 1.0)

PRESET_3D_GENERATORS = {
    "Flat":   _gen_flat_3d,
    "Sloped": _gen_sloped_3d,
    "Peaked": _gen_peaked_3d,
    "Valley": _gen_valley_3d,
    "Rough":  _gen_rough_3d,
}
# Full options list including special modes
PRESET_3D_OPTIONS = list(PRESET_3D_GENERATORS.keys()) + ["Randomise", "Use 2D profile"]

# ----------------------------------------------------------------------
# Session state defaults
# ----------------------------------------------------------------------

_DEFAULTS = {
    "srw_2d": True,  "rw_2d": (0.0, 1.0), "sray_2d": True,
    "nr_2d":  25,    "sp_2d": True,        "wn_2d":   0.01,
    "srw_3d": True,  "rw_3d": (0.0, 1.0), "sray_3d": True,
    "nr_3d":  50,    "sp_3d": True,        "wn_3d":   0.01,
    "roughness_3d":   0.5,
    "last_preset_2d": None,
    "last_preset_3d": None,
}
for _k, _v in _DEFAULTS.items():
    st.session_state.setdefault(_k, _v)

# ----------------------------------------------------------------------
# UI helpers
# ----------------------------------------------------------------------

def _btn_spacer():
    """Vertical spacer that aligns buttons with adjacent selectbox labels."""
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

def section_heading(emoji, text):
    st.markdown(
        f"<h2 style='margin-top:0.4rem;margin-bottom:0.2rem'>{emoji} {text}</h2>",
        unsafe_allow_html=True,
    )

def render_plot(dim, topo):
    """Render the 2D or 3D altimetry animation. dim must be '2d' or '3d'."""
    ss  = st.session_state
    fn  = animate_altimetry_waveform_2d if dim == "2d" else animate_altimetry_waveform_3d
    fig = fn(
        topo, output_path=None,
        num_rays_to_display=ss[f"nr_{dim}"],
        range_window_top=ss[f"rw_{dim}"][1],
        range_window_bottom=ss[f"rw_{dim}"][0],
        show_poca=ss[f"sp_{dim}"],
        show_range_window=ss[f"srw_{dim}"],
        show_rays=ss[f"sray_{dim}"],
        wf_noise_amplitude=ss[f"wn_{dim}"],
    )
    st.plotly_chart(fig, theme="streamlit", width="stretch", config=dict(displayModeBar=False))

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():

    # ------------------------------------------------------------------
    # Sidebar
    # ------------------------------------------------------------------

    st.title("🛰️ Satellite Radar Altimetry Visualiser")

    st.sidebar.title(":globe_with_meridians: About")
    st.sidebar.info(
        "Learn how satellites measure the height of Earth from space! This interactive "
        "tool combines clear explanations with hands-on simulators to show how radar pulses "
        "reflect from the surface, form waveforms, and reveal elevation."
    )
    st.sidebar.title(":email: Contact")
    st.sidebar.info(
        """
    Made by **Dr Joe Phillips**

    [![GitHub](https://badgen.net/badge/icon/GitHub/green?icon=github&label)](https://github.com/Joe-Phillips) [![LinkedIn](https://badgen.net/badge/icon/linkedin/blue?icon=linkedin&label)](https://www.linkedin.com/in/joe-b-phillips/) j.phillips5@lancaster.ac.uk

    Special thanks to **Dom Hardy**
    d.j.hardy@lancaster.ac.uk
        """
    )
    st.sidebar.image("lancs_logo.png")
    st.sidebar.image("cpom_logo.png")

    # ------------------------------------------------------------------
    # Introduction
    # ------------------------------------------------------------------

    st.markdown("""<hr style="border: 1px solid black;">""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    section_heading("📡", "What is satellite radar altimetry?")

    st.markdown(
        "There are currently over *13,000* satellites in space, each performing a huge variety of different "
        "jobs. Some of these, such as the James Webb telescope, take detailed pictures of outer space. Others, "
        "such as Sentinel-2, orbit around the planet, taking high quality pictures of the surface of the "
        "Earth. And some, such as CryoSat-2, use special instruments to accurately measure the height of "
        "the world below them. "
        "The study of the measurements these satellites take is known as **remote sensing**. Here, we "
        "cover how satellites like CryoSat-2 work, and how they are able to give scientists a "
        "detailed picture of the height of the planet everywhere."
    )

    st.markdown(
        "Satellites carrying radar altimeters measure the height of the ground below them by firing a "
        "**pulse** (🔴) of energy at the ground and seeing how long it takes to come back. This is kind "
        "of like how a bat navigates in the dark using echo-location, but on a much larger scale! "
        "The energy travels down to the ground and spreads out in a wide circle, with the strongest signal "
        "directly beneath the satellite at a point called **nadir**. This area the satellite illuminates is "
        "called its footprint. To picture this, imagine pointing a torch at a wall. In fact, both "
        "altimeters and torches are very similar in that they both emit light (energy) - just the "
        "altimeter uses light at wavelengths too long for us to see (radio), and also times how long "
        "the light takes to bounce back! This type of light allows the satellites to see through the "
        "atmosphere and even through weather such as clouds and rain, which other satellites carrying "
        "cameras cannot do. "
        "Since these echoes (radio waves) travel at the speed of light (*299,792,458* meters per second!), "
        "the satellite can work out how far away the ground is below it. Even tiny differences in these "
        "timings can reveal detailed information about the shape of the surface below. By doing this "
        "continuously, tracing a path all across the globe (called its **track**), scientists can build "
        "up precise maps of surface height *everywhere*. This is useful for many things such as "
        "tracking sea level rise and figuring out how much ice sheets are melting."
    )

    st.markdown(
        "As these echoes return to the satellite, they are recorded as a **waveform** (🟦). This is just "
        "a graph of how many echoes (the power) the satellite receives back over time. "
        "The shape of this waveform carries lots of useful information about the ground below. "
        "The first steep rise in the waveform, which is called the **leading edge**, corresponds to the "
        "highest point on the surface, closest to the satellite. Because this point is closest, any "
        "echoes that bounce off it return back to the satellite first. This point is known as the "
        "**Point of Closest Approach**, or **POCA** (⭐). "
        "By figuring out the time between when the satellite first sends a pulse of energy, to when the echo first "
        "returns on the leading edge of the waveform, scientists can figure out the distance between the POCA "
        "on the surface and the satellite. By knowing how high up the satellite is (over *700,000* meters!), they can "
        "then work out the height of the POCA!"
    )

    st.markdown(
        "Once the satellite fires a burst of energy, it only listens for the returning echo over a small "
        "period of time called the **range window** (🟩). This listening window is both a window in time "
        "(on the waveform) and also a window in physical space (below the satellite). "
        "Because the energy the satellite fires and listens for travels at a constant speed, the distance "
        "travelled by the energy over a given bit of time is always the same. Therefore, these windows in space and time "
        "are actually the same thing! The less time the satellite listens for, the less of the surface it might "
        "capture. For very mountainous areas, which might have a large difference in their lowest and "
        "highest elevation, a small range window might mean that only the top or the bottom of the "
        "surface below is captured. If the window is placed in the wrong position, echoes can be missed "
        "entirely. This is called **losing track**. "
        "Getting this right - knowing when/where to listen for echoes - can be tough, and even the "
        "complicated algorithms onboard sometimes struggle."
    )

    st.markdown(
        """
        <div class="callout-box">
        <b><i>Phew</i></b> - that was quite a lot of information. And we even simplified it a lot (<i>sorry scientists</i>)!
        So to make it all easier to understand, we put together two interactive simulators below that
        will hopefully help it click.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    section_heading("🌍", "But what does this actually look like?")

    st.markdown(
        """
        We put together two new simulators to help you visualise how satellite radar altimetry works:

        ##### ▸ **2D simulator**

        This is a simplified two-dimensional model, where we only consider the surface elevation and its
        position left and right. This is actually similar to how many modern satellite radar altimeters work,
        using a process called **Synthetic Aperture Radar** (SAR), which changes the satellite's footprint on the
        surface into thin, wide strips (almost 2D). This lets the satellite capture much more information along
        its track, rather than having to wait for echoes to return before emitting more energy.

        ##### ▸ **3D simulator**

        This is a fully three-dimensional version of the model. Here, the emitted pulses spread out across a
        full circular footprint. This is similar to how altimeters work without SAR processing, and is generally
        referred to as **Low Resolution Mode** (LRM). Instruments such as these can be found onboard older
        satellites and current satellites operating over less complicated surfaces such as parts of the ocean.

        <br>

        **Both of these simulators are approximations!** Although they don't replicate the real instruments
        perfectly, they capture the key ideas. In general, the 2D (sort-of-SAR) waveforms look a bit more
        realistic than what we would expect from the 3D (sort-of-LRM).

        To get started, feel free to select an option from the surface shape menu below and press **▶ Play**!
        """,
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------
    # 2D Simulator
    # ------------------------------------------------------------------

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    section_heading("📊", "2D Simulator")
    st.markdown(
        "Choose a surface shape below, or select **Custom** and enter your own numbers "
        "representing height from left to right. Press **▶ Play** to start!"
    )

    sel_col_2d, btn_col_2d, _ = st.columns([1, 0.18, 1.82])
    with sel_col_2d:
        preset_2d = st.selectbox("Surface shape:", options=list(PRESET_2D_GENERATORS.keys()) + ["Custom"], key="preset_2d")
    with btn_col_2d:
        _btn_spacer()
        if preset_2d == "Custom":
            regen_2d = False
            st.button("▶ Run", key="run_btn_2d", type="primary", help="Run the simulator with your custom surface")
        else:
            regen_2d = st.button("↺ New", key="regen_btn_2d", type="primary", help="Regenerate this surface")

    if (preset_2d != st.session_state.get("last_preset_2d") or regen_2d) and preset_2d in PRESET_2D_GENERATORS:
        st.session_state["last_preset_2d"] = preset_2d
        st.session_state["topo_2d_cached"] = PRESET_2D_GENERATORS[preset_2d]()

    if preset_2d == "Custom":
        custom_input = st.text_input(
            "Surface heights (space or comma separated):",
            placeholder="e.g. 3 8 2 9 5",
            key="topo_2d_custom",
        )
        if custom_input:
            try:
                parsed = np.array(custom_input.replace(",", " ").split(), dtype=float)
                if len(parsed) > 1 and (parsed >= 0).all():
                    mn, mx        = parsed.min(), parsed.max()
                    topo_2d       = parsed if mx == mn else (parsed - mn) / (mx - mn)
                    topo_2d_valid = True
                else:
                    st.warning("Please enter at least two non-negative numbers.")
                    topo_2d, topo_2d_valid = np.array([0.5, 0.5]), False
            except Exception:
                st.warning("Invalid input — please enter a list of numbers.")
                topo_2d, topo_2d_valid = np.array([0.5, 0.5]), False
        else:
            topo_2d, topo_2d_valid = np.array([0.5, 0.5]), True
    else:
        topo_2d       = st.session_state.get("topo_2d_cached", PRESET_2D_GENERATORS[preset_2d]())
        topo_2d_valid = True

    if topo_2d_valid:
        try:
            render_plot("2d", topo_2d)
        except Exception:
            st.warning("Something went wrong generating the 2D plot. Check your surface input.")
    
    st.checkbox("Show range window", key="srw_2d")
    st.slider("Range window position:", 0.0, 1.0, key="rw_2d", step=0.01)
    st.checkbox("Show rays", key="sray_2d")
    st.slider("Number of rays to display:", 5, 75, key="nr_2d", step=1)
    st.checkbox("Show POCA", key="sp_2d")
    st.slider("Amount of noise:", 0.0, 0.1, key="wn_2d", step=0.01)

    # ------------------------------------------------------------------
    # 3D Simulator
    # ------------------------------------------------------------------

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    section_heading("📊", "3D Simulator")
    st.markdown(
        "Choose a surface shape below, and press **▶ Play** to get started!"
    )
    st.caption(
        "**Navigating the 3D view:** left-click and drag to rotate, right-click and drag to pan, "
        "scroll to zoom. Pressing Play will reset the view - pause first if you want to look around freely."
    )

    sel_col_3d, btn_col_3d, _ = st.columns([1, 0.18, 1.82])
    with sel_col_3d:
        preset_3d = st.selectbox("Surface shape:", options=PRESET_3D_OPTIONS, key="preset_3d")
    with btn_col_3d:
        _btn_spacer()
        if preset_3d in PRESET_3D_GENERATORS:
            regen_3d = st.button("↺ New", key="regen_btn_3d", type="primary", help="Regenerate this surface")
        else:
            regen_3d = False

    if (preset_3d != st.session_state.get("last_preset_3d") or regen_3d) and preset_3d in PRESET_3D_GENERATORS:
        st.session_state["last_preset_3d"] = preset_3d
        st.session_state["topo_3d_cached"] = PRESET_3D_GENERATORS[preset_3d]()

    if preset_3d == "Use 2D profile":
        st.info(
            "**Using 2D profile:** The surface profile from the 2D simulator above is repeated across the new "
            "direction to fill the circular 3D footprint. Change the shape or preset in the 2D "
            "simulator and the 3D view will update to match.",
            icon="ℹ️",
        )

    if preset_3d == "Randomise":
        rough_col, rand_btn_col = st.columns([2, 1])
        with rough_col:
            st.slider("Roughness:", 0.0, 1.0, key="roughness_3d", step=0.05,
                      help="0 = perfectly flat, 1 = full peaks and troughs. Drag to reshape; press **Generate** for a new surface.")
        with rand_btn_col:
            _btn_spacer()
            if st.button("🎲 Generate", key="rand_btn_3d", type="primary"):
                st.session_state["random_topo_3d_base"] = _gen_sinusoidal_3d()

    if preset_3d == "Use 2D profile":
        if topo_2d_valid:
            n   = max(len(topo_2d), 8)
            row = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(topo_2d)), topo_2d)
            topo_3d, topo_3d_valid = np.tile(row, (n, 1)), True
        else:
            st.warning("Enter a valid surface profile in the 2D simulator above first.")
            topo_3d, topo_3d_valid = None, False
    elif preset_3d == "Randomise":
        if "random_topo_3d_base" not in st.session_state:
            st.session_state["random_topo_3d_base"] = _gen_sinusoidal_3d()
        topo_3d       = _apply_roughness(st.session_state["random_topo_3d_base"], st.session_state["roughness_3d"])
        topo_3d_valid = True
    else:
        topo_3d       = st.session_state.get("topo_3d_cached", PRESET_3D_GENERATORS[preset_3d]())
        topo_3d_valid = True

    if topo_3d_valid:
        try:
            render_plot("3d", topo_3d)
        except Exception:
            st.warning("Something went wrong generating the 3D plot.")

    st.checkbox("Show range window", key="srw_3d")
    st.slider("Range window position:", 0.0, 1.0, key="rw_3d", step=0.01)
    st.checkbox("Show rays", key="sray_3d")
    st.slider("Number of rays to display:", 5, 100, key="nr_3d", step=1)
    st.checkbox("Show POCA", key="sp_3d")
    st.slider("Amount of noise:", 0.0, 0.1, key="wn_3d", step=0.01)

    # ------------------------------------------------------------------
    # How does altimetry actually work?
    # ------------------------------------------------------------------

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    section_heading("📡", "How does altimetry actually work?")
    st.markdown(
        """
        We covered the basics above, but we simplified a lot. There is actually quite a lot more nuance in practice! Here
        we go into a bit more detail.
        """
    )

    with st.expander("Technical detail"):
        st.markdown(
            """
            <div style="background-color:#fff8e1; border-left:4px solid #f0a500; padding:10px 14px; border-radius:4px; color:#4a3800; margin-bottom:1em;">
            ⚠️ This section goes into more technical detail - so readers beware! It is also intended only as a brief overview.
            </div>

            **The waveform**

            Over a flat surface, the expected waveform shape is well described by the **Brown model** (Brown, 1977),
            which treats the surface as a collection of small independent scatterers. The model predicts a rapid
            power rise at the leading edge - as the pulse wavefront first intersects the surface - followed by a
            peak near the POCA return, and then a gradual exponential decay as echoes return from progressively
            greater distances - those from the edges of the footprint travel further and are received with
            diminishing antenna gain, while the trailing edge also carries weaker returns from various secondary
            effects.

            **Footprints and measurement resolution**

            Three different concepts of 'footprint' matter in altimetry, and they are easy to confuse.

            The **beam-limited footprint** is the total area on the ground illuminated by the antenna - defined by
            the antenna's radiation pattern at its −3 dB half-power point. For a small spaceborne antenna this
            can span tens of kilometres in diameter. Because the antenna aperture is physically small relative to
            the radar wavelength, the beam cannot be focused more tightly without a much larger antenna - a
            practical constraint for satellite instruments. Using this entire illuminated area as the measurement
            resolution would make it impossible to attribute any given reflection to a specific surface location,
            so a finer effective resolution is needed.

            To achieve this, the transmitted pulse is shortened so that only returns within a narrow time window
            contribute to each waveform bin. This defines the **pulse-limited footprint** - roughly 1-2 km in
            diameter over the surface, centred on POCA. Think of each waveform bin as representing reflections
            from a specific, small area of the surface: the shorter the pulse, the smaller that area, and the
            finer the effective resolution. Returns from surfaces beyond this pulse-limited region but still
            within the beam-limited area form concentric annuli at increasing ranges, contributing to later
            parts of the waveform. However, because the altimeter must wait for each echo to return before
            transmitting the next, the along-track sampling rate is limited to a low pulse repetition frequency
            (PRF, typically ~2 kHz), and kilometre-scale along-track resolution is typical. This conventional
            approach is called **Low Resolution Mode (LRM)**, and has been the standard for missions including
            ERS-1/2, Envisat, and CryoSat-2 over ice sheet interiors. This is what the 3D simulator approximates.

            SAR altimetry overcomes this along-track limitation by transmitting bursts of pulses at a much
            higher PRF (~18 kHz for CryoSat-2). By coherently processing overlapping returns using Doppler
            filtering - which exploits frequency shifts induced by satellite motion - SAR synthesises a longer
            effective antenna aperture along-track, defining a third concept: the **Doppler-limited footprint**.
            For CryoSat-2 this compresses the along-track footprint to ~380 m, dramatically improving
            along-track resolution relative to LRM, while the across-track footprint remains beam-limited at
            ~15 km. Stacking multiple looks of the same location also reduces radar speckle noise - this is
            called **multilooking**. The across-track geometry is therefore dominant in SAR, making the 2D
            (across-track slice) simulator a reasonable analogy.

            **Retracking and slope correction**

            Once a waveform is captured, extracting a usable elevation requires two key steps. The altimeter
            does not measure absolute range directly - instead, each waveform is recorded relative to a
            **reference range**: a known distance used to position the centre of the range window, computed
            onboard from predicted orbit and surface elevation and stored alongside the waveform in the
            Level-1 data products (the minimally-processed instrument output). The actual range to the surface
            is then determined by finding where the leading edge falls relative to this reference and adding
            the corresponding offset.

            **Retracking** performs this step: it fits or thresholds the waveform's leading edge to identify
            the precise range bin corresponding to the first surface return, which - combined with the
            reference range - gives the total range from satellite to surface. Various algorithms exist,
            from physical models fitting theoretical waveform shapes to empirical approaches based on waveform
            geometry. **Slope correction** then determines *where on the surface* that range should be
            attributed to, since POCA is rarely directly below the satellite over sloping terrain, and
            typically relies on an auxiliary digital elevation model. Together, these steps reduce the full
            waveform to a single point elevation estimate. Slope correction is the dominant source of error
            in ice sheet altimetry, with uncertainties commonly reaching tens of metres in elevation and
            kilometre-scale horizontal offsets over complex terrain.

            **Surface tracking**

            Before any of this can happen, the satellite's onboard system must position the range window
            correctly - a process called **surface tracking**. Two broad approaches exist: *closed-loop
            tracking*, where the window is continuously updated based on the most recently received echo,
            and *open-loop tracking*, where it is driven by a pre-loaded elevation model rather than live
            feedback. Open-loop tracking is more robust over complex terrain, where steep or rapidly changing
            topography means the previous echo is often a poor predictor of the next. Over flat, predictable
            terrain closed-loop tracking works well, but over rough or steeply sloping ground either approach
            can struggle, causing *loss of track* where echoes fall outside the range window and no measurement
            is recorded. For this reason, data coverage commonly degrades over complex terrain such as the
            margins of ice sheets.

            **Interferometric SAR (SARIn) mode**

            CryoSat-2 carries a second antenna, offset in the across-track direction. By comparing the phase
            of the signal received at each antenna - a technique called **interferometry** - it is possible to
            measure the angle of arrival of the echo, resolving *where across-track* the POCA return came
            from and bypassing slope correction entirely. Taking this further, *swath processing* uses phase
            information throughout the waveform, not just at the leading edge, to recover elevations from
            across the whole illuminated footprint, producing dense grids of elevation measurements from a
            single pass. Neither simulator here uses interferometric information; both are based on power
            waveforms only, which is the more general and challenging case faced by SAR-only missions such
            as Sentinel-3.
            """,
            unsafe_allow_html=True,
        )

    # ------------------------------------------------------------------
    # How does the simulator work?
    # ------------------------------------------------------------------

    st.markdown("<br>", unsafe_allow_html=True)
    section_heading("🔧", "How does the simulator work?")
    st.markdown(
        """
        Each simulator works by tracing imaginary lines (rays) from the satellite down to the surface. For
        each ray, it finds where the line hits the ground and calculates how long the signal would take to
        travel there and back. That travel time is mapped to a position in the waveform. The radar signal is
        strongest directly below the satellite and naturally fades towards the edges, so rays pointing more
        steeply downward contribute more to the waveform.

        In the **2D simulator**, rays fan out left and right across a vertical slice of the surface. In the
        **3D simulator**, they spread out in all directions across a circular area.

        Below, we cover this in a bit more detail.
        """
    )

    with st.expander("Technical detail"):
        st.markdown(
            """
            <div style="background-color:#fff8e1; border-left:4px solid #f0a500; padding:10px 14px; border-radius:4px; color:#4a3800; margin-bottom:1em;">
            ⚠️ This section goes into more technical detail - so readers beware! It is also intended only as a brief overview.
            </div>

            **The nitty gritty**

            Both simulators use a geometric ray-casting approach. A virtual satellite is placed at a fixed
            altitude and rays are cast downward across the footprint. For each ray, the intersection with
            the topography surface is found numerically. The slant range to that intersection is then
            compared to the total range window extent to assign a waveform bin:
            ```
            bin = round((1 - dist_from_range_window_bottom / range_window_size) * num_bins)
            ```

            Contributions are weighted by the antenna gain pattern, which describes how the antenna's
            emission power drops with angle from nadir. This uses the two-dimensional antenna gain equation
            from Wingham et al. (2006):
            ```
            G(θ, φ) = exp(-θ² x (cos²φ / γ₁² + sin²φ / γ₂²))
            ```

            where θ is the off-nadir angle, φ is the azimuth angle, and γ₁ = 0.0133 rad,
            γ₂ = 0.0148 rad are the CryoSat-2 3 dB beamwidths in the along- and across-track directions.

            In both simulators, instrument parameters - satellite altitude (~717 km), across-track
            footprint size (~15 km), and range window height (~240 m) - are taken from CryoSat-2 in SARIn
            mode. For the 2D case, the number of rays (512) and waveform bins (128) were tuned against
            real waveforms. This has yet to be done for the 3D case, where we currently use 2048 rays and
            128 bins. For reference, a real CryoSat-2 SARIn waveform has 1024 bins. Matching this in
            practice can be done using interpolation.

            Currently, the simulators are unable to account for surface elevations that fall outside the
            range window. To handle this, input topography is rescaled from [0, 1] to [0.2, 0.8] before
            being passed to the simulator. The resulting waveform is then embedded into a longer
            zero-padded waveform spanning the full animation, with the range window parameters adjusted
            to match.

            **2D simulator (SAR-analogous)**

            Rays are cast across a 1D topographic profile at uniform across-track intervals, with the range
            window bottom following a curved arc - the locus of points at equal slant range from the
            satellite. This collapses the along-track dimension and is loosely analogous to SAR altimetry
            as used by CryoSat-2 and Sentinel-3. In SAR mode, coherent processing of overlapping pulse
            returns compresses the along-track footprint to around 380 m (for CryoSat-2), making the
            across-track geometry dominant (~15 km beam-limited).

            **3D simulator (LRM-analogous)**

            Rays are cast across a 2D circular footprint (radius ~7.5 km) arranged on a hexagonal grid
            with uniform spacing, with the number of rays dynamically adjusted to maintain equidistance.
            This is loosely analogous to Low Resolution Mode (LRM) altimetry - the conventional
            pulse-limited mode used by missions such as ERS-1/2 and Envisat, and by CryoSat-2 over ice
            sheet interiors. LRM waveforms integrate returns from all azimuth directions simultaneously,
            producing the characteristic shape over flat surfaces commonly modelled using the Brown model
            (Brown, 1977).

            **Limitations**

            - *Trailing edge*: The most visible departure from real waveforms. In the simulator, each ray
            contributes to a single waveform bin, so power drops off sharply after the peak. In reality,
            the trailing edge decays more gradually. This artefact is most pronounced in the 3D case over
            flat surfaces, where the departure from the Brown model trailing edge is most obvious.
            - *Scattering*: The model assumes spatially uniform reflectivity. In reality, backscatter varies
            with surface type, and over snow and ice, volume scattering and signal penetration into the
            snowpack can shift the apparent surface return, affecting both waveform shape and retrieved
            elevation.
            - *No instrument effects*: Pulse compression, range migration, multilooking, and thermal noise
            are not modelled. The noise visible on the waveform is uniform random noise added purely for
            visual clarity.
            - *Single intersection per ray*: Only the first surface hit per ray is recorded. Multi-path
            returns and radar layover - where steep terrain causes reflections from different locations to
            arrive at the same range - are not handled. Angle-dependent reflectivity (i.e. variation in
            backscatter with incidence angle) is also not modelled; this was tested during development but
            found to have negligible effect on the simulated waveform shape and was therefore omitted.
            """,
            unsafe_allow_html=True,
        )


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    main()