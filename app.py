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
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# CSS
# ----------------------------------------------------------------------

st.markdown(
    """
    <style>
        /* Global */
        [data-testid="stHeader"],
        button[kind="header"],
        [data-testid="collapsedControl"],
        [data-testid="stSidebarCollapseButton"]     { display: none !important; }

        [data-testid="stSidebar"][aria-expanded="false"] {
                                                      display: block !important;
                                                      transform: none !important;
                                                      min-width: 250px !important;
                                                      width: 250px !important; }

        hr                                          { margin: 0.75rem 0 !important; }

        html, body,
        [data-testid="stAppViewContainer"],
        [data-testid="stApp"]                       { background-color: white !important;
                                                      color: black !important; }

        /* Sidebar */
        [data-testid="stSidebar"]                   { min-width: 250px !important;
                                                      max-width: 250px !important;
                                                      width: 250px !important;
                                                      background-color: #f0f4f8 !important;
                                                      overflow: hidden !important; }

        [data-testid="stSidebarContent"]            { padding: 1rem 0.85rem !important; }

        [data-testid="stSidebar"],
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] div,
        [data-testid="stSidebar"] a,
        [data-testid="stSidebar"] li,
        [data-testid="stSidebar"] label             { color: #1a1a1a !important; }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3                { color: #1a3a5c !important;
                                                      font-size: 0.95rem !important;
                                                      text-transform: uppercase;
                                                      letter-spacing: 0.06em;
                                                      margin-top: 1.2rem !important;
                                                      margin-bottom: 0.2rem !important; }

        [data-testid="stSidebar"] hr                { border-color: #c8d6e5 !important; }

        [data-testid="stSidebar"] [data-testid="stAlert"],
        [data-testid="stSidebar"] [data-baseweb="notification"] {
                                                      background-color: white !important;
                                                      border: 1px solid #dde6ef !important;
                                                      border-radius: 8px !important;
                                                      color: #222 !important; }

        [data-testid="stSidebar"] [data-testid="stAlert"] p,
        [data-testid="stSidebar"] [data-baseweb="notification"] p
                                                    { color: #222 !important;
                                                      font-size: 0.88rem !important; }

        [data-testid="stSidebar"] img               { display: block;
                                                      margin: 0.5rem auto 0.25rem;
                                                      max-width: 80% !important; }

        [data-testid="stSidebar"] p img             { display: inline !important;
                                                      margin: 0 !important;
                                                      max-width: unset !important; }

        /* Sidebar scrollbar hidden */
        [data-testid="stSidebar"] > div:first-child { overflow-y: scroll !important;
                                                      scrollbar-width: none !important;
                                                      -ms-overflow-style: none !important; }

        [data-testid="stSidebar"] > div:first-child::-webkit-scrollbar
                                                    { display: none !important; }

        /* Expanders */
        [data-testid="stExpander"]                  { border: 1px solid #dde6ef !important;
                                                      border-radius: 6px !important; }

        [data-testid="stExpander"] summary          { background-color: #f0f4f8 !important;
                                                      border-radius: 6px !important; }

        [data-testid="stExpander"] summary p,
        [data-testid="stExpander"] summary span,
        [data-testid="stExpander"] summary svg      { color: #1a3a5c !important;
                                                      fill: #1a3a5c !important; }

        [data-testid="stExpander"] > div            { background-color: #fafbfc !important; }

        /* Widget labels */
        label,
        [data-testid="stWidgetLabel"],
        [data-testid="stWidgetLabel"] p,
        .stSlider p, .stCheckbox p,
        .stMarkdown p                               { color: black !important; }

        /* Dropdowns */
        .stSelectbox [data-baseweb="select"] input  { cursor: pointer !important;
                                                      caret-color: transparent !important; }

        .stSelectbox [data-baseweb="select"],
        .stSelectbox [data-baseweb="select"] *,
        .stSelectbox div[data-baseweb="select"],
        .stSelectbox div[data-baseweb="select"] *,
        [data-baseweb="popover"] *,
        [data-baseweb="menu"] *                     { cursor: pointer !important;
                                                      color: black !important;
                                                      background-color: white !important; }

        /* Text input */
        .stTextInput input                          { color: #111111 !important;
                                                      background-color: #ffffff !important; }

        /* Plotly Play/Pause buttons */
        .updatemenu-item-rect                       { fill: #4a7fb5 !important;
                                                      stroke: #4a7fb5 !important;
                                                      stroke-width: 0 !important; }

        .updatemenu-item-text                       { fill: #ffffff !important;
                                                      font-weight: 700 !important;
                                                      font-size: 12px !important;
                                                      letter-spacing: 0.05em !important; }

        .updatemenu-item-rect:hover                 { fill: #3a6a99 !important;
                                                      stroke: #3a6a99 !important; }

        g.updatemenu rect.bg                        { fill: transparent !important;
                                                      stroke: none !important; }

        /* Action buttons */
        [data-testid="stButton"] button             { width: fit-content !important;
                                                      min-width: unset !important;
                                                      max-width: unset !important;
                                                      white-space: nowrap !important; }

        /* Alert text */
        [data-testid="stAlert"] p,
        div[data-testid="stAlert"]                  { color: #4a3800 !important; }

        /* Intro callout box */
        .callout-box                                { background-color: #f0f4f8;
                                                      border-left: 4px solid #1a3a5c;
                                                      padding: 0.85rem 1.1rem;
                                                      border-radius: 0 6px 6px 0;
                                                      margin: 0.75rem 0 1rem 0;
                                                      color: #1a1a1a;
                                                      font-size: 0.97rem;
                                                      line-height: 1.6; }

        /* Legend box */
        .legend-box                                 { background-color: #f8f9fb;
                                                      border: 1px solid #dde6ef;
                                                      border-radius: 6px;
                                                      padding: 0.75rem 1.1rem;
                                                      margin: 0.75rem 0 1rem 0;
                                                      font-size: 0.93rem;
                                                      line-height: 2.0; }

        /* Diagram placeholder */
        .diagram-placeholder                        { background-color: #f0f4f8;
                                                      border: 2px dashed #b0bec5;
                                                      border-radius: 6px;
                                                      padding: 0.9rem 1.1rem;
                                                      margin: 0.9rem 0;
                                                      color: #546e7a;
                                                      font-size: 0.92rem;
                                                      font-style: italic;
                                                      line-height: 1.5; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Noise helpers - two-layer: rolling (low-freq) + gentle peaky (mid-freq)
# ----------------------------------------------------------------------

def _noise_1d(n, rng, roll_amp=0.05, peak_amp=0.012):
    x       = np.linspace(0, 2 * np.pi, n)
    rolling = roll_amp * np.sin(float(rng.uniform(0.5, 1.5)) * x + float(rng.uniform(0, 2 * np.pi)))
    peaky   = peak_amp * np.sin(float(rng.uniform(2.5, 5.0)) * x + float(rng.uniform(0, 2 * np.pi)))
    return rolling + peaky


def _noise_2d(shape, rng, roll_amp=0.05, peak_amp=0.012):
    ny, nx  = shape
    xx, yy  = np.meshgrid(np.linspace(0, 2 * np.pi, nx), np.linspace(0, 2 * np.pi, ny))
    rolling = roll_amp * (
        np.sin(float(rng.uniform(0.5, 1.5)) * xx + float(rng.uniform(0, 2 * np.pi))) +
        np.sin(float(rng.uniform(0.5, 1.5)) * yy + float(rng.uniform(0, 2 * np.pi)))
    ) / 2
    peaky   = peak_amp * (
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
    rng  = np.random.default_rng()
    n    = int(rng.integers(16, 22))
    amp  = float(rng.uniform(0.15, 0.38))
    x    = np.linspace(0, 2 * np.pi * int(rng.integers(2, 5)), n)
    vals = 0.5 + amp * np.sin(float(rng.uniform(0.6, 1.5)) * x + float(rng.uniform(0, 2 * np.pi)))
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
PRESET_3D_OPTIONS = list(PRESET_3D_GENERATORS.keys()) + ["Randomise", "Use 2D profile"]

# ----------------------------------------------------------------------
# Session state defaults
# ----------------------------------------------------------------------

_DEFAULTS = {
    # 2D simulator
    "srw_2d":  True,  "rw_2d":  (0.0, 1.0), "sray_2d": True,
    "nr_2d":   25,    "sp_2d":  True,        "sn_2d":   True,
    "sle_2d":  True,  "wn_2d":  0.01,
    # 3D simulator
    "srw_3d":  True,  "rw_3d":  (0.0, 1.0), "sray_3d": True,
    "nr_3d":   50,    "sp_3d":  True,        "sn_3d":   True,
    "sle_3d":  True,  "wn_3d":  0.01,
    # 3D special modes
    "roughness_3d":   0.5,
    # Cache keys
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
    ss = st.session_state
    fn = animate_altimetry_waveform_2d if dim == "2d" else animate_altimetry_waveform_3d
    fig = fn(
        topo, output_path=None,
        num_rays_to_display = ss[f"nr_{dim}"],
        range_window_top    = ss[f"rw_{dim}"][1],
        range_window_bottom = ss[f"rw_{dim}"][0],
        show_poca           = ss[f"sp_{dim}"],
        show_range_window   = ss[f"srw_{dim}"],
        show_rays           = ss[f"sray_{dim}"],
        wf_noise_amplitude  = ss[f"wn_{dim}"],
        show_nadir          = ss[f"sn_{dim}"],
        show_leading_edge   = ss[f"sle_{dim}"],
    )
    st.plotly_chart(fig, theme="streamlit", width="stretch", config=dict(displayModeBar=False))


def _swatch(colour):
    return (
        f"<span style='display:inline-block;width:13px;height:13px;"
        f"background:{colour};border-radius:2px;"
        f"border:0.5px solid black;"
        f"vertical-align:middle;margin-right:5px;'></span>"
    )


def _dash_swatch(colour):
    return (
        f"<span style='display:inline-block;width:24px;height:3px;"
        f"border-top:2.5px dashed {colour};vertical-align:middle;"
        f"margin-right:5px;'></span>"
    )


def _diamond_swatch(colour):
    return (
        f"<span style='display:inline-block;width:12px;height:12px;"
        f"background:{colour};transform:rotate(45deg);"
        f"border:0.5px solid black;"
        f"margin-right:6px;vertical-align:middle;'></span>"
    )


def _star_swatch(colour):
    return (
        f"<span style='color:{colour};font-size:14px;"
        f"-webkit-text-stroke:0.4px black;"
        f"text-shadow:0 0 0 black;"
        f"margin-right:6px;vertical-align:middle;'>★</span>"
    )


def render_legend():
    """Render the consolidated marker/symbol legend."""
    # Colours must match the palette defined in animate_altimetry_waveform.py
    c_pulse = "rgb(255, 0, 0)"
    c_wf    = "#87ceeb"
    c_le    = "rgb(0, 175, 155)"
    c_poca  = "rgb(255,237,41)"
    c_nadir = "rgb(255,29,206)"
    c_rw    = "rgba(0, 200, 0, 0.7)"

    st.markdown(
        f"""
        <div class="legend-box">
        <strong>Symbol reference</strong><br>
        {_swatch(c_pulse)}<strong>Pulse / rays</strong> - radar energy emitted by the satellite<br>
        {_swatch(c_wf)}<strong>Waveform</strong> - recorded echo power over time<br>
        {_dash_swatch(c_le)}<strong>Leading edge bounds</strong> - start and end of the waveform's leading edge<br>
        {_star_swatch(c_poca)}<strong>POCA</strong> - Point of Closest Approach; the first surface return, coinciding with the leading-edge<br>
        {_diamond_swatch(c_nadir)}<strong>Nadir</strong> - the point directly below the satellite<br>
        {_swatch(c_rw)}<strong>Range window</strong> - the satellite's listening window in time and space
        </div>
        """,
        unsafe_allow_html=True,
    )

# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------

def render_sidebar():
    st.sidebar.title(":globe_with_meridians: About")
    st.sidebar.info(
        """
        Learn how satellites measure the height of Earth from space! This interactive 
        tool combines clear explanations with hands-on simulators to show how radar pulses are emitted from satellites,
        reflect from the surface, form waveforms, and reveal elevation across the globe.

        **This website is best viewed on a desktop or laptop!**
        """
    )
    st.sidebar.title(":email: Contact")
    st.sidebar.info(
        """
    Made by **Joe Phillips**

    [![GitHub](https://badgen.net/badge/icon/GitHub/green?icon=github&label)](https://github.com/Joe-Phillips) [![LinkedIn](https://badgen.net/badge/icon/linkedin/blue?icon=linkedin&label)](https://www.linkedin.com/in/joe-b-phillips/) j.phillips5@lancaster.ac.uk

    Special thanks to **Dom Hardy**
    d.j.hardy@lancaster.ac.uk
        """
    )
    st.sidebar.markdown("<br><br>", unsafe_allow_html=True)
    st.sidebar.image("lancs_logo.png")
    st.sidebar.image("cpom_logo.png")

# ----------------------------------------------------------------------
# Simulators
# ----------------------------------------------------------------------

def render_2d_simulator():
    section_heading("📊", "2D Simulator")
    st.markdown(
        "Choose a surface shape below, or select **Custom** and enter your own numbers "
        "representing height from left to right. Press **▶ Play** to start!"
    )

    sel_col, btn_col, _ = st.columns([1, 0.18, 1.82])
    with sel_col:
        preset = st.selectbox(
            "Surface shape:", options=list(PRESET_2D_GENERATORS.keys()) + ["Custom"], key="preset_2d"
        )
    with btn_col:
        _btn_spacer()
        if preset == "Custom":
            regen = False
            st.button("▶ Run", key="run_btn_2d", type="primary", help="Run the simulator with your custom surface")
        else:
            regen = st.button("↺ New", key="regen_btn_2d", type="primary", help="Regenerate this surface")

    if (preset != st.session_state.get("last_preset_2d") or regen) and preset in PRESET_2D_GENERATORS:
        st.session_state["last_preset_2d"]  = preset
        st.session_state["topo_2d_cached"]  = PRESET_2D_GENERATORS[preset]()

    if preset == "Custom":
        custom_input = st.text_input(
            "Surface heights (space or comma separated):",
            placeholder="e.g. 3 8 2 9 5",
            key="topo_2d_custom",
        )
        if custom_input:
            try:
                parsed = np.array(custom_input.replace(",", " ").split(), dtype=float)
                if len(parsed) > 1 and (parsed >= 0).all():
                    mn, mx = parsed.min(), parsed.max()
                    topo   = parsed if mx == mn else (parsed - mn) / (mx - mn)
                    valid  = True
                else:
                    st.warning("Please enter at least two non-negative numbers.")
                    topo, valid = np.array([0.5, 0.5]), False
            except Exception:
                st.warning("Invalid input - please enter a list of numbers.")
                topo, valid = np.array([0.5, 0.5]), False
        else:
            topo, valid = np.array([0.5, 0.5]), True
    else:
        topo  = st.session_state.get("topo_2d_cached", PRESET_2D_GENERATORS[preset]())
        valid = True

    if valid:
        try:
            render_plot("2d", topo)
        except Exception:
            st.warning("Something went wrong generating the 2D plot. Check your surface input.")

    st.checkbox("Show range window", key="srw_2d")
    st.slider("Range window position:", 0.0, 1.0, key="rw_2d", step=0.01)
    st.checkbox("Show rays", key="sray_2d")
    st.slider("Number of rays to display:", 5, 75, key="nr_2d", step=1)
    st.checkbox("Show POCA", key="sp_2d")
    st.checkbox("Show nadir", key="sn_2d")
    st.checkbox("Show leading edge", key="sle_2d")
    st.slider("Amount of waveform noise:", 0.0, 0.1, key="wn_2d", step=0.01)

    return topo, valid


def render_3d_simulator(topo_2d, topo_2d_valid):
    section_heading("📊", "3D Simulator")
    st.markdown(
        "Choose a surface shape below and press **▶ Play** to get started!"
    )
    st.caption(
        "**Navigating the 3D view:** left-click and drag to rotate, right-click and drag to pan, "
        "scroll to zoom. Note that the 3D view cannot be rotated while the animation is playing. "
        "Press **⏸ Pause** first or wait for the animation to finish if you want to have a look around."
    )

    sel_col, btn_col, _ = st.columns([1, 0.18, 1.82])
    with sel_col:
        preset = st.selectbox("Surface shape:", options=PRESET_3D_OPTIONS, key="preset_3d")
    with btn_col:
        _btn_spacer()
        regen = False
        if preset in PRESET_3D_GENERATORS:
            regen = st.button("↺ New", key="regen_btn_3d", type="primary", help="Regenerate this surface")

    if (preset != st.session_state.get("last_preset_3d") or regen) and preset in PRESET_3D_GENERATORS:
        st.session_state["last_preset_3d"] = preset
        st.session_state["topo_3d_cached"] = PRESET_3D_GENERATORS[preset]()

    if preset == "Use 2D profile":
        st.info(
            "**Using 2D profile:** The surface profile from the 2D simulator above is repeated across "
            "the new direction to fill the circular 3D footprint. Change the shape or preset in the 2D "
            "simulator and the 3D view will update to match.",
            icon="ℹ️",
        )

    if preset == "Randomise":
        rough_col, rand_btn_col = st.columns([2, 1])
        with rough_col:
            st.slider(
                "Roughness:", 0.0, 1.0, key="roughness_3d", step=0.05,
                help="0 = perfectly flat, 1 = full peaks and troughs. Drag to reshape; press **Generate** for a new surface.",
            )
        with rand_btn_col:
            _btn_spacer()
            if st.button("🎲 Generate", key="rand_btn_3d", type="primary"):
                st.session_state["random_topo_3d_base"] = _gen_sinusoidal_3d()

    if preset == "Use 2D profile":
        if topo_2d_valid:
            n      = max(len(topo_2d), 8)
            row    = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(topo_2d)), topo_2d)
            topo   = np.tile(row, (n, 1))
            valid  = True
        else:
            st.warning("Enter a valid surface profile in the 2D simulator above first.")
            topo, valid = None, False
    elif preset == "Randomise":
        if "random_topo_3d_base" not in st.session_state:
            st.session_state["random_topo_3d_base"] = _gen_sinusoidal_3d()
        topo  = _apply_roughness(st.session_state["random_topo_3d_base"], st.session_state["roughness_3d"])
        valid = True
    else:
        topo  = st.session_state.get("topo_3d_cached", PRESET_3D_GENERATORS[preset]())
        valid = True

    if valid:
        try:
            render_plot("3d", topo)
        except Exception:
            st.warning("Something went wrong generating the 3D plot.")

    st.checkbox("Show range window", key="srw_3d")
    st.slider("Range window position:", 0.0, 1.0, key="rw_3d", step=0.01)
    st.checkbox("Show rays", key="sray_3d")
    st.slider("Number of rays to display:", 5, 100, key="nr_3d", step=1)
    st.checkbox("Show POCA", key="sp_3d")
    st.checkbox("Show nadir", key="sn_3d")
    st.checkbox("Show leading edge", key="sle_3d")
    st.slider("Amount of waveform noise:", 0.0, 0.1, key="wn_3d", step=0.01)

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():

    render_sidebar()

    st.title("🛰️ Satellite Radar Altimetry Visualiser")

    st.markdown("""<hr style="border: 1px solid black;">""", unsafe_allow_html=True)

    # Introduction
    st.markdown("<br>", unsafe_allow_html=True)
    section_heading("📡", "What is satellite radar altimetry?")

    # st.markdown(
    #     "TO-DO: (1) VERIFY RANGE WINDOW PLACEMENT FOR 3D IS CORRECT. (2) MENTION THAT WE USE 1.5X BFP IN THE MODEL. (3) THINK OF AND ADD OTHER LIMITATIONS. (4) ADD REFERENCES. (5) ADD DIAGRAMS."
    # )

    st.markdown(
        "There are currently over 13,000 satellites in space, performing a huge variety of different "
        "jobs. Some, such as the James Webb telescope, collect images of the distant universe. Others, "
        "such as Sentinel-2, orbit Earth taking high-resolution pictures of the surface. "
        "And some, such as CryoSat-2, use specialised instruments to accurately measure the height of "
        "the ground below them. "
        "The study of these measurements is known as **remote sensing**. Here, we "
        "cover how satellites like CryoSat-2 work and how they give scientists a "
        "detailed picture of surface elevation across the planet."
    )

    st.markdown(
        "Satellites carrying radar altimeters measure the height of the ground below them by firing a "
        "**pulse** (🔴) of energy at the surface and timing how long it takes to return. This is similar "
        "to how a bat navigates in the dark using echolocation, but on a much larger scale. "
        "The energy travels down to the ground and spreads out in a wide circle, with the strongest signal "
        "directly beneath the satellite at a point called **nadir** (♦️). The area the satellite illuminates is "
        "called its **footprint**. To picture this, imagine pointing a torch at a wall. Just like a torch, a "
        "radar altimeter emits light (energy), just at wavelengths invisible to the human eye. These are called "
        "**radio waves**, and can pass through the atmosphere, cloud, "
        "and rain, which normal optical cameras cannot do. "
        "Since radio waves travel at the speed of light (299,792,458 metres per second), "
        "the satellite is able to work out how far away the ground is to very high precision. Even tiny "
        "differences in return time reveal detailed information about the shape of the surface below. "
        "By repeating this process continuously along its **track** as it travels across the globe, scientists can build "
        "up precise maps of surface height everywhere. This useful for many things such as tracking sea level rise, "
        "monitoring glaciers, and measuring how much ice sheets are gaining or losing mass."
    )

    st.markdown(
        "As echoes return to the satellite, they are recorded as a **waveform** (🟦). This is a graph of "
        "the echo power the satellite recieves over time. "
        "The shape of this waveform carries a lot of information about the ground below. "
        f"The first steep rise, called the **leading edge** ({_dash_swatch('gray')}), corresponds to the "
        "highest point on the surface. This is the point closest to the satellite, whose echo arrives back to the satellite first. "
        "This point is called the **Point of Closest Approach**, or **POCA** (⭐). "
        "By measuring the time between firing the pulse and the start of the leading edge (when the echo from POCA returns), "
        "scientists can calculate the distance from the satellite to the POCA. Since the satellite's "
        "altitude is known precisely (over 700,000 metres), they can then work out the height "
        "of the POCA itself. Therefore, for every pulse of light the satellite sends, we can know how far away the closest point on the surface is.",
        unsafe_allow_html=True
    )

    st.markdown(
        "Once a pulse is fired, the satellite only listens for returning echoes over a short "
        "period of time called the **range window** (🟩). This window exists both in time "
        "(on the waveform) and in physical space (as an area below the satellite). Because the "
        "signal travels at a constant speed, the window in space and time are equivalent. "
        "Over very mountainous terrain, where the surface spans a large range of elevations, "
        "a small window may fail to capture the full picture, with echoes returning before or after the recording window. If the window is positioned too high "
        "or too low, echoes can be missed entirely. This is known as **losing track**. "
        "Getting this right is harder than it sounds, and even sophisticated onboard algorithms "
        "sometimes struggle."
    )

    st.markdown(
        """
        <div class="callout-box">
        <b><i>Phew</i></b> - that was quite a lot! And we simplified things considerably (<i>sorry, scientists</i>).
        To make it easier to understand, we have put together two interactive simulators below that
        should help it all click.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Simulators
    st.markdown("<br>", unsafe_allow_html=True)
    section_heading("🌍", "But what does this actually look like?")

    st.markdown(
        """
        We put together two simulators to help you visualise how satellite radar altimetry works:

        ##### ▸ **2D simulator**

        This is a simplified two-dimensional model considering only surface elevation along a single
        across-track slice (perpendicular to the direction of the satellite). This is loosely analogous to how many modern satellite radar altimeters
        operate, using a technique called **Synthetic Aperture Radar** (SAR), which compresses the
        satellite's footprint into a thin, wide strip, making the across-track geometry dominant.

        ##### ▸ **3D simulator**

        This is a fully three-dimensional version of the model, where emitted pulses spread out across
        a circular footprint. This is loosely analogous to **Low Resolution Mode** (LRM), the
        conventional approach used by older missions and by current satellites operating over simpler
        surfaces such as parts of the open ocean.

        <br>

        **Both simulators are approximations.** They don't replicate real instruments exactly, but they
        capture the key ideas. In general, the 2D waveforms look somewhat more realistic than the 3D ones.

        To get started, select a surface shape below and press **▶ Play**!<br><br>
        """,
        unsafe_allow_html=True,
    )

    render_legend()

    st.markdown("<br>", unsafe_allow_html=True)
    topo_2d, topo_2d_valid = render_2d_simulator()

    st.markdown("<br>", unsafe_allow_html=True)
    render_3d_simulator(topo_2d, topo_2d_valid)

    # How does altimetry actually work?
    st.markdown("<br>", unsafe_allow_html=True)
    section_heading("📡", "How does altimetry actually work?")
    st.markdown(
        "We covered the basics above, but simplified a lot. In practice there is considerably "
        "more nuance. Here we go into a bit more detail."
    )

    with st.expander("Technical detail"):
        st.markdown(
            """
            <div style="background-color:#fff8e1; border-left:4px solid #f0a500; padding:10px 14px; border-radius:4px; color:#4a3800; margin-bottom:1em;">
            ⚠️ This section goes into more technical detail - readers beware! It is also intended only as a brief overview.
            </div>

            **The waveform**

            Over a flat surface, the expected waveform shape is well described by the **Brown model**,
            which treats the surface as a collection of small independent scatterers. The model predicts a rapid
            power rise at the leading edge - as the pulse wavefront first intersects the surface - followed by a
            peak near the POCA return, and then a gradual exponential decay as echoes return from progressively
            greater distances. Returns from the edges of the footprint travel further and arrive with diminishing
            antenna gain, while the trailing edge also carries weaker returns from various secondary effects.<br><br>

            **Footprint geometry and measurement modes**

            Three different concepts of 'footprint' matter in altimetry, and they are easy to confuse.

            The **beam-limited footprint** is the total area on the ground illuminated by the antenna - defined by
            the antenna's radiation pattern at its -3 dB half-power point. For a small spaceborne antenna this
            can span tens of kilometres in diameter. Because the antenna aperture is physically small relative to
            the radar wavelength, the beam cannot be focused more tightly without a much larger antenna - a
            practical constraint for satellite instruments. Using this entire illuminated area as the measurement
            resolution would make it impossible to attribute any given reflection to a specific surface location,
            so a finer effective resolution is needed.

            To achieve this, the transmitted pulse is shortened so that only returns within a narrow time window
            contribute to each waveform bin. This defines the **pulse-limited footprint** - typically 3-5 km in
            diameter depending on surface roughness and slope. As the pulse wavefront expands outward
            from the point of closest approach, each range bin in the waveform corresponds to a specific
            concentric annulus of the surface rather than the beam-limited footprint as a whole. Think of it as
            an expanding ring: the first return comes from the small central region at POCA, and subsequent
            returns come from progressively wider rings at greater range. This bin-to-annulus correspondence is
            what makes it possible to attribute waveform signal to specific parts of the surface. Flat surfaces
            yield narrower annuli and sharper pulse-limited footprints than rough or sloping terrain. However,
            because the altimeter must wait for each echo to return before transmitting the next, the along-track
            sampling rate is limited to a low pulse repetition frequency (PRF, typically ~2 kHz), and
            kilometre-scale along-track resolution is typical. This conventional approach is called
            **Low Resolution Mode (LRM)**, and has been the standard for missions including ERS-1/2, Envisat,
            and CryoSat-2 over ice sheet interiors. This is what the 3D simulator approximates.

            **Synthetic Aperture Radar** (SAR) altimetry overcomes this along-track limitation by transmitting
            bursts of pulses at a much higher PRF (~18 kHz). By coherently processing overlapping returns using
            Doppler processing - which exploits frequency shifts induced by satellite motion - SAR synthesises a
            longer effective antenna aperture along-track, defining a third concept: the **Doppler-limited
            footprint**. This compresses the along-track footprint to ~380 m for CryoSat-2 (~300 m for
            Sentinel-3), dramatically improving along-track resolution relative to LRM, while the across-track
            footprint remains pulse-limited. Stacking multiple looks of the same location also reduces
            radar speckle noise - a process called **multilooking**. The across-track geometry is therefore
            dominant in SAR, making the 2D (across-track slice) simulator an approximate analogy.<br><br>

            **Retracking and slope correction**

            Once a waveform is captured, extracting a usable elevation requires two key steps. The altimeter
            does not measure absolute range directly - instead, each waveform is recorded relative to a
            **reference range**: a known distance used to position the range window, computed
            onboard from predicted orbit and surface elevation and stored alongside the waveform in the
            Level-1 data products. The actual range to the surface is then determined by finding where the
            leading edge falls relative to this reference and adding the corresponding offset.

            **Retracking** performs this step: it identifies the precise range bin corresponding to the first
            surface return, which - combined with the reference range - gives the total range from satellite
            to surface. Retracking algorithms vary considerably in approach: physical retrackers fit an
            analytical model to the waveform to represent the underlying physics of the radar interaction
            with the surface, while empirical retrackers focus solely on the geometry of the recorded waveform
            and make no assumption about its shape. In practice, empirical approaches are more commonly used
            over ice sheets, where complex terrain produces irregular waveforms that physical models struggle
            to fit. **Slope correction** then determines *where* on the surface that range corresponds to,
            since POCA is rarely directly below the satellite over sloping terrain, and typically relies on
            an auxiliary digital elevation model. Together, these steps reduce the full waveform to a single
            point elevation estimate. Slope correction is the dominant source of error in ice sheet altimetry,
            with uncertainties commonly reaching tens of metres in elevation and kilometre-scale horizontal
            offsets over complex terrain - a limitation that interferometric techniques (see below) were
            specifically designed to address.<br><br>

            **Surface tracking**

            Before any of this can happen, the satellite's onboard system must position the range window
            correctly - a process called **surface tracking**. Two broad approaches exist: **closed-loop
            tracking**, where the window is continuously updated based on the most recently received echo,
            and **open-loop tracking**, where it is driven by a pre-loaded elevation model rather than live
            feedback. Open-loop tracking is more robust over complex terrain, where steep or rapidly changing
            topography means the previous echo is often a poor predictor of the next. Over flat, predictable
            terrain closed-loop tracking works well, but over rough or steeply sloping ground either approach
            can struggle, causing **loss of track** where echoes fall outside the range window and no measurement
            is recorded. This is one reason why data coverage commonly degrades over the complex topography
            found at ice sheet margins, where measurements are often most needed.<br><br>

            **Interferometric SAR (SARIn) mode**

            CryoSat-2 carries a second antenna, offset in the across-track direction. By comparing the phase
            of the signal received at each antenna - a technique called **interferometry** - it is possible to
            measure the angle of arrival of the echo, resolving *where* across-track the POCA return came
            from. This directly addresses the slope correction problem described above, bypassing the need
            for an auxiliary DEM entirely. Taking this further, **swath processing** uses phase
            information throughout the waveform, not just at the leading edge, to recover elevations from
            across the whole illuminated footprint, producing dense elevation measurements from a
            single pass. Neither simulator here uses interferometric information; both are based on power
            waveforms only, which is the more general and challenging case faced by SAR-only missions such
            as Sentinel-3.
            """,
            unsafe_allow_html=True,
        )

    # How does the simulator work?
    st.markdown("<br>", unsafe_allow_html=True)
    section_heading("🔧", "How does the simulator work?")
    st.markdown(
        "Each simulator works by tracing imaginary lines (rays) from the satellite down to the surface. "
        "For each ray, it finds where the line hits the ground and calculates how long the signal would "
        "take to travel there and back. That travel time is mapped to a position in the waveform. The "
        "radar signal is strongest directly below the satellite and fades naturally towards the edges, "
        "so rays pointing more steeply downward contribute more to the waveform. "
        "In the **2D simulator**, rays fan out left and right across a vertical slice of the surface. "
        "In the **3D simulator**, they spread out in all directions across a circular area. "
        "Below, we cover this in a bit more detail."
    )

    with st.expander("Technical detail"):
        st.markdown(
            """
            <div style="background-color:#fff8e1; border-left:4px solid #f0a500; padding:10px 14px; border-radius:4px; color:#4a3800; margin-bottom:1em;">
            ⚠️ This section goes into more technical detail - readers beware! It is also intended only as a brief overview.
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
            G(θ, φ) = exp(-θ² × (cos²φ / γ₁² + sin²φ / γ₂²))
            ```

            where θ is the off-nadir angle, φ is the azimuth angle, and γ₁ = 0.0133 rad,
            γ₂ = 0.0148 rad are the CryoSat-2 3 dB beamwidths in the along- and across-track directions.

            In both simulators, instrument parameters - satellite altitude (~717 km), across-track
            footprint size (~15 km), and range window height (~240 m) - are taken from CryoSat-2 in SARIn
            mode. For the 2D case, the number of rays (1024) and waveform bins (128) were tuned against
            real waveforms. This has yet to be done for the 3D case, where we currently use 2048 rays and
            128 bins. For reference, a real CryoSat-2 SARIn waveform has 1024 bins; matching this in
            practice can be done using interpolation.

            Within the simulator, the input topography is taken to be 1.5x the size of the beam-limited footprint.
            Although returns from outside the beam-limited footprint have reduced contribution, they still affect the waveform shape,
            particularly - and most notably - the trailing edge. 

            Currently, the simulators cannot account for surface elevations that fall outside the
            range window. To handle this, input topography is rescaled from [0, 1] to [0.2, 0.8] before
            being passed to the simulator. The resulting waveform is then embedded into a longer
            zero-padded waveform spanning the full animation, with the range window parameters adjusted
            to match.<br><br>

            **2D simulator (SAR-analogous)**

            Rays are cast across a 1D topographic profile at uniform across-track intervals, with the range
            window bottom following a curved arc - the locus of points at equal slant range from the
            satellite. This collapses the along-track dimension and is loosely analogous to SAR altimetry
            as used by CryoSat-2 and Sentinel-3. In SAR mode, coherent processing of overlapping pulse
            returns compresses the along-track footprint to around 380 m (for CryoSat-2), making the
            across-track geometry dominant (~15 km beam-limited).<br><br>

            **3D simulator (LRM-analogous)**

            Rays are cast across a 2D circular footprint (radius ~7.5 km) arranged on a hexagonal grid
            with uniform spacing, with the number of rays dynamically adjusted to maintain equidistance.
            This is loosely analogous to Low Resolution Mode (LRM) altimetry - the conventional
            pulse-limited mode used by missions such as ERS-1/2 and Envisat, and by CryoSat-2 over ice
            sheet interiors. LRM waveforms integrate returns from all azimuth directions simultaneously,
            producing the characteristic shape over flat surfaces commonly modelled using the Brown model.<br><br>

             **Leading edge detection**
             ...<br><br>

            **Some limitations**

            - *Scattering*: The model assumes spatially uniform reflectivity. In reality, backscatter varies
            with surface type, and over snow and ice, volume scattering and signal penetration into the
            snowpack can shift the apparent surface return, affecting both waveform shape and retrieved
            elevation.
            - *No instrument effects*: Pulse compression, range migration, multilooking, and thermal noise
            are not modelled. The noise visible on the waveform is uniform random noise added purely for
            visual clarity.
            - *Single intersection per ray*: Only the first surface hit per ray is recorded. Multi-path
            returns and radar layover - where steep terrain causes reflections from different locations to
            arrive at the same range - are not handled. Angle-dependent reflectivity is also not modelled;
            this was tested during development but found to have negligible effect on simulated waveform
            shape and was therefore omitted.
            """,
            unsafe_allow_html=True,
        )


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    main()