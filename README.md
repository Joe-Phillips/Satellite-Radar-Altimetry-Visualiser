# Satellite Radar Altimetry Visualiser

An interactive Streamlit app for visualising satellite radar altimetry, built for educational use.
Access it [here](https://satellite-radar-altimetry-tool.streamlit.app/)!

---

Made by **Dr Joe Phillips**  
[![GitHub](https://badgen.net/badge/icon/GitHub/green?icon=github&label)](https://github.com/Joe-Phillips)
[![LinkedIn](https://badgen.net/badge/icon/linkedin/blue?icon=linkedin&label)](https://www.linkedin.com/in/joe-b-phillips/)
&nbsp; ✉️ j.phillips5@lancaster.ac.uk

Special thanks to **Dom Hardy**  
&nbsp;✉️ d.j.hardy@lancaster.ac.uk

---

## Overview

The app combines an accessible introduction to satellite radar altimetry with two interactive waveform simulators. Both animate radar pulses travelling from a virtual satellite to the surface in real time, building up the corresponding waveform as they go. Users can adjust surface shape, range window position, number of rays, noise level, and toggle display elements such as the POCA marker.

## Simulators

**2D (SAR-analogous)** - Rays are cast across a 1D across-track topographic profile on a geometric ray-casting approach, loosely analogous to SAR altimetry. Supports preset surfaces (flat, sloped, peaked, valley, rough) and custom user-defined height profiles.

**3D (LRM-analogous)** - Rays are cast across a 2D circular footprint sampled on a hexagonal grid, loosely analogous to LRM altimetry. Supports presets, a roughness-controlled randomiser, and a mode that extrudes the 2D profile into 3D.

Instrument parameters (satellite altitude ~717 km, footprint ~15 km, range window ~240 m) are taken from CryoSat-2. See the in-app technical detail expanders for full implementation notes.

## Structure
```
├── app.py                            # Main Streamlit application
├── simulate_altimetry_waveform.py    # 2D and 3D waveform simulators
├── animate_altimetry_waveform.py     # Plotly animation generators
├── s3.png                            # Satellite image used in 2D plot
├── lancs_logo.png                    
├── cpom_logo.png                     
└── .streamlit/
    └── config.toml                   # Theme configuration
```

## Running Locally
```bash
pip install streamlit numpy scipy plotly
streamlit run app.py
```
