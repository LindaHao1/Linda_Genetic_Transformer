templates/simplified_layout.il
Draws only the two electrically significant layers:

M9 top radiating element (full pixel square)
M4 ground shield (same footprint)
Three EMX layout pins: PIX_OUT (top edge, port 1), VSS (right edge, port 2), CENTER (center, port 3 — the switch node)
templates/simplified_sim.il

Exports GDS, runs EMX with -p PIX_OUT -p VSS -p CENTER → produces .s3p
Runs Spectre twice with different switch circuit topologies:
Run 0 (ON): CENTER → c_sw(cap_value) → sw_mid → r_sw(r_on) → GND
Run 1 (OFF): CENTER → c_sw(cap_value) → sw_mid1 → r_off_comp(r_off) → sw_mid2 → c_off_comp(c_off) → GND
No TSMC model includes needed — uses ideal Spectre resistor/capacitor
Modified files
engine.py — Mode-dispatch architecture:

_MODE_TEMPLATES registry maps mode names to template filenames
_render_layout / _render_sim dispatch to mode-specific renderers
_make_cell_name is mode-aware (simplepix_ps35p0_cap1e-12_ron3_roff260_coff8e-14)
_get_run_labels returns ["on", "off"] for simplified mode, voltages for layout mode
New renderers: _render_simplified_layout and _render_simplified_sim
config.yaml — New sections:

mode: "layout_pixel" top-level selector (overridable per-run)
switch_model: section with cap_value, r_on, r_off, c_off
Usage

# Use simplified_pixel mode (from config default)
engine = VirtuosoPixelEngine("config.yaml")
result = engine.run({"mode": "simplified_pixel"})

# Sweep r_on in simplified mode
results = engine.sweep(["switch_model", "r_on"], [2, 3, 5])

# Original layout_pixel mode unchanged
result = engine.run({"nmos": {"width": "6u"}})