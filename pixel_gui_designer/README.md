# Pixel Filter Designer — GUI

Interactive PyQt5 GUI for editing pixelated RF filter designs stored in `config.yaml`.

## Launch

```bash
# From the repo root:
python pixel_gui_designer/designer.py

# Open a specific config directly:
python pixel_gui_designer/designer.py virtuoso_engine/config.yaml

# Or the larger-pixel variant:
python pixel_gui_designer/designer.py virtuoso_engine/config_larger_pixel.yaml
```

---

## Window Layout

```
┌──────────────────────────────────────────────────────────────┐
│  Toolbar: Open | Save | Save As | Mode | Brush | Grid ops    │
├──────────────────────────────────────┬───────────────────────┤
│                                      │  States Table         │
│         Pixel Grid                   │  (right panel)        │
│         (scrollable)                 │                       │
│                                      │                       │
├──────────────────────────────────────┴───────────────────────┤
│  Property Panel  (appears when a pixel is selected)          │
└──────────────────────────────────────────────────────────────┘
```

---

## Pixel Types

| # | Name | Colour     | Description                         |
|---|------|------------|-------------------------------------|
| 0 | GND  | Dark grey  | Ground layer only (M4, no top metal)|
| 1 | MTL  | Steel blue | Top + ground metal (M9 + M4)        |
| 2 | IND  | Amber      | Inductor port — carries inductance  |
| 3 | SW   | Red        | Switch port — ON/OFF per state      |
| 4 | VIA  | Purple     | Via / ground-short                  |

---

## Selection Mode  (`S` key or toolbar button)

Click any cell on the grid to select it.  
The **Property Panel** (bottom) shows its properties.

### Changing pixel type
Use the **Type** dropdown.  
Switching a pixel *away* from SW or IND automatically removes it from all
bookkeeping lists (`cap_values`, `inductor_values`, `state_list` columns, cap pairs).

### Inductor pixel (type 2)
A dropdown selects the inductance value:

| Index | Value  | s1p file              |
|-------|--------|-----------------------|
| 0     | 100 nH | 100nH_active_inductance.s1p |
| 1     | 200 nH | 200nH_active_inductance.s1p |
| 2     | 400 nH | 400nH_active_inductance.s1p |
| 3     | 800 nH | 800nH_active_inductance.s1p |

### Switch pixel (type 3)
A spinbox sets the **series coupling capacitance** in pF (stored as `cap_values[sw_index]`).

### Cap Pair
Any non-ground pixel can be linked to another non-ground pixel with a
**capacitor** (maps to `cap_pixel_pairs` + `cap_pair_values`).

- Click **Create pair…** → enter the target row/col and cap value in pF.
- The GUI draws a dashed line between paired pixels.
- If you change either pixel to GND the pair is automatically removed.

---

## Brush Mode  (`B` key or toolbar button)

Select brush type from the **Brush** dropdown (GND=0 or MTL=1 only).  
**Click and drag** across pixels to paint them.

> Brush mode is intentionally restricted to GND and MTL because inductors and
> switches carry per-pixel metadata (inductance value, cap value, state columns)
> that must be set individually in Selection mode.

---

## States Table (right panel)

Rows = simulation states (from `state_list` in the YAML).  
Columns = `SW_0 … SW_N` (and optionally `IND_0 … IND_M`).

**Click a cell** to toggle between ON (1) and OFF (0):
- Switch cells: red = ON, grey = OFF.
- Inductor cells (when enabled): amber = ON, dark green = OFF.

| Button              | Action                                      |
|---------------------|---------------------------------------------|
| **+ Add state**     | Append a new all-OFF state                  |
| **– Remove state**  | Remove the selected (or last) state         |
| **Enable per-state inductor switching** | Activates `inductor_switch_states`; adds IND columns |

---

## Grid Resizing (toolbar)

| Button  | Action                        |
|---------|-------------------------------|
| +Row↓   | Append a GND row at the bottom|
| +Row↑   | Prepend a GND row at the top  |
| –Row    | Remove a row (asks for index) |
| +Col→   | Append a GND column at right  |
| +Col←   | Prepend a GND column at left  |
| –Col    | Remove a column (asks for index)|

Cap pair coordinates are automatically adjusted after insertions/deletions.

---

## File Operations

| Action     | Shortcut      | Behaviour                                                 |
|------------|---------------|-----------------------------------------------------------|
| Open       | Ctrl+O        | Load any `config.yaml` — only the `pixelated:` section is edited |
| Save       | Ctrl+S        | Overwrite the currently-open file                         |
| Save As…   | Ctrl+Shift+S  | Write to a new path                                       |

The entire YAML is preserved (mode, layout, simulation, paths…).  
Only the `pixelated:` subsection is modified.

---

## YAML Keys Managed by the GUI

```yaml
pixelated:
  pixel_state_matrix:      # 2-D list of pixel types (0–4)
  state_list:              # [[sw0, sw1, ...], ...]  — one list per state
  cap_values:              # [F, F, ...]  — per SW pixel, row-first order
  inductor_values:         # [0–3, ...]  — per IND pixel, index into translation table
  inductor_switch_states:  # [[ind0, ind1, ...], ...]  — optional, per IND per state
  cap_pixel_pairs:         # [[[r1,c1],[r2,c2]], ...]
  cap_pair_values:         # [F, F, ...]  — one per pair
```

All other top-level keys are left untouched.

---

## Keyboard Shortcuts

| Key     | Action              |
|---------|---------------------|
| `S`     | Switch to Selection mode |
| `B`     | Switch to Brush mode     |
| `Esc`   | Return to Selection mode |
| `Ctrl+O`| Open file               |
| `Ctrl+S`| Save                    |
| `Ctrl+Shift+S` | Save As          |
