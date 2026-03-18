"""
VirtuosoPixelEngine — Python driver for parametric RF pixel simulation.

Simulation Modes
----------------
``layout_pixel``  (default)
    Full layout-based single-pixel simulation.  Uses an NMOS switch + MIM
    capacitor placed via Virtuoso layout, exported to GDS, and extracted with
    EMX.  Spectre then re-inserts the device models and sweeps the gate control
    voltage to produce on/off S-parameter results.

``simplified_pixel``
    Compact model single-pixel simulation.  Only the top radiating layer (M9)
    and the ground shield (M4) are drawn — no transistor layout.  EMX extracts
    a 3-port S-parameter (PIX_OUT, VSS, CENTER).  Spectre re-connects the
    CENTER port to a compact switch circuit model for each state:

        ON  — CENTER → cap_value → r_on → GND
        OFF — CENTER → cap_value → r_off → c_off → GND

    All switch model parameters (cap_value, r_on, r_off, c_off) are tunable
    via config.

``pixelated_design_simple``
    Full NxM pixel-grid simulation for a switchable RF filter.  An arbitrary
    matrix of pixel types (ground, metal, inductor, switch) is drawn in
    Virtuoso.  EMX extracts a multi-port S-parameter covering all central
    ports plus the left/right I/O feedline ports.  Python then generates one
    Spectre testbench per switch state and runs them, producing one .s2p file
    per state.

    Pixel types:
        0 — ground layer only (M4)
        1 — top + ground layer (M9 + M4)
        2 — top + ground + inductor central port (connected to s1p)
        3 — top + ground + switch central port (ON/OFF per state)

Adding New Modes
----------------
1. Add a new entry to ``_MODE_TEMPLATES`` mapping the mode name to a dict
   with "layout" and "sim" (or "emx") template filenames relative to
   templates/<mode_name>/.
2. Implement ``_render_<mode>_layout`` and ``_render_<mode>_sim`` (or
   ``_render_<mode>_emx``) methods on ``VirtuosoPixelEngine``.
3. Register the render methods in ``_render_layout`` and ``_render_sim``
   dispatch blocks.
4. Implement the cell name logic in ``_make_cell_name``.
5. Add a branch to ``_get_run_labels`` for the new mode.
6. If the mode needs post-EMX Python work, add a branch to ``_post_emx_step``.

Virtuoso session
----------------
A single Virtuoso process is shared across all runs (including sweeps).
It is started lazily on the first call to ``run()`` and can be closed
explicitly via ``engine.close()`` or by using the engine as a context manager::

    with VirtuosoPixelEngine("config.yaml") as engine:
        results = engine.sweep(["nmos", "width"], ["3u", "6u", "9u"])

The Xvfb virtual display must already be running on the configured display
(default ``:99``) **before** the first ``run()`` call, e.g. in a tmux session::

    Xvfb :99 -screen 0 1024x768x24 -extension XKEYBOARD &
"""

from __future__ import annotations

import copy
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .runner import PersistentVirtuosoRunner, VirtuosoResult
from .plotter import plot_s11_results, plot_s2p_results
from .pixel_matrix import (
    analyze_matrix,
    build_emx_port_flags,
    build_skill_pixel_types_list,
    build_skill_port_names_list,
    total_ports as matrix_total_ports,
)
from .spectre_runner import run_all_states

# Absolute path to the templates directory (next to this file)
_TEMPLATES_DIR = Path(__file__).parent / "templates"

# ---------------------------------------------------------------------------
# Mode registry
# Maps mode name → dict describing template file(s) within templates/<mode>/
# "layout" key  — SKILL layout generation script
# "sim"    key  — SKILL EMX+Spectre script  (layout_pixel, simplified_pixel)
# "emx"    key  — SKILL EMX-only script      (pixelated_design_simple)
#                 For modes with "emx", Spectre is driven by Python.
# ---------------------------------------------------------------------------
_MODE_TEMPLATES: Dict[str, dict] = {
    "layout_pixel": {
        "layout": "layout_pixel/layout.il",
        "sim":    "layout_pixel/sim.il",
    },
    "simplified_pixel": {
        "layout": "simplified_pixel/layout.il",
        "sim":    "simplified_pixel/sim.il",
    },
    "pixelated_design_simple": {
        "layout": "pixelated_design_simple/layout.il",
        "emx":    "pixelated_design_simple/emx.il",
    },
}


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    cell_name:      str
    run_dir:        Path
    layout_result:  VirtuosoResult
    sim_result:     VirtuosoResult
    s_param_files:  List[Path] = field(default_factory=list)
    plot_paths:     dict       = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.layout_result.success and self.sim_result.success


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class VirtuosoPixelEngine:
    """
    Parametric driver for Virtuoso layout + EMX/Spectre simulation.

    Supports multiple simulation modes (see module docstring).  The active
    mode is set via the ``mode`` key in the config YAML and can be overridden
    per-run via the ``config_override`` argument.

    Usage::

        engine = VirtuosoPixelEngine("virtuoso_engine/config.yaml")

        # --- layout_pixel mode (default) ---
        result = engine.run()
        result = engine.run({"nmos": {"width": "6u", "fingers": 8}})
        results = engine.sweep(["nmos", "width"], ["3u", "6u", "9u"])

        # --- simplified_pixel mode ---
        result = engine.run({
            "mode": "simplified_pixel",
            "switch_model": {"r_on": 5, "r_off": 300},
        })

        # --- pixelated_design_simple mode ---
        result = engine.run({
            "mode": "pixelated_design_simple",
            "pixelated": {
                "pixel_state_matrix": [[1, 3, 1], [0, 2, 0]],
                "state_list": [[0], [1]],
                "cap_values": [1e-12],
            },
        })

        # Always close the Virtuoso session when done
        engine.close()

        # Or use as a context manager:
        with VirtuosoPixelEngine("config.yaml") as engine:
            results = engine.sweep(...)
    """

    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)
        self.base_config = self._load_yaml(self.config_path)
        self._runner: Optional[PersistentVirtuosoRunner] = None
        print(f"[VirtuosoPixelEngine] Loaded config: {self.config_path}")

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "VirtuosoPixelEngine":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def close(self) -> None:
        """Shut down the persistent Virtuoso session."""
        if self._runner is not None:
            self._runner.close()
            self._runner = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, config_override: Optional[Dict] = None) -> RunResult:
        """
        Execute one full layout + simulation run.

        Args:
            config_override: Nested dict deep-merged over the base config.
                             Only the keys you want to change need to be provided.

        Returns:
            RunResult with success flags, log text, and plot paths.
        """
        cfg  = _deep_merge(self.base_config, config_override or {})
        mode = cfg.get("mode", "layout_pixel")

        if mode not in _MODE_TEMPLATES:
            raise ValueError(
                f"Unknown simulation mode {mode!r}. "
                f"Valid modes: {list(_MODE_TEMPLATES)}"
            )

        cell_name = _make_cell_name(cfg)
        run_dir   = Path(cfg["project"]["parent_output_dir"]) / cell_name
        emx_dir   = run_dir / "emx_results"
        log_dir   = run_dir / "logs"
        plot_dir  = run_dir / "plots"

        for d in (run_dir, emx_dir, log_dir, plot_dir):
            d.mkdir(parents=True, exist_ok=True)

        # --- Save config snapshot ---
        config_snapshot = run_dir / "config.yaml"
        with config_snapshot.open("w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

        print(f"\n[run] Mode:   {mode}")
        print(f"[run] Cell:   {cell_name}")
        print(f"[run] Output: {run_dir}")

        # --- Ensure Virtuoso session is running ---
        runner = self._get_runner(cfg)

        # --- Render and run layout script ---
        layout_il  = run_dir / "generated_layout.il"
        layout_log = log_dir / "virtuoso_layout.log"
        self._render_layout(cfg, cell_name, layout_il)

        print(f"[run] Running Virtuoso layout generation ({mode})...")
        layout_result = runner.run_script(
            skill_file=layout_il,
            log_file=layout_log,
        )
        _print_result("layout", layout_result)

        if not layout_result.success:
            print("[run] Layout failed — skipping simulation.")
            return RunResult(
                cell_name=cell_name,
                run_dir=run_dir,
                layout_result=layout_result,
                sim_result=VirtuosoResult(
                    success=False, returncode=-1,
                    log_path=log_dir / "virtuoso_sim.log",
                    error_message="Skipped due to layout failure.",
                ),
            )

        # --- Render and run simulation / EMX script ---
        sim_il  = run_dir / "generated_sim.il"
        sim_log = log_dir / "virtuoso_sim.log"
        self._render_sim(cfg, cell_name, emx_dir, sim_il)

        print(f"[run] Running Virtuoso EMX + Spectre simulation ({mode})...")
        sim_result = runner.run_script(
            skill_file=sim_il,
            log_file=sim_log,
        )
        _print_result("simulation", sim_result)

        # --- Post-EMX Python step (pixelated_design_simple only) ---
        extra_s_param_files = self._post_emx_step(cfg, cell_name, emx_dir)

        # --- Collect S-parameter files and plot ---
        run_labels    = _get_run_labels(cfg)
        s_param_files = extra_s_param_files or _find_s1p_files(emx_dir, cell_name, len(run_labels))

        plot_paths: dict = {}
        if s_param_files:
            plot_paths = _make_plots(
                mode=mode,
                s_param_files=s_param_files,
                labels=run_labels,
                plot_dir=plot_dir,
            )
        else:
            print("[run] No S-parameter files found — skipping plots.")

        return RunResult(
            cell_name=cell_name,
            run_dir=run_dir,
            layout_result=layout_result,
            sim_result=sim_result,
            s_param_files=s_param_files,
            plot_paths=plot_paths,
        )

    def sweep(
        self,
        param_path: List[str],
        values: List[Any],
    ) -> List[RunResult]:
        """
        Sweep a single parameter over multiple values.

        Args:
            param_path: Nested key path into the config dict,
                        e.g. ``["nmos", "width"]`` or ``["switch_model", "r_on"]``.
            values:     List of values to set at that path.

        Returns:
            List of RunResult, one per value.

        Example::

            results = engine.sweep(["nmos", "width"], ["3u", "6u", "9u"])
            results = engine.sweep(["switch_model", "r_on"], [2, 3, 5])
        """
        overrides = [_build_nested(param_path, v) for v in values]
        labels    = [f"{'.'.join(param_path)}={v!r}" for v in values]
        return self.sweep_configs(overrides, labels=labels)

    def sweep_configs(
        self,
        overrides: List[Dict],
        labels: Optional[List[str]] = None,
    ) -> List[RunResult]:
        """
        Run a sequence of configurations, each defined by an arbitrary
        config override dict merged over the base config.

        Use this when you need to vary **multiple parameters at once** per
        run, or when the parameter combinations don't fit a single-axis sweep.

        Args:
            overrides: List of nested dicts, each deep-merged over the base
                       config before the run.  Only the keys you want to
                       change need to be present.
            labels:    Optional list of human-readable labels (one per
                       override) printed in the sweep header.  Defaults to
                       the index number if not provided.

        Returns:
            List of RunResult, one per override, in the same order.

        Example::

            # layout_pixel sweep
            results = engine.sweep_configs([
                {"nmos": {"width": "3u", "fingers": 4}},
                {"nmos": {"width": "6u", "fingers": 8}},
            ])

            # pixelated_design_simple multi-state run
            results = engine.sweep_configs([
                {"mode": "pixelated_design_simple",
                 "pixelated": {"pixel_state_matrix": [[1,3,1],[0,2,0]],
                               "state_list": [[0],[1]],
                               "cap_values": [1e-12]}},
            ])
        """
        results = []
        for idx, override in enumerate(overrides):
            label = labels[idx] if labels else str(idx)
            print(f"\n{'='*60}")
            print(f"[sweep_configs] run {idx + 1}/{len(overrides)}: {label}")
            results.append(self.run(override))
        return results

    # ------------------------------------------------------------------
    # Session management (internal)
    # ------------------------------------------------------------------

    def _get_runner(self, cfg: dict) -> PersistentVirtuosoRunner:
        """Lazily create and start the Virtuoso session on first use."""
        if self._runner is None:
            proj = cfg["project"]
            display = proj.get("display", ":99")
            log_dir = Path(proj["parent_output_dir"]) / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            self._runner = PersistentVirtuosoRunner(
                virtuoso_dir=proj["virtuoso_dir"],
                display=display,
                session_log=log_dir / "virtuoso_session.log",
            )
            self._runner.start()
        return self._runner

    # ------------------------------------------------------------------
    # Template rendering — mode dispatch
    # ------------------------------------------------------------------

    def _render_layout(
        self,
        cfg: dict,
        cell_name: str,
        output_path: Path,
    ) -> None:
        """Dispatch to the mode-appropriate layout renderer."""
        mode = cfg.get("mode", "layout_pixel")
        if mode == "layout_pixel":
            self._render_layout_pixel(cfg, cell_name, output_path)
        elif mode == "simplified_pixel":
            self._render_simplified_layout(cfg, cell_name, output_path)
        elif mode == "pixelated_design_simple":
            self._render_pixelated_simple_layout(cfg, cell_name, output_path)
        else:
            raise ValueError(f"No layout renderer for mode {mode!r}")

    def _render_sim(
        self,
        cfg: dict,
        cell_name: str,
        emx_dir: Path,
        output_path: Path,
    ) -> None:
        """Dispatch to the mode-appropriate simulation/EMX renderer."""
        mode = cfg.get("mode", "layout_pixel")
        if mode == "layout_pixel":
            self._render_layout_pixel_sim(cfg, cell_name, emx_dir, output_path)
        elif mode == "simplified_pixel":
            self._render_simplified_sim(cfg, cell_name, emx_dir, output_path)
        elif mode == "pixelated_design_simple":
            self._render_pixelated_simple_emx(cfg, cell_name, emx_dir, output_path)
        else:
            raise ValueError(f"No sim renderer for mode {mode!r}")

    def _post_emx_step(
        self,
        cfg: dict,
        cell_name: str,
        emx_dir: Path,
    ) -> List[Path]:
        """
        Mode-specific post-processing after the SKILL sim/emx script completes.

        For modes where Spectre is driven by Python (pixelated_design_simple),
        this method runs all per-state Spectre simulations and returns the
        resulting S-parameter file paths.

        For other modes (SKILL handles Spectre internally), returns an empty
        list so that ``run()`` falls back to finding .s1p files on disk.
        """
        mode = cfg.get("mode", "layout_pixel")
        if mode == "pixelated_design_simple":
            return self._run_pixelated_simple_states(cfg, cell_name, emx_dir)
        return []

    # ------------------------------------------------------------------
    # layout_pixel renderers
    # ------------------------------------------------------------------

    def _render_layout_pixel(
        self,
        cfg: dict,
        cell_name: str,
        output_path: Path,
    ) -> None:
        """Render the layout.il template for layout_pixel mode."""
        template_text = (_TEMPLATES_DIR / "layout_pixel/layout.il").read_text()
        n  = cfg["nmos"]
        m  = cfg["mim_cap"]
        la = cfg["layout"]
        rendered = string.Template(template_text).substitute(
            LIB_NAME        = cfg["project"]["lib_name"],
            CELL_NAME       = cell_name,
            PDK_LIB         = cfg["project"]["pdk_lib"],
            NMOS_CELL       = n["cell_layout"],
            MIM_CELL_LAYOUT = m["cell_layout"],
            MIM_TOTAL_C     = m["total_c"],
            MIM_LENGTH      = m["length"],
            MIM_WIDTH       = m["width"],
            PIXEL_SIZE      = la["pixel_size"],
            GND_LAYER       = la["gnd_layer"],
            PIXEL_LAYER     = la["pixel_layer"],
            VIA_SIZE        = la["via_size"],
            NMOS_WIDTH      = n["width"],
            NMOS_LENGTH     = n["length"],
            NMOS_FINGERS    = n["fingers"],
        )
        output_path.write_text(rendered)

    def _render_layout_pixel_sim(
        self,
        cfg: dict,
        cell_name: str,
        emx_dir: Path,
        output_path: Path,
    ) -> None:
        """Render the sim.il template for layout_pixel mode."""
        template_text = (_TEMPLATES_DIR / "layout_pixel/sim.il").read_text()
        n   = cfg["nmos"]
        m   = cfg["mim_cap"]
        sim = cfg["simulation"]
        p   = cfg["paths"]

        vctrl_skill = "list(" + " ".join(str(v) for v in sim["control_voltages"]) + ")"

        rendered = string.Template(template_text).substitute(
            LIB_NAME         = cfg["project"]["lib_name"],
            CELL_NAME        = cell_name,
            DEST_FOLDER      = str(emx_dir),
            EMX_BINARY       = p["emx_binary"],
            PROC_FILE        = p["proc_file"],
            EMX_FREQ_START   = sim["emx_freq_start"],
            EMX_FREQ_STOP    = sim["emx_freq_stop"],
            EMX_FREQ_STEP    = sim["emx_freq_step"],
            EMX_PARALLEL     = sim["emx_parallel_threads"],
            TSMC_MODELS      = p["tsmc_models"],
            NMOS_WIDTH       = n["width"],
            NMOS_LENGTH      = n["length"],
            NMOS_FINGERS     = n["fingers"],
            MIM_CELL_SPECTRE = m["cell_spectre"],
            MIM_LENGTH       = m["length"],
            MIM_WIDTH        = m["width"],
            VCTRL_VALS_SKILL = vctrl_skill,
            SPECTRE_START    = sim["spectre_freq_start"],
            SPECTRE_STOP     = sim["spectre_freq_stop"],
            SPECTRE_STEP     = sim["spectre_freq_step"],
        )
        output_path.write_text(rendered)

    # ------------------------------------------------------------------
    # simplified_pixel renderers
    # ------------------------------------------------------------------

    def _render_simplified_layout(
        self,
        cfg: dict,
        cell_name: str,
        output_path: Path,
    ) -> None:
        """Render the layout.il template for simplified_pixel mode."""
        template_text = (_TEMPLATES_DIR / "simplified_pixel/layout.il").read_text()
        la = cfg["layout"]
        rendered = string.Template(template_text).substitute(
            LIB_NAME    = cfg["project"]["lib_name"],
            CELL_NAME   = cell_name,
            PIXEL_SIZE  = la["pixel_size"],
            GND_LAYER   = la["gnd_layer"],
            PIXEL_LAYER = la["pixel_layer"],
        )
        output_path.write_text(rendered)

    def _render_simplified_sim(
        self,
        cfg: dict,
        cell_name: str,
        emx_dir: Path,
        output_path: Path,
    ) -> None:
        """Render the sim.il template for simplified_pixel mode."""
        template_text = (_TEMPLATES_DIR / "simplified_pixel/sim.il").read_text()
        sim = cfg["simulation"]
        p   = cfg["paths"]
        sw  = cfg["switch_model"]

        rendered = string.Template(template_text).substitute(
            LIB_NAME       = cfg["project"]["lib_name"],
            CELL_NAME      = cell_name,
            DEST_FOLDER    = str(emx_dir),
            EMX_BINARY     = p["emx_binary"],
            PROC_FILE      = p["proc_file"],
            EMX_FREQ_START = sim["emx_freq_start"],
            EMX_FREQ_STOP  = sim["emx_freq_stop"],
            EMX_FREQ_STEP  = sim["emx_freq_step"],
            EMX_PARALLEL   = sim["emx_parallel_threads"],
            SPECTRE_START  = sim["spectre_freq_start"],
            SPECTRE_STOP   = sim["spectre_freq_stop"],
            SPECTRE_STEP   = sim["spectre_freq_step"],
            CAP_VALUE      = sw["cap_value"],
            R_ON           = sw["r_on"],
            R_OFF          = sw["r_off"],
            C_OFF          = sw["c_off"],
        )
        output_path.write_text(rendered)

    # ------------------------------------------------------------------
    # pixelated_design_simple renderers
    # ------------------------------------------------------------------

    def _render_pixelated_simple_layout(
        self,
        cfg: dict,
        cell_name: str,
        output_path: Path,
    ) -> None:
        """Render layout.il for pixelated_design_simple mode."""
        pd = cfg["pixelated"]
        la = cfg["layout"]
        matrix = pd["pixel_state_matrix"]

        # Validate inputs before touching Virtuoso
        mat_info   = analyze_matrix(matrix)
        cap_values = pd.get("cap_values", [])
        state_list = pd.get("state_list", [])
        feedline   = pd.get("feedline", {})
        feedline_row = feedline.get("feedline_row", 0)

        if len(cap_values) != mat_info["num_sw"]:
            raise ValueError(
                f"cap_values length ({len(cap_values)}) must equal "
                f"number of switch pixels ({mat_info['num_sw']})."
            )
        if state_list:
            for i, sv in enumerate(state_list):
                if len(sv) != mat_info["num_sw"]:
                    raise ValueError(
                        f"state_list[{i}] length ({len(sv)}) must equal "
                        f"number of switch pixels ({mat_info['num_sw']})."
                    )
        if not (0 <= feedline_row < mat_info["num_rows"]):
            raise ValueError(
                f"feedline_row {feedline_row} out of range "
                f"[0, {mat_info['num_rows'] - 1}]."
            )

        template_text = (_TEMPLATES_DIR / "pixelated_design_simple/layout.il").read_text()

        rendered = string.Template(template_text).substitute(
            LIB_NAME          = cfg["project"]["lib_name"],
            CELL_NAME         = cell_name,
            PIXEL_SIZE        = la["pixel_size"],
            GRID_SIZE         = la.get("grid_size", la["pixel_size"]),
            GND_LAYER         = la["gnd_layer"],
            PIXEL_LAYER       = la["pixel_layer"],
            NUM_ROWS          = mat_info["num_rows"],
            NUM_COLS          = mat_info["num_cols"],
            PIXEL_TYPES_SKILL = build_skill_pixel_types_list(matrix),
            PORT_NAMES_SKILL  = build_skill_port_names_list(mat_info),
            FEEDLINE_ROW      = feedline_row,
            FEEDLINE_WIDTH    = feedline.get("feedline_width", 5.0),
            FEEDLINE_PADDING  = feedline.get("feedline_padding", 10.0),
            GND_Y_EXTENSION   = la.get("gnd_y_extension", 0.0),
        )
        output_path.write_text(rendered)

    def _render_pixelated_simple_emx(
        self,
        cfg: dict,
        cell_name: str,
        emx_dir: Path,
        output_path: Path,
    ) -> None:
        """Render emx.il for pixelated_design_simple mode (GDS export + EMX only)."""
        pd = cfg["pixelated"]
        sim = cfg["simulation"]
        p   = cfg["paths"]

        # Validate inductor s1p path if there are inductor pixels
        matrix   = pd["pixel_state_matrix"]
        mat_info = analyze_matrix(matrix)
        if mat_info["num_ind"] > 0:
            ind_s1p = pd.get("inductor_s1p", "")
            if not Path(ind_s1p).exists():
                raise FileNotFoundError(
                    f"inductor_s1p not found: {ind_s1p!r}\n"
                    "Check the 'pixelated.inductor_s1p' config key."
                )

        template_text = (_TEMPLATES_DIR / "pixelated_design_simple/emx.il").read_text()

        rendered = string.Template(template_text).substitute(
            LIB_NAME       = cfg["project"]["lib_name"],
            CELL_NAME      = cell_name,
            DEST_FOLDER    = str(emx_dir),
            EMX_BINARY     = p["emx_binary"],
            PROC_FILE      = p["proc_file"],
            EMX_FREQ_START = sim["emx_freq_start"],
            EMX_FREQ_STOP  = sim["emx_freq_stop"],
            EMX_FREQ_STEP  = sim["emx_freq_step"],
            EMX_PARALLEL   = sim["emx_parallel_threads"],
            EMX_PORT_FLAGS = build_emx_port_flags(mat_info),
            TOTAL_PORTS    = matrix_total_ports(mat_info),
        )
        output_path.write_text(rendered)

    def _run_pixelated_simple_states(
        self,
        cfg: dict,
        cell_name: str,
        emx_dir: Path,
    ) -> List[Path]:
        """
        Post-EMX driver for pixelated_design_simple.

        Finds the .snp file written by EMX, then calls spectre_runner.run_all_states()
        to generate per-state Spectre netlists, run Spectre, and collect .s2p outputs.
        """
        pd       = cfg["pixelated"]
        matrix   = pd["pixel_state_matrix"]
        mat_info = analyze_matrix(matrix)
        n_ports  = matrix_total_ports(mat_info)

        snp_path = emx_dir / f"{cell_name}.s{n_ports}p"
        if not snp_path.exists():
            print(f"  [engine] WARNING: EMX SNP file not found: {snp_path}")
            print("  [engine] Skipping Spectre simulation.")
            return []

        s2p_paths = run_all_states(
            snp_path   = snp_path,
            state_list = pd.get("state_list", []),
            cap_values = pd.get("cap_values", []),
            cfg        = cfg,
            emx_dir    = emx_dir,
            cell_name  = cell_name,
        )
        return [p for p in s2p_paths if p is not None]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_yaml(path: Path) -> dict:
        with path.open() as f:
            return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _cellname_safe(v: Any) -> str:
    """
    Convert a parameter value to a GDS/Virtuoso cell-name-safe string.

    GDS II cell names only allow [A-Za-z0-9_$?].  Scientific notation like
    "1e-12" is common in RF configs but the hyphen character is illegal —
    strmout silently mangles it when writing the GDS, so EMX can then no
    longer find the structure by name.

    Replacements applied (in order):
        '-'  → 'm'   (minus sign in exponents: 1e-12 → 1em12)
        '.'  → 'p'   (decimal point:  35.0 → 35p0)
        '+'  → ''    (explicit plus sign, rare)
    """
    return str(v).replace("-", "m").replace(".", "p").replace("+", "")


def _make_cell_name(cfg: dict) -> str:
    """
    Generate a unique, filesystem-safe cell name from the tunable parameters
    of the active simulation mode.

    layout_pixel             → pixel_nmos_w3u_l80n_nf4_miml10u_mimw10u
    simplified_pixel         → simplepix_ps35p0_cap1em12_ron3_roff260_coff8em14
    pixelated_design_simple  → pixsimple_r3c5_ps35p0_fr1
    """
    mode = cfg.get("mode", "layout_pixel")

    if mode == "simplified_pixel":
        la  = cfg["layout"]
        sw  = cfg["switch_model"]
        return (
            f"simplepix"
            f"_ps{_cellname_safe(la['pixel_size'])}"
            f"_cap{_cellname_safe(sw['cap_value'])}"
            f"_ron{_cellname_safe(sw['r_on'])}"
            f"_roff{_cellname_safe(sw['r_off'])}"
            f"_coff{_cellname_safe(sw['c_off'])}"
        )

    if mode == "pixelated_design_simple":
        pd  = cfg["pixelated"]
        la  = cfg["layout"]
        mat = pd["pixel_state_matrix"]
        rows = len(mat)
        cols = len(mat[0]) if rows else 0
        feedline_row = pd.get("feedline", {}).get("feedline_row", 0)
        grid_size    = la.get("grid_size", la["pixel_size"])
        return (
            f"pixsimple"
            f"_r{rows}c{cols}"
            f"_ps{_cellname_safe(la['pixel_size'])}"
            f"_gs{_cellname_safe(grid_size)}"
            f"_fr{feedline_row}"
        )

    # Default: layout_pixel
    n = cfg["nmos"]
    m = cfg["mim_cap"]
    return (
        f"pixel_nmos"
        f"_w{n['width']}"
        f"_l{n['length']}"
        f"_nf{n['fingers']}"
        f"_miml{m['length']}"
        f"_mimw{m['width']}"
    )


def _get_run_labels(cfg: dict) -> list:
    """
    Return the ordered labels for each simulation run / state.

    layout_pixel             → list of gate control voltages, e.g. [1.0, 0.0]
    simplified_pixel         → ["on", "off"]
    pixelated_design_simple  → ["state0", "state1", ...]
    """
    mode = cfg.get("mode", "layout_pixel")
    if mode == "simplified_pixel":
        return ["on", "off"]
    if mode == "pixelated_design_simple":
        n_states = len(cfg.get("pixelated", {}).get("state_list", []))
        return [f"state{i}" for i in range(n_states)]
    return cfg["simulation"]["control_voltages"]


def _find_s1p_files(emx_dir: Path, cell_name: str, n_runs: int) -> List[Path]:
    """Locate the .s1p output files written by Spectre (layout_pixel / simplified_pixel)."""
    files = []
    for idx in range(n_runs):
        p = emx_dir / f"{cell_name}_run{idx}.s1p"
        if p.exists():
            files.append(p)
        else:
            print(f"  [engine] WARNING: expected {p} not found.")
    return files


def _make_plots(
    mode: str,
    s_param_files: List[Path],
    labels: list,
    plot_dir: Path,
) -> dict:
    """Dispatch to the appropriate plotter based on mode."""
    if mode == "pixelated_design_simple":
        return plot_s2p_results(
            s2p_files=s_param_files,
            state_labels=labels,
            output_dir=plot_dir,
        )
    return plot_s11_results(
        s1p_files=s_param_files,
        voltage_labels=labels,
        output_dir=plot_dir,
    )


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into a deep copy of *base*."""
    result = copy.deepcopy(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def _build_nested(path: List[str], value: Any) -> dict:
    """Build a nested dict from a key path and a leaf value."""
    d: dict = {}
    ref = d
    for key in path[:-1]:
        ref[key] = {}
        ref = ref[key]
    ref[path[-1]] = value
    return d


def _print_result(stage: str, result: VirtuosoResult) -> None:
    status = "OK" if result.success else "FAILED"
    print(f"[run] {stage} → {status} ({result.elapsed_sec:.1f}s)")
    if not result.success:
        print(f"      error: {result.error_message}")
