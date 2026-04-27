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
import shutil
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import csv

import yaml
from virtuoso_engine.config_loader import load_config

from .runner import PersistentVirtuosoRunner, VirtuosoResult
from .plotter import plot_s11_results, plot_s2p_results
from .pixel_matrix import (
    analyze_matrix,
    build_cap_pair_port_entries,
    build_emx_port_flags,
    build_skill_pixel_types_list,
    build_skill_port_names_list,
    get_feed_rows,
    total_ports as matrix_total_ports,
)
from .spectre_runner import run_all_states
from .cap_utils import _lookup_cap_dims

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
    "full_layout": {
        "layout": "full_layout/layout.il",
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

    def __init__(self, config_path: str | Path, worker_id: Optional[int] = None):
        self.config_path = Path(config_path)
        self.base_config = self._load_yaml(self.config_path)
        self._runner: Optional[PersistentVirtuosoRunner] = None
        self._worker_id  = worker_id
        self._lp = f"[W{worker_id}] " if worker_id is not None else ""
        print(f"[VirtuosoPixelEngine] Loaded config: {self.config_path}"
              + (f" (worker {worker_id})" if worker_id is not None else ""))

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

    def run(self, config_override: Optional[Dict | str | Path] = None) -> RunResult:
        """
        Execute one full layout + simulation run.

        Args:
            config_override: One of:
              - A nested dict deep-merged over the base config (only changed keys
                need to be provided).
              - A path (str or Path) to a YAML config file.  The file is loaded
                with ``load_config``, which resolves any ``_base_`` inheritance,
                and the fully-merged result is then deep-merged over the engine's
                base config.  This guarantees that values like ``layout.pixel_size``
                set in child configs are visible to all renderers.

        Returns:
            RunResult with success flags, log text, and plot paths.
        """
        if isinstance(config_override, (str, Path)):
            config_override = load_config(Path(config_override))
        cfg  = _deep_merge(self.base_config, config_override or {})
        if self._worker_id is not None:
            cfg = _apply_worker_config(cfg, self._worker_id)
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

        print(f"\n{self._lp}[run] Mode:   {mode}")
        print(f"{self._lp}[run] Cell:   {cell_name}")
        print(f"{self._lp}[run] Output: {run_dir}")

        # --- Ensure Virtuoso session is running ---
        runner = self._get_runner(cfg)

        # --- Render and run layout script ---
        layout_il  = run_dir / "generated_layout.il"
        layout_log = log_dir / "virtuoso_layout.log"
        self._render_layout(cfg, cell_name, layout_il)

        print(f"{self._lp}[run] Running Virtuoso layout generation ({mode})...")
        layout_result = runner.run_script(
            skill_file=layout_il,
            log_file=layout_log,
        )
        _print_result("layout", layout_result, self._lp)

        if not layout_result.success:
            print(f"{self._lp}[run] Layout failed — skipping simulation.")
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

        print(f"{self._lp}[run] Running Virtuoso EMX + Spectre simulation ({mode})...")
        sim_result = runner.run_script(
            skill_file=sim_il,
            log_file=sim_log,
        )
        _print_result("simulation", sim_result, self._lp)

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
                cfg=cfg,
            )
        else:
            print(f"{self._lp}[run] No S-parameter files found — skipping plots.")

        # --- Optional cleanup of intermediate files ---
        self._cleanup_run_dir(cfg, emx_dir, log_dir)

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
            runner_label = (
                f"VirtuosoRunner-W{self._worker_id}"
                if self._worker_id is not None
                else "VirtuosoRunner"
            )
            self._runner = PersistentVirtuosoRunner(
                virtuoso_dir=proj["virtuoso_dir"],
                display=display,
                session_log=log_dir / "virtuoso_session.log",
                label=runner_label,
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
        elif mode == "full_layout":
            self._render_full_layout(cfg, cell_name, output_path)
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
        elif mode == "full_layout":
            # EMX + Spectre not yet implemented for full_layout — emit a no-op script.
            output_path.write_text(
                '; full_layout: EMX/Spectre not yet implemented\n'
                'printf("full_layout sim step skipped (layout-only mode).\\n")\n'
            )
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

        # Validate cap pixel pairs
        cap_pixel_pairs = pd.get("cap_pixel_pairs", [])
        cap_pair_values = pd.get("cap_pair_values", [])
        if len(cap_pixel_pairs) != len(cap_pair_values):
            raise ValueError(
                f"cap_pixel_pairs length ({len(cap_pixel_pairs)}) must equal "
                f"cap_pair_values length ({len(cap_pair_values)})."
            )
        for i, pair in enumerate(cap_pixel_pairs):
            if len(pair) != 2:
                raise ValueError(
                    f"cap_pixel_pairs[{i}] must have exactly 2 entries, got {len(pair)}."
                )
            for row, col in pair:
                if not (0 <= row < mat_info["num_rows"] and 0 <= col < mat_info["num_cols"]):
                    raise ValueError(
                        f"cap_pixel_pairs[{i}] coordinate ({row}, {col}) is out of bounds "
                        f"for a {mat_info['num_rows']}×{mat_info['num_cols']} matrix."
                    )

        # Feed rows are read directly from col 0 and col N-1 of the matrix.
        feed_row_left, feed_row_right = get_feed_rows(matrix)

        grid_size = la.get("grid_size", la["pixel_size"])
        cap_port_data_skill = _build_cap_port_data_skill(
            cap_pixel_pairs, mat_info["num_rows"], grid_size
        )

        template_text = (_TEMPLATES_DIR / "pixelated_design_simple/layout.il").read_text()

        rendered = string.Template(template_text).substitute(
            LIB_NAME            = cfg["project"]["lib_name"],
            CELL_NAME           = cell_name,
            PIXEL_SIZE          = la["pixel_size"],
            GRID_SIZE           = grid_size,
            GND_LAYER           = la["gnd_layer"],
            PIXEL_LAYER         = la["pixel_layer"],
            NUM_ROWS            = mat_info["num_rows"],
            NUM_COLS            = mat_info["num_cols"],
            PIXEL_TYPES_SKILL   = build_skill_pixel_types_list(matrix),
            PORT_NAMES_SKILL    = build_skill_port_names_list(mat_info),
            FEEDLINE_ROW_LEFT   = feed_row_left,
            FEEDLINE_ROW_RIGHT  = feed_row_right,
            FEEDLINE_WIDTH      = feedline.get("feedline_width", 5.0),
            FEEDLINE_PADDING    = feedline.get("feedline_padding", 10.0),
            GND_Y_EXTENSION     = la.get("gnd_y_extension", 0.0),
            CAP_PORT_DATA_SKILL = cap_port_data_skill,
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

        # Validate inductor s1p path(s) if there are inductor pixels
        matrix   = pd["pixel_state_matrix"]
        mat_info = analyze_matrix(matrix)
        if mat_info["num_ind"] > 0:
            table: list = pd.get("inductor_translation_table", [])
            ind_vals: list = pd.get("inductor_values", [])
            if table and ind_vals:
                # Per-inductor validation via translation table
                for idx in ind_vals:
                    if idx < 0 or idx >= len(table):
                        raise ValueError(
                            f"inductor_values entry {idx} is out of range for "
                            f"inductor_translation_table (length {len(table)})."
                        )
                    s1p = table[idx]
                    if not Path(s1p).exists():
                        raise FileNotFoundError(
                            f"inductor_translation_table[{idx}] not found: {s1p!r}\n"
                            "Check the 'pixelated.inductor_translation_table' config key."
                        )
            else:
                # Legacy single-file fallback
                ind_s1p = pd.get("inductor_s1p", "")
                if not Path(ind_s1p).exists():
                    raise FileNotFoundError(
                        f"inductor_s1p not found: {ind_s1p!r}\n"
                        "Check the 'pixelated.inductor_s1p' config key."
                    )

        cap_pixel_pairs = pd.get("cap_pixel_pairs", [])
        cap_pair_entries = build_cap_pair_port_entries(cap_pixel_pairs)

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
            EMX_PORT_FLAGS = build_emx_port_flags(mat_info, cap_pair_entries),
            TOTAL_PORTS    = matrix_total_ports(mat_info, len(cap_pixel_pairs)),
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
        pd              = cfg["pixelated"]
        matrix          = pd["pixel_state_matrix"]
        mat_info        = analyze_matrix(matrix)
        cap_pixel_pairs = pd.get("cap_pixel_pairs", [])
        n_ports         = matrix_total_ports(mat_info, len(cap_pixel_pairs))

        snp_path = emx_dir / f"{cell_name}.s{n_ports}p"
        if not snp_path.exists():
            print(f"  [engine] WARNING: EMX SNP file not found: {snp_path}")
            print("  [engine] Skipping Spectre simulation.")
            return []

        s2p_paths = run_all_states(
            snp_path             = snp_path,
            state_list           = pd.get("state_list", []),
            cap_values           = pd.get("cap_values", []),
            cfg                  = cfg,
            emx_dir              = emx_dir,
            cell_name            = cell_name,
            cap_pair_values      = pd.get("cap_pair_values", []),
            inductor_state_list  = pd.get("inductor_switch_states"),
        )
        return [p for p in s2p_paths if p is not None]

    # ------------------------------------------------------------------
    # full_layout renderer
    # ------------------------------------------------------------------

    def _render_full_layout(
        self,
        cfg: dict,
        cell_name: str,
        output_path: Path,
    ) -> None:
        """Render layout.il for full_layout mode.

        Reads the full_layout config section, resolves layer names, looks up
        cap dimensions from CSV tables, and substitutes all parameters into
        the SKILL template.
        """
        fl  = cfg["full_layout"]
        pd  = cfg["pixelated"]
        la  = cfg["layout"]
        n   = cfg["nmos"]
        m   = cfg["mim_cap"]

        matrix      = pd["pixel_state_matrix"]
        mat_info    = analyze_matrix(matrix)
        num_rows    = mat_info["num_rows"]
        num_cols    = mat_info["num_cols"]

        pixel_size  = la["pixel_size"]
        grid_size   = la.get("grid_size", pixel_size)

        # --- Layer resolution: full_layout.layers overrides layout.* ---
        layers_cfg      = fl.get("layers", {})
        gnd_layer       = layers_cfg.get("gnd_layer")   or la["gnd_layer"]
        pixel_layer     = layers_cfg.get("pixel_layer") or la["pixel_layer"]
        cap_pair_route  = layers_cfg.get("cap_pair_route") or pixel_layer
        drain_via_bot, drain_via_top = (layers_cfg.get("switch_drain_via") or ["M2", "M8"])
        bulk_via_bot,  bulk_via_top  = (layers_cfg.get("switch_bulk_via")  or ["M1", "M4"])
        mim_via_bot,   mim_via_top   = (layers_cfg.get("switch_mim_via")   or ["M8", "M9"])
        via_stack_bot, via_stack_top = (layers_cfg.get("via_pixel_stack")  or ["M4", "AP"])

        # --- Resolve table paths relative to config or repo root ---
        repo_root = Path(__file__).parent.parent
        sw_table_path  = _resolve_path(fl["switch_cap_table"], repo_root)
        cap_table_path = _resolve_path(fl["cap_pair_table"],   repo_root)

        # --- Switch cap dims: one (length, width) per switch pixel (row-major order) ---
        sw_cap_values = pd.get("cap_values", [])
        if len(sw_cap_values) != mat_info["num_sw"]:
            raise ValueError(
                f"pixelated.cap_values length ({len(sw_cap_values)}) must equal "
                f"number of switch pixels ({mat_info['num_sw']})."
            )
        sw_dims = _lookup_cap_dims(sw_table_path, sw_cap_values)

        # --- Cap pair dims and orientations ---
        cap_pairs  = pd.get("cap_pixel_pairs", [])
        cap_values = pd.get("cap_pair_values", [])
        if len(cap_pairs) != len(cap_values):
            raise ValueError(
                f"pixelated.cap_pixel_pairs length ({len(cap_pairs)}) must equal "
                f"cap_pair_values length ({len(cap_values)})."
            )
        cp_dims   = _lookup_cap_dims(cap_table_path, cap_values)
        cp_orient = []
        for pair in cap_pairs:
            (r1, c1), (r2, c2) = pair
            if r1 == r2:
                cp_orient.append("H")
            elif c1 == c2:
                cp_orient.append("V")
            else:
                raise ValueError(
                    f"Cap pair {pair} is neither horizontal nor vertical. "
                    "Only axis-aligned pairs are supported."
                )

        # --- Feedline: row derived from matrix (same as pixelated_design_simple) ---
        feed_row_left, feed_row_right = get_feed_rows(matrix)
        feedline      = pd.get("feedline", {})
        feedline_w    = feedline.get("feedline_width", 10.2)
        feedline_pad  = feedline.get("feedline_padding", 10.0)

        # --- Build SKILL array literals ---
        sw_cap_dims_skill = _build_skill_list_of_pairs(sw_dims)
        cp_dims_skill     = _build_skill_list_of_pairs(cp_dims)
        cp_orient_skill   = _build_skill_string_list(cp_orient)
        cp_pixels_skill   = _build_skill_cap_pair_pixels(cap_pairs)

        # Flat pixel-types list in row-major order
        pixel_types_flat  = [ptype for row in matrix for ptype in row]
        pixel_types_skill = "list(" + " ".join(str(t) for t in pixel_types_flat) + ")"

        template_text = (_TEMPLATES_DIR / "full_layout/layout.il").read_text()

        rendered = string.Template(template_text).substitute(
            LIB_NAME               = fl.get("lib_name", cfg["project"]["lib_name"]),
            CELL_NAME              = cell_name,
            PIXEL_SIZE             = pixel_size,
            GRID_SIZE              = grid_size,
            GND_LAYER              = gnd_layer,
            PIXEL_LAYER            = pixel_layer,
            SWITCH_DRAIN_VIA_BOT   = drain_via_bot,
            SWITCH_DRAIN_VIA_TOP   = drain_via_top,
            SWITCH_BULK_VIA_BOT    = bulk_via_bot,
            SWITCH_BULK_VIA_TOP    = bulk_via_top,
            SWITCH_MIM_VIA_BOT     = mim_via_bot,
            SWITCH_MIM_VIA_TOP     = mim_via_top,
            CAP_PAIR_ROUTE_LAYER   = cap_pair_route,
            VIA_PIXEL_STACK_BOT    = via_stack_bot,
            VIA_PIXEL_STACK_TOP    = via_stack_top,
            NUM_ROWS               = num_rows,
            NUM_COLS               = num_cols,
            PIXEL_TYPES_SKILL      = pixel_types_skill,
            SWITCH_CAP_DIMS_SKILL  = sw_cap_dims_skill,
            NMOS_WIDTH             = n["width"],
            NMOS_LENGTH            = n["length"],
            NMOS_FINGERS           = n["fingers"],
            PDK_LIB                = cfg["project"]["pdk_lib"],
            NMOS_CELL              = n["cell_layout"],
            MIM_CELL_LAYOUT        = m["cell_layout"],
            INDUCTOR_LIB           = fl.get("inductor_lib", ""),
            INDUCTOR_CELL          = fl.get("inductor_cell", ""),
            INDUCTOR_OFFSET_X      = fl.get("inductor_offset", [0.0, 0.0])[0],
            INDUCTOR_OFFSET_Y      = fl.get("inductor_offset", [0.0, 0.0])[1],
            CAP_PAIR_WIRE_WIDTH    = fl.get("cap_pair_wire_width", 0.5),
            CAP_PAIR_COUNT         = len(cap_pairs),
            CAP_PAIR_PIXELS_SKILL  = cp_pixels_skill,
            CAP_PAIR_DIMS_SKILL    = cp_dims_skill,
            CAP_PAIR_ORIENT_SKILL  = cp_orient_skill,
            VIA_SIZE               = fl.get("via_size", la.get("via_size", 3.8)),
            GND_Y_EXTENSION        = la.get("gnd_y_extension", 0.0),
            FEEDLINE_ROW_LEFT      = feed_row_left,
            FEEDLINE_ROW_RIGHT     = feed_row_right,
            FEEDLINE_WIDTH         = feedline_w,
            FEEDLINE_PADDING       = feedline_pad,
        )
        output_path.write_text(rendered)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _cleanup_run_dir(self, cfg: dict, emx_dir: Path, log_dir: Path) -> None:
        """
        Delete intermediate simulation files if ``cleanup.delete_intermediate_files``
        is enabled in the config.

        What is deleted:
          - ``logs/`` directory entirely (Virtuoso/EMX log files).
          - Everything in ``emx_results/`` **except** ``*.s2p`` files.
            This removes the EMX multi-port SNP, Spectre netlists (.scs),
            Spectre PSF directories, and any other EMX scratch output.

        What is kept:
          - ``emx_results/*.s2p`` — the final per-state S-parameter files
            used by GA/SA optimisers.
          - All other run_dir files (config.yaml, generated_layout.il,
            generated_sim.il, plots/).
        """
        if not cfg.get("cleanup", {}).get("delete_intermediate_files", False):
            return

        # Delete logs directory
        if log_dir.exists():
            shutil.rmtree(log_dir, ignore_errors=True)
            print(f"{self._lp}[cleanup] Deleted {log_dir}")

        # In emx_results: delete everything except *.s2p files
        if emx_dir.exists():
            deleted = []
            for item in list(emx_dir.iterdir()):
                if item.is_file() and item.suffix.lower() == ".s2p":
                    continue  # keep final s2p outputs
                if item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
                    deleted.append(item.name + "/")
                else:
                    item.unlink(missing_ok=True)
                    deleted.append(item.name)
            if deleted:
                print(f"{self._lp}[cleanup] emx_results: removed {deleted}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_yaml(path: Path) -> dict:
        return load_config(path)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _resolve_path(path_str: str, repo_root: Path) -> Path:
    """Resolve a path that may be relative (to repo root) or absolute."""
    p = Path(path_str)
    return p if p.is_absolute() else (repo_root / p)


def _build_skill_list_of_pairs(pairs: List[tuple]) -> str:
    """Build a SKILL list of quoted-string (length width) pairs with 'u' unit suffix.

    dbCreateParamInst expects string-typed params like list("lt" "string" "12u").
    Values from the CSV table are floats (um); appending 'u' matches what the
    single-pixel layout_pixel mode passes via mim_cap.length / mim_cap.width.
    """
    if not pairs:
        return "list()"
    inner = " ".join(f'list("{l}u" "{w}u")' for l, w in pairs)
    return f"list({inner})"


def _build_skill_string_list(strings: List[str]) -> str:
    """Build a SKILL list of quoted strings: list("H" "V" ...)"""
    if not strings:
        return "list()"
    inner = " ".join(f'"{s}"' for s in strings)
    return f"list({inner})"


def _build_skill_cap_pair_pixels(
    cap_pairs: List[List[List[int]]],
) -> str:
    """Build a SKILL list of (r1 c1 r2 c2) per cap pair."""
    if not cap_pairs:
        return "list()"
    entries = []
    for pair in cap_pairs:
        (r1, c1), (r2, c2) = pair
        entries.append(f"list({r1} {c1} {r2} {c2})")
    return "list(" + " ".join(entries) + ")"


def _build_cap_port_data_skill(
    cap_pixel_pairs: List[List[List[int]]],
    num_rows: int,
    grid_size: float,
) -> str:
    """
    Build the SKILL list literal for the CAP_PORT_DATA_SKILL template variable.

    Returns a SKILL expression like:
        list(
            list(30.0 50.0 "CP_0A")
            list(50.0 30.0 "CP_0B")
        )
    or "list()" when cap_pixel_pairs is empty.

    The y-flip formula matches the layout template:
        center_x = col * grid_size + grid_size / 2
        center_y = (num_rows - 1 - row) * grid_size + grid_size / 2
    """
    if not cap_pixel_pairs:
        return "list()"
    entries = []
    for i, pair in enumerate(cap_pixel_pairs):
        for side, (row, col) in zip(["A", "B"], pair):
            cx = col * grid_size + grid_size / 2.0
            cy = (num_rows - 1 - row) * grid_size + grid_size / 2.0
            entries.append(f'list({cx} {cy} "CP_{i}{side}")')
    return "list(\n    " + "\n    ".join(entries) + "\n)"


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
    mode   = cfg.get("mode", "layout_pixel")
    suffix = f"_w{cfg['_worker_id']}" if cfg.get("_worker_id") is not None else ""

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
        ) + suffix

    if mode == "pixelated_design_simple":
        pd  = cfg["pixelated"]
        la  = cfg["layout"]
        mat = pd["pixel_state_matrix"]
        rows = len(mat)
        cols = len(mat[0]) if rows else 0
        grid_size = la.get("grid_size", la["pixel_size"])
        frl, frr  = get_feed_rows(mat)
        return (
            f"pixsimple"
            f"_r{rows}c{cols}"
            f"_ps{_cellname_safe(la['pixel_size'])}"
            f"_gs{_cellname_safe(grid_size)}"
            f"_frl{frl}frr{frr}"
        ) + suffix

    if mode == "full_layout":
        return cfg["full_layout"]["cell_name"] + suffix

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
    ) + suffix


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
    if mode == "full_layout":
        return []  # No Spectre states in layout-only mode
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
    cfg: Optional[Dict[str, Any]] = None,
) -> dict:
    """Dispatch to the appropriate plotter based on mode."""
    if mode == "pixelated_design_simple":
        matrix = None
        cap_pixel_pairs = None
        if cfg is not None:
            pd = cfg.get("pixelated", {})
            matrix = pd.get("pixel_state_matrix")
            cap_pixel_pairs = pd.get("cap_pixel_pairs") or None
        return plot_s2p_results(
            s2p_files=s_param_files,
            state_labels=labels,
            output_dir=plot_dir,
            pixel_matrix=matrix,
            cap_pixel_pairs=cap_pixel_pairs,
        )
    return plot_s11_results(
        s1p_files=s_param_files,
        voltage_labels=labels,
        output_dir=plot_dir,
    )


def _apply_worker_config(cfg: dict, worker_id: int) -> dict:
    """
    Return a deep copy of *cfg* with worker-specific isolation applied.

    Each worker uses:
      - ``project.lib_name``         →  ``{base}_w{worker_id}``
      - ``project.parent_output_dir`` → ``{base}/worker_{worker_id}``
      - ``_worker_id``                  stored for cell-name suffix generation
    """
    cfg = copy.deepcopy(cfg)
    proj = cfg["project"]
    proj["lib_name"]           = f"{proj['lib_name']}_w{worker_id}"
    proj["parent_output_dir"]  = f"{proj['parent_output_dir']}/worker_{worker_id}"
    cfg["_worker_id"]          = worker_id
    return cfg


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


def _print_result(stage: str, result: VirtuosoResult, prefix: str = "") -> None:
    status = "OK" if result.success else "FAILED"
    print(f"{prefix}[run] {stage} → {status} ({result.elapsed_sec:.1f}s)")
    if not result.success:
        print(f"{prefix}      error: {result.error_message}")
