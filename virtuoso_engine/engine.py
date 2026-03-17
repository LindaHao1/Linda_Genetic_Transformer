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

Adding New Modes
----------------
1. Add a new entry to ``_MODE_TEMPLATES`` mapping the mode name to a
   (layout_template, sim_template) tuple.
2. Implement ``_render_<mode>_layout`` and ``_render_<mode>_sim`` methods
   on ``VirtuosoPixelEngine``.
3. Register the render methods in ``_render_layout`` and ``_render_sim``
   dispatch blocks.
4. Implement ``_make_cell_name_<mode>`` and add a branch to
   ``_make_cell_name``.
5. Add a branch to ``_get_run_labels`` for the new mode's output labels.

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
from .plotter import plot_s11_results

# Absolute path to the templates directory (next to this file)
_TEMPLATES_DIR = Path(__file__).parent / "templates"

# ---------------------------------------------------------------------------
# Mode registry
# Maps mode name → (layout_template_filename, sim_template_filename)
# Add new modes here as the engine grows.
# ---------------------------------------------------------------------------
_MODE_TEMPLATES: Dict[str, tuple] = {
    "layout_pixel":     ("layout.il",            "sim.il"),
    "simplified_pixel": ("simplified_layout.il", "simplified_sim.il"),
}


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    cell_name:     str
    run_dir:       Path
    layout_result: VirtuosoResult
    sim_result:    VirtuosoResult
    s1p_files:     List[Path] = field(default_factory=list)
    plot_paths:    dict       = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.layout_result.success and self.sim_result.success


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class VirtuosoPixelEngine:
    """
    Parametric driver for single-pixel Virtuoso layout + EMX/Spectre simulation.

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

        # --- Render and run simulation script ---
        sim_il  = run_dir / "generated_sim.il"
        sim_log = log_dir / "virtuoso_sim.log"
        self._render_sim(cfg, cell_name, emx_dir, sim_il)

        print(f"[run] Running Virtuoso EMX + Spectre simulation ({mode})...")
        sim_result = runner.run_script(
            skill_file=sim_il,
            log_file=sim_log,
        )
        _print_result("simulation", sim_result)

        # --- Collect .s1p files and plot ---
        run_labels = _get_run_labels(cfg)
        s1p_files  = _find_s1p_files(emx_dir, cell_name, len(run_labels))

        plot_paths: dict = {}
        if s1p_files:
            plot_paths = plot_s11_results(
                s1p_files=s1p_files,
                voltage_labels=run_labels,
                output_dir=plot_dir,
            )
        else:
            print("[run] No .s1p files found — skipping plots.")

        return RunResult(
            cell_name=cell_name,
            run_dir=run_dir,
            layout_result=layout_result,
            sim_result=sim_result,
            s1p_files=s1p_files,
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

            # simplified_pixel sweep
            results = engine.sweep_configs([
                {"mode": "simplified_pixel", "switch_model": {"r_on": 2}},
                {"mode": "simplified_pixel", "switch_model": {"r_on": 5}},
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
        else:
            raise ValueError(f"No layout renderer for mode {mode!r}")

    def _render_sim(
        self,
        cfg: dict,
        cell_name: str,
        emx_dir: Path,
        output_path: Path,
    ) -> None:
        """Dispatch to the mode-appropriate simulation renderer."""
        mode = cfg.get("mode", "layout_pixel")
        if mode == "layout_pixel":
            self._render_layout_pixel_sim(cfg, cell_name, emx_dir, output_path)
        elif mode == "simplified_pixel":
            self._render_simplified_sim(cfg, cell_name, emx_dir, output_path)
        else:
            raise ValueError(f"No sim renderer for mode {mode!r}")

    # ------------------------------------------------------------------
    # layout_pixel renderers (original mode)
    # ------------------------------------------------------------------

    def _render_layout_pixel(
        self,
        cfg: dict,
        cell_name: str,
        output_path: Path,
    ) -> None:
        """Render the full layout.il template for layout_pixel mode."""
        template_text = (_TEMPLATES_DIR / "layout.il").read_text()
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
        template_text = (_TEMPLATES_DIR / "sim.il").read_text()
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
    # simplified_pixel renderers (new mode)
    # ------------------------------------------------------------------

    def _render_simplified_layout(
        self,
        cfg: dict,
        cell_name: str,
        output_path: Path,
    ) -> None:
        """Render the simplified_layout.il template for simplified_pixel mode."""
        template_text = (_TEMPLATES_DIR / "simplified_layout.il").read_text()
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
        """Render the simplified_sim.il template for simplified_pixel mode."""
        template_text = (_TEMPLATES_DIR / "simplified_sim.il").read_text()
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

    layout_pixel    → pixel_nmos_w3u_l80n_nf4_miml10u_mimw10u
    simplified_pixel → simplepix_ps35p0_cap1em12_ron3_roff260_coff8em14
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
    Return the ordered labels for each Spectre simulation run.

    layout_pixel    → list of gate control voltages, e.g. [1.0, 0.0]
    simplified_pixel → ["on", "off"]
    """
    mode = cfg.get("mode", "layout_pixel")
    if mode == "simplified_pixel":
        return ["on", "off"]
    return cfg["simulation"]["control_voltages"]


def _find_s1p_files(emx_dir: Path, cell_name: str, n_runs: int) -> List[Path]:
    """Locate the .s1p output files written by Spectre."""
    files = []
    for idx in range(n_runs):
        p = emx_dir / f"{cell_name}_run{idx}.s1p"
        if p.exists():
            files.append(p)
        else:
            print(f"  [engine] WARNING: expected {p} not found.")
    return files


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
