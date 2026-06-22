"""
spectre_runner.py — Post-EMX Spectre simulation driver for pixelated_design_simple.

This module handles everything after EMX completes:
  1. Parse EMX .snp file header to get the exact port order.
  2. For each switch state: generate a Spectre .scs testbench and run Spectre.
  3. Collect .s2p output paths.

The Spectre netlist structure per state
---------------------------------------
  nport1 (pix_in 0 pix_out 0 pix_in_gnd 0 pix_out_gnd 0 ind_0 0 ... sw_0 0 ...)
         nport file=<snp> N=<total>

  // Ground reference ports (PIX_IN_GND, PIX_OUT_GND) — shorted to global ground
  v_pix_in_gnd  (pix_in_gnd  0) vsource dc=0
  v_pix_out_gnd (pix_out_gnd 0) vsource dc=0

  // Inductor connections (same for every state)
  nport_ind0 (ind_0 0) nport file=<inductor.s1p> N=1
  ...

  // Switch connections (state-dependent — TSMC mimcap_um_sin_rf model)
  //  mimcap_sw<i>(sw_mid<i> sw_<i> 0) mimcap_um_sin_rf lt=...u wt=...u
  //  nport_sw<i>(sw_mid<i> 0) nport file=<on|off.s1p> N=1
  //  Topology: SW EMX port (mim_m) → mimcap → (mim_p) sw_mid → nport(s1p) → GND

  p_in  (pix_in  0) port r=<port1_impedance> num=1
  p_out (pix_out 0) port r=<port2_impedance> num=2
  // optional parasitic input shunt cap (simulation.input_shunt_cap):
  c_in_par (pix_in 0) capacitor c=<input_shunt_cap>

  sp_sim sp ports=[p_in p_out] start=... stop=... step=...
         file=<output.s2p> datafmt=touchstone

Port impedances and the input shunt capacitor are optional simulation-config
keys (port1_impedance, port2_impedance, input_shunt_cap).  When absent they
default to 50 ohm / 50 ohm / no capacitor — i.e. the original behaviour.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from .cap_utils import _lookup_cap_dims
from .server_launch import wrap_command

# ---------------------------------------------------------------------------
# Backward-compatibility normalization helpers
# ---------------------------------------------------------------------------

def _normalize_cap_values(cap_values: list) -> list:
    """Convert flat List[float] → List[List[float]] if needed."""
    if cap_values and isinstance(cap_values[0], (int, float)):
        return [[v] for v in cap_values]
    return cap_values


def _normalize_state_list(state_list: list) -> list:
    """Convert List[List[int]] → List[List[List[int]]] if needed."""
    if state_list and state_list[0] and isinstance(state_list[0][0], (int, float)):
        return [[[bit] for bit in sv] for sv in state_list]
    return state_list


# ---------------------------------------------------------------------------
# SNP port-order parser
# ---------------------------------------------------------------------------

_PORT_LINE_RE = re.compile(r"^!\s+(\d+)\s+(\S+)\s*$")


def parse_snp_port_order(snp_path: Path) -> List[str]:
    """
    Read the Touchstone SNP file header and return port names in port-number order.

    EMX writes lines like:
        ! Port order:
        ! 1 PIX_IN
        ! 2 PIX_OUT
        ! 3 IND_0
        ! 4 SW_0
    when called with --include-port-order.

    Returns a list where result[0] is the name of port 1, result[1] port 2, etc.

    Raises:
        FileNotFoundError  if snp_path does not exist.
        ValueError         if no port-order lines are found in the header.
    """
    snp_path = Path(snp_path)
    if not snp_path.exists():
        raise FileNotFoundError(f"EMX SNP file not found: {snp_path}")

    port_map: Dict[int, str] = {}
    with snp_path.open() as fh:
        for line in fh:
            line = line.rstrip("\n")
            # Stop at first non-comment data line
            if line and not line.startswith("!") and not line.startswith("#"):
                break
            m = _PORT_LINE_RE.match(line)
            if m:
                port_num  = int(m.group(1))
                port_name = m.group(2)
                port_map[port_num] = port_name

    if not port_map:
        raise ValueError(
            f"No port-order lines found in {snp_path}.\n"
            "Make sure EMX was called with --include-port-order."
        )

    # Build ordered list (1-indexed → 0-indexed)
    max_port = max(port_map)
    return [port_map[i + 1] for i in range(max_port)]


# ---------------------------------------------------------------------------
# Spectre netlist generator
# ---------------------------------------------------------------------------

def generate_state_netlist(
    snp_path: Path,
    port_order: List[str],
    state_index: int,
    state_vector: List[List[int]],
    cap_values: List[List[float]],
    sw_indices_in_port_order: List[int],
    ind_indices_in_port_order: List[int],
    gnd_indices_in_port_order: List[int],
    switch_on_s1p: str,
    switch_off_s1p: str,
    tsmc_models_path: str,
    sw_dims: List[List[tuple]],
    inductor_s1p_list: List[str],
    sim_cfg: dict,
    output_s2p: Path,
    netlist_dir: Path,
    cap_pair_indices: Optional[List[tuple]] = None,
    cap_pair_values: Optional[List[float]] = None,
    cp_dims: Optional[List[tuple]] = None,
    cap_pair_orientations: Optional[List[str]] = None,
    inductor_state_vector: Optional[List[int]] = None,
    via_indices_in_port_order: Optional[List[int]] = None,
    inductor_off_s1p_list: Optional[List[str]] = None,
    mimflag: int = 3,
    mismatchflag: int = 0,
    ind_dims: Optional[List[tuple]] = None,
    scap_indices_in_port_order: Optional[List[int]] = None,
    scap_cap_values: Optional[List[float]] = None,
    scap_dims: Optional[List[tuple]] = None,
) -> Path:
    """
    Write a Spectre .scs testbench for one switch state and return its path.

    Switch model: pixel SW port → (mim_m) mimcap_um_sin_rf (mim_p) → sw_mid → nport(s1p) → GND
                                                    ↓ um
                                                   GND

    Args:
        snp_path                  Path to the EMX .snp file.
        port_order                List of port names from parse_snp_port_order().
        state_index               Index of this state (0-based, used in filename).
        state_vector              List[List[int]]: outer = switch index, inner = on/off bits per cap in bank.
        cap_values                List[List[float]]: outer = switch index, inner = cap values per bank.
        sw_indices_in_port_order  Indices into port_order that are SW_* ports.
        ind_indices_in_port_order Indices into port_order that are IND_* ports.
        gnd_indices_in_port_order Indices into port_order that are GND ref ports
                                  (PIX_IN_GND, PIX_OUT_GND) — shorted to ground.
        switch_on_s1p             Path to the NMOS switch ON-state S1P file.
        switch_off_s1p            Path to the NMOS switch OFF-state S1P file.
        tsmc_models_path          Path to the TSMC Spectre models file (for rf_mim include).
        sw_dims                   List[List[tuple]]: per-switch, per-cap (lt_um, wt_um) from _lookup_cap_dims.
        inductor_s1p_list         Per-inductor ON-state s1p file paths (IND_* order).
        inductor_off_s1p_list     Per-inductor OFF-state s1p paths (IND_* order).
                                  When None or entry is empty, falls back to large
                                  resistor (1 TΩ) for that inductor's OFF state.
        sim_cfg                   simulation config dict (spectre_freq_* keys).
        output_s2p                Path where Spectre should write the .s2p file.
        netlist_dir               Directory to write the .scs file into.
        cap_pair_indices          List of (port_idx_a, port_idx_b) tuples for each
                                  capacitor pair (CP_{i}A, CP_{i}B ports).
        cap_pair_values           Per-pair capacitance values (F) — kept for reference.
        cp_dims                   Per-pair (lt_um, wt_um) from _lookup_cap_dims.
        cap_pair_orientations     Per-pair orientation: "H"/"H_rev" (horizontal) or
                                  "V"/"V_rev" (vertical). Controls which node is mim_p.
                                  Defaults to "H" (node_a = mim_p) when None.
        ind_dims                  Per-inductor (lt_um, wt_um) for the series MIM cap.
                                  Defaults to (10.0, 10.0) for each inductor when None.
        scap_indices_in_port_order Indices into port_order that are SCAP_* (shunt-cap) ports.
        scap_cap_values           Per-shunt-cap capacitance values (F), SCAP_* order.
        scap_dims                 Per-shunt-cap (lt_um, wt_um) from _lookup_cap_dims.

    Returns:
        Path to the written .scs netlist file.
    """
    total_ports = len(port_order)
    # Node names: lowercase port names, with '~' stripped (EMX auto-names)
    node_names = [p.lower().lstrip("~").replace("~", "_") for p in port_order]

    lines: List[str] = []

    # --- Header ---
    lines.append(f"// Pixelated Design Simple — state {state_index}")
    lines.append(f"// Switch state vector: {state_vector}")
    lines.append(f"// EMX SNP file: {snp_path}")
    lines.append("simulator lang=spectre")
    if tsmc_models_path:
        lines.append(f'include "{tsmc_models_path}" section=tt_rfmim')
    lines.append("")

    # --- EMX nport block ---
    # Node list: each port paired with node 0 (global ground)
    node_pairs = " ".join(f"{n} 0" for n in node_names)
    lines.append(f"// EM extraction result ({total_ports} ports)")
    lines.append(
        f"emx_blk ({node_pairs}) nport file=\"{snp_path}\" N={total_ports}\n"
    )

    # --- Ground reference ports (PIX_IN_GND, PIX_OUT_GND) ---
    # These ports are on the gnd_layer and are shorted to global ground.
    if gnd_indices_in_port_order:
        lines.append("// Ground reference ports (shorted to global ground)")
        for port_idx in gnd_indices_in_port_order:
            node = node_names[port_idx]
            lines.append(f"v_{node} ({node} 0) vsource dc=0")
        lines.append("")

    # --- Via connections (always shorted to GND) ---
    if via_indices_in_port_order:
        lines.append("// Via connections (shorted to global ground)")
        for local_idx, port_idx in enumerate(via_indices_in_port_order):
            node = node_names[port_idx]
            lines.append(f"v_via{local_idx} ({node} 0) vsource dc=0")
        lines.append("")

    # --- Shunt capacitor connections (state-independent, always present) ---
    # A type-5 pixel is a switch pixel whose switch was removed: the pixel port
    # is tied to ground through a MIM cap (no s1p, no ON/OFF state).
    # Topology: SCAP pixel port (mim_m) → mimcap_um_sin_rf → (mim_p) GND ; bulk → GND
    if scap_indices_in_port_order:
        lines.append("// Shunt capacitor connections (pixel → MIM cap → GND, always connected)")
        for local_idx, port_idx in enumerate(scap_indices_in_port_order):
            node = node_names[port_idx]
            lt, wt = (
                scap_dims[local_idx]
                if scap_dims and local_idx < len(scap_dims)
                else (10.0, 10.0)
            )
            cval_fF = (
                scap_cap_values[local_idx] * 1e15
                if scap_cap_values and local_idx < len(scap_cap_values)
                else 0.0
            )
            lines.append(f"// SCAP_{local_idx}: {cval_fF:.0f} fF (mim_m=pixel → mim_p=GND)")
            lines.append(
                f"mimcap_scap{local_idx} (0 {node} 0) mimcap_um_sin_rf "
                f"wt={wt}u lt={lt}u mimflag={mimflag} mismatchflag={mismatchflag}"
            )
        lines.append("")

    # --- Inductor connections (state-dependent) ---
    # Topology: IND port (mim_m) → mimcap_um_sin_rf → (mim_p) ind_mid → nport(s1p) → GND
    if ind_indices_in_port_order:
        lines.append("// Inductor connections (state-dependent; series MIM cap)")
        for local_idx, port_idx in enumerate(ind_indices_in_port_order):
            node = node_names[port_idx]
            is_on = True
            if inductor_state_vector is not None and local_idx < len(inductor_state_vector):
                is_on = bool(inductor_state_vector[local_idx])

            mid = f"ind_mid{local_idx}"
            lt, wt = (ind_dims[local_idx] if ind_dims and local_idx < len(ind_dims) else (10.0, 10.0))

            if is_on:
                s1p = inductor_s1p_list[local_idx] if local_idx < len(inductor_s1p_list) else ""
                lines.append(f"// IND_{local_idx}: ON (mimcap mim_m=pixel → mim_p=mid → inductor s1p)")
                lines.append(f"mimcap_ind{local_idx} ({mid} {node} 0) mimcap_um_sin_rf wt={wt}u lt={lt}u mimflag={mimflag} mismatchflag={mismatchflag}")
                lines.append(f"nport_ind{local_idx} ({mid} 0) nport file=\"{s1p}\" N=1")
            else:
                off_s1p = ""
                if inductor_off_s1p_list and local_idx < len(inductor_off_s1p_list):
                    off_s1p = inductor_off_s1p_list[local_idx] or ""
                if off_s1p:
                    lines.append(f"// IND_{local_idx}: OFF (mimcap → layout off s1p)")
                    lines.append(f"mimcap_ind{local_idx} ({mid} {node} 0) mimcap_um_sin_rf wt={wt}u lt={lt}u mimflag={mimflag} mismatchflag={mismatchflag}")
                    lines.append(f"nport_ind{local_idx} ({mid} 0) nport file=\"{off_s1p}\" N=1")
                else:
                    lines.append(f"// IND_{local_idx}: OFF (mimcap → open-circuit termination)")
                    lines.append(f"mimcap_ind{local_idx} ({mid} {node} 0) mimcap_um_sin_rf wt={wt}u lt={lt}u mimflag={mimflag} mismatchflag={mismatchflag}")
                    lines.append(f"r_open_ind{local_idx} ({mid} 0) resistor r=1e12")
        lines.append("")

    # --- Switch connections (state-dependent, capacitor-bank model) ---
    # Each switch pixel has a bank of N cap+switch pairs connected in parallel to the same
    # EMX port node. Topology per cap j in bank i:
    #   SW_i port (mim_m) → mimcap_um_sin_rf → (mim_p) sw_i_j_mid → nport(s1p) → GND
    lines.append("// Switch connections (capacitor bank: N parallel mimcap+nport per switch pixel)")
    for local_sw_idx, port_idx in enumerate(sw_indices_in_port_order):
        node       = node_names[port_idx]
        bank_states = state_vector[local_sw_idx]   # List[int]
        bank_dims   = sw_dims[local_sw_idx]        # List[tuple]
        lines.append(f"// SW_{local_sw_idx}: bank of {len(bank_states)} cap(s)")
        for j, (is_on, (lt, wt)) in enumerate(zip(bank_states, bank_dims)):
            mid = f"sw_{local_sw_idx}_{j}_mid"
            s1p = switch_on_s1p if is_on else switch_off_s1p
            state_label = "ON" if is_on else "OFF"
            lines.append(f"// SW_{local_sw_idx} cap {j}: {state_label} (mimcap mim_m=pixel → mim_p=mid → switch s1p)")
            lines.append(f"mimcap_sw{local_sw_idx}_{j} ({mid} {node} 0) mimcap_um_sin_rf wt={wt}u lt={lt}u mimflag={mimflag} mismatchflag={mismatchflag}")
            lines.append(f"nport_sw{local_sw_idx}_{j} ({mid} 0) nport file=\"{s1p}\" N=1")
    lines.append("")

    # --- Capacitor pair connections (state-independent) ---
    # Topology: mim_p pixel → mimcap_um_sin_rf → mim_m pixel; bulk (um) → GND
    # Orientation "H"/"V": node_a is mim_p; "H_rev"/"V_rev": node_b is mim_p.
    if cap_pair_indices:
        lines.append("// Capacitor pair connections (always connected)")
        for pair_idx, (idx_a, idx_b) in enumerate(cap_pair_indices):
            node_a = node_names[idx_a]
            node_b = node_names[idx_b]
            lt, wt = (cp_dims or [])[pair_idx] if cp_dims and pair_idx < len(cp_dims) else (10.0, 10.0)
            orientation = (cap_pair_orientations or [])[pair_idx] if cap_pair_orientations else "H"
            if orientation in ("H_rev", "V_rev"):
                node_a, node_b = node_b, node_a
            lines.append(f"mimcap_pair{pair_idx} ({node_a} {node_b} 0) mimcap_um_sin_rf wt={wt}u lt={lt}u mimflag={mimflag} mismatchflag={mismatchflag}")
        lines.append("")

    # --- I/O ports ---
    # Port reference impedances and an optional parasitic input shunt capacitor
    # are read from the simulation config.  Defaults reproduce the original
    # behaviour (both ports 50 ohm, no capacitor).
    r_in   = sim_cfg.get("port1_impedance", 50)
    r_out  = sim_cfg.get("port2_impedance", 50)
    shunt  = sim_cfg.get("input_shunt_cap", None)   # F (e.g. "122e-15"); None/0 → omit
    lines.append("// Input/output RF ports")
    lines.append(f"p_in  (pix_in  0) port r={r_in} num=1")
    lines.append(f"p_out (pix_out 0) port r={r_out} num=2")
    if shunt:
        lines.append("// Parasitic input shunt capacitor (port 1 to ground)")
        lines.append(f"c_in_par (pix_in 0) capacitor c={shunt}")
    lines.append("")

    # --- S-parameter analysis ---
    lines.append("// 2-port S-parameter sweep")
    lines.append(
        f"sp_sim sp ports=[p_in p_out]"
        f" start={sim_cfg['spectre_freq_start']}"
        f" stop={sim_cfg['spectre_freq_stop']}"
        f" step={sim_cfg['spectre_freq_step']}"
        f' file="{output_s2p}" datafmt=touchstone'
    )

    netlist_path = Path(netlist_dir) / f"tb_state{state_index}.scs"
    netlist_path.write_text("\n".join(lines) + "\n")
    return netlist_path


# ---------------------------------------------------------------------------
# Orchestrator: run all states
# ---------------------------------------------------------------------------

def run_all_states(
    snp_path: Path,
    state_list: List,
    cap_values: List,
    cfg: dict,
    emx_dir: Path,
    cell_name: str,
    cap_pair_values: Optional[List[float]] = None,
    inductor_state_list: Optional[List[List[int]]] = None,
) -> List[Optional[Path]]:
    """
    Post-EMX driver: parse SNP port order, generate netlists, run Spectre.

    Args:
        snp_path         Path to the EMX .snp result file.
        state_list       List of binary switch-state vectors.
        cap_values       Per-switch capacitances (same order as SW_* ports).
        cfg              Full merged engine config dict.
        emx_dir          Directory containing EMX results (netlists written here too).
        cell_name        Cell name prefix for output file naming.
        cap_pair_values  Per-pair capacitances (F) for CP_* ports; None or [] if unused.

    Returns:
        List of .s2p paths, one per state (None for any state where Spectre failed).
    """
    snp_path = Path(snp_path)
    emx_dir  = Path(emx_dir)

    # --- Normalize legacy flat formats to capacitor-bank format ---
    cap_values = _normalize_cap_values(cap_values)
    state_list = _normalize_state_list(state_list)

    # --- Parse port order from SNP header ---
    print(f"  [spectre_runner] Parsing port order from {snp_path.name} ...")
    port_order = parse_snp_port_order(snp_path)
    print(f"  [spectre_runner] Port order: {port_order}")

    # --- Derive per-type port index lists ---
    sw_indices  = [i for i, name in enumerate(port_order) if name.startswith("SW_")]
    ind_indices = [i for i, name in enumerate(port_order) if name.startswith("IND_")]
    gnd_indices = [i for i, name in enumerate(port_order)
                   if name in ("PIX_IN_GND", "PIX_OUT_GND")]
    via_indices = [i for i, name in enumerate(port_order) if name.startswith("VIA_")]
    scap_indices = [i for i, name in enumerate(port_order) if name.startswith("SCAP_")]

    print(f"  [spectre_runner] GND ports: {[port_order[i] for i in gnd_indices]}")

    # --- Derive cap pair index pairs from port_order ---
    cap_pair_values = cap_pair_values or []
    num_cap_pairs   = len(cap_pair_values)
    cap_pair_indices: List[tuple] = []
    for pair_idx in range(num_cap_pairs):
        name_a = f"CP_{pair_idx}A"
        name_b = f"CP_{pair_idx}B"
        idx_a  = next((i for i, n in enumerate(port_order) if n == name_a), None)
        idx_b  = next((i for i, n in enumerate(port_order) if n == name_b), None)
        if idx_a is not None and idx_b is not None:
            cap_pair_indices.append((idx_a, idx_b))
        else:
            print(
                f"  [spectre_runner] WARNING: cap pair {pair_idx} ports "
                f"({name_a}, {name_b}) not found in SNP — skipping pair."
            )

    # Sanity checks
    if len(sw_indices) != len(cap_values):
        raise ValueError(
            f"cap_values length ({len(cap_values)}) does not match "
            f"number of SW_* ports in SNP ({len(sw_indices)})."
        )
    if state_list and len(state_list[0]) != len(sw_indices):
        raise ValueError(
            f"state_list inner length ({len(state_list[0])}) does not match "
            f"number of SW_* ports ({len(sw_indices)})."
        )
    if state_list:
        for si, sv in enumerate(state_list):
            for sw_i, bank in enumerate(sv):
                if len(bank) != len(cap_values[sw_i]):
                    raise ValueError(
                        f"state_list[{si}][{sw_i}] bank length ({len(bank)}) does not match "
                        f"cap_values[{sw_i}] bank length ({len(cap_values[sw_i])})."
                    )

    sw  = cfg.get("switch_model", {})
    sim = cfg["simulation"]
    pd  = cfg.get("pixelated", {})
    fl  = cfg.get("full_layout", {})
    p   = cfg.get("paths", {})
    mc  = cfg.get("mim_cap", {})

    mimflag      = int(mc.get("mimflag",      3))
    mismatchflag = int(mc.get("mismatchflag", 0))

    switch_on_s1p  = sw.get("switch_on_s1p", "")
    switch_off_s1p = sw.get("switch_off_s1p", "")
    tsmc_models    = p.get("tsmc_models", "")

    # Resolve cap dimension tables (paths may be relative to repo root)
    _repo_root = Path(__file__).parent.parent
    sw_table = Path(fl.get("switch_cap_table", "virtuoso_engine/data/switch_cap_table.csv"))
    cp_table = Path(fl.get("cap_pair_table",   "virtuoso_engine/data/cap_pair_table.csv"))
    if not sw_table.is_absolute():
        sw_table = _repo_root / sw_table
    if not cp_table.is_absolute():
        cp_table = _repo_root / cp_table

    sw_dims = [_lookup_cap_dims(sw_table, cap_bank) for cap_bank in cap_values]
    cp_dims = _lookup_cap_dims(cp_table, cap_pair_values) if cap_pair_values else []

    # --- Shunt capacitor (type-5 pixel) values + MIM dims ---
    # One capacitance per SCAP_* port, in row-first order (same as shunt_cap_values).
    shunt_cap_values: List[float] = pd.get("shunt_cap_values", []) or []
    if len(shunt_cap_values) != len(scap_indices):
        raise ValueError(
            f"shunt_cap_values length ({len(shunt_cap_values)}) does not match "
            f"number of SCAP_* ports in SNP ({len(scap_indices)})."
        )
    scap_dims = _lookup_cap_dims(sw_table, shunt_cap_values) if shunt_cap_values else []
    if scap_indices:
        print(f"  [spectre_runner] Shunt-cap ports: {[port_order[i] for i in scap_indices]} "
              f"= {[f'{c*1e15:.0f}fF' for c in shunt_cap_values]}")

    inductor_cap_values_cfg: List[float] = pd.get("inductor_cap_values", [])
    ind_dims = _lookup_cap_dims(sw_table, inductor_cap_values_cfg) if inductor_cap_values_cfg else []

    # Infer cap pair orientations from pixel coordinates in config
    raw_pairs = pd.get("cap_pixel_pairs", [])
    cap_pair_orientations: List[str] = []
    for pair in raw_pairs:
        (r1, c1), (r2, c2) = pair
        if r1 == r2:
            cap_pair_orientations.append("H" if c1 <= c2 else "H_rev")
        else:
            cap_pair_orientations.append("V" if r1 <= r2 else "V_rev")

    # Build per-inductor s1p list from translation table + inductor_values indices.
    # Falls back to the legacy single inductor_s1p for all inductors when the
    # translation table or per-inductor indices are absent.
    translation_table: List[str] = pd.get("inductor_translation_table", [])
    inductor_value_indices: List[int] = pd.get("inductor_values", [])
    fallback_s1p: str = pd.get("inductor_s1p", "")

    if translation_table and inductor_value_indices:
        inductor_s1p_list = [
            translation_table[idx] for idx in inductor_value_indices
        ]
    else:
        # Legacy: all inductors share the same file; list will be indexed by
        # local_idx in generate_state_netlist — pad to len(ind_indices) lazily.
        inductor_s1p_list = [fallback_s1p] * max(len(ind_indices), 1)

    # Build per-inductor OFF-state s1p list.
    # Uses inductor_off_translation_table (same value indices as ON table) when
    # present; falls back to inductor_off_s1p (single path for all inductors);
    # falls back to None → large-resistor open-circuit (original behaviour).
    off_translation_table: List[str] = pd.get("inductor_off_translation_table", [])
    fallback_off_s1p: str = pd.get("inductor_off_s1p", "")

    if off_translation_table:
        if inductor_value_indices:
            inductor_off_s1p_list: Optional[List[str]] = [
                off_translation_table[idx] if idx < len(off_translation_table) else ""
                for idx in inductor_value_indices
            ]
        else:
            inductor_off_s1p_list = [off_translation_table[0]] * max(len(ind_indices), 1)
    elif fallback_off_s1p:
        inductor_off_s1p_list = [fallback_off_s1p] * max(len(ind_indices), 1)
    else:
        inductor_off_s1p_list = None

    s2p_paths: List[Optional[Path]] = []

    for state_idx, state_vector in enumerate(state_list):
        # Determine inductor ON/OFF vector for this state (None = all-ON default)
        if inductor_state_list is not None and state_idx < len(inductor_state_list):
            inductor_state_vector = inductor_state_list[state_idx]
        else:
            inductor_state_vector = None

        sw_group_strs = []
        for sw_i, bank in enumerate(state_vector):
            bits_str = ",".join("ON" if b else "OFF" for b in bank)
            if sw_i < len(cap_values):
                caps_str = ",".join(f"{c*1e15:.0f}fF" for c in cap_values[sw_i])
                sw_group_strs.append(f"SW{sw_i}([{bits_str}]|[{caps_str}])")
            else:
                sw_group_strs.append(f"SW{sw_i}([{bits_str}])")
        print(f"  [spectre_runner] State {state_idx}: {' '.join(sw_group_strs)}  IND={inductor_state_vector}")

        output_s2p  = emx_dir / f"{cell_name}_state{state_idx}.s2p"
        psf_dir     = emx_dir / f"psf_state{state_idx}"
        psf_dir.mkdir(exist_ok=True)

        netlist_path = generate_state_netlist(
            snp_path                  = snp_path,
            port_order                = port_order,
            state_index               = state_idx,
            state_vector              = state_vector,
            cap_values                = cap_values,
            sw_indices_in_port_order  = sw_indices,
            ind_indices_in_port_order = ind_indices,
            gnd_indices_in_port_order = gnd_indices,
            switch_on_s1p             = switch_on_s1p,
            switch_off_s1p            = switch_off_s1p,
            tsmc_models_path          = tsmc_models,
            sw_dims                   = sw_dims,
            inductor_s1p_list         = inductor_s1p_list,
            sim_cfg                   = sim,
            output_s2p                = output_s2p,
            netlist_dir               = emx_dir,
            cap_pair_indices          = cap_pair_indices,
            cap_pair_values           = cap_pair_values,
            cp_dims                   = cp_dims,
            cap_pair_orientations     = cap_pair_orientations,
            inductor_state_vector     = inductor_state_vector,
            via_indices_in_port_order = via_indices,
            inductor_off_s1p_list     = inductor_off_s1p_list,
            mimflag                   = mimflag,
            mismatchflag              = mismatchflag,
            ind_dims                  = ind_dims,
            scap_indices_in_port_order = scap_indices,
            scap_cap_values           = shunt_cap_values,
            scap_dims                 = scap_dims,
        )

        # --- Run Spectre ---
        spectre_log = psf_dir / "spectre.log"
        cmd = ["spectre", "-64", str(netlist_path), "-raw", str(psf_dir), "+aps", "+mt"]
        cmd = wrap_command(cmd, cfg.get("server"))
        print(f"  [spectre_runner] Executing: {' '.join(cmd)}")

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(emx_dir),
        )
        spectre_log.write_text(proc.stdout + proc.stderr)

        if proc.returncode != 0:
            print(f"  [spectre_runner] WARNING: Spectre returned {proc.returncode} for state {state_idx}")

        if output_s2p.exists():
            print(f"  [spectre_runner] State {state_idx} → {output_s2p.name}")
            s2p_paths.append(output_s2p)
        else:
            print(f"  [spectre_runner] WARNING: {output_s2p} not produced for state {state_idx}")
            s2p_paths.append(None)

    return s2p_paths


# ---------------------------------------------------------------------------
# Differential-mode filter testbench
# ---------------------------------------------------------------------------

def generate_differential_netlist(
    snp_path: Path,
    port_order: List[str],
    sim_cfg: dict,
    output_s2p: Path,
    netlist_dir: Path,
    tsmc_models_path: str = "",
    diff_cap: str = "61e-15",
    port1_impedance: float = 60,
    port2_impedance: float = 50,
    shunt_cap_values: Optional[List[float]] = None,
) -> Path:
    """
    Build the differential S-parameter testbench netlist.

    Topology (multi-layer update — differential output pair):
        * EMX block exposes PIX_IN_T, PIX_IN_B, PIX_OUT_T, PIX_OUT_B plus their
          ground refs.
        * A configurable cap (default 61 fF) bridges the two input nodes.
        * Port 1 is a *floating* differential port across the two input nodes
          (default 60 Ω).  Port 2 is a single-ended output port (default 50 Ω)
          on the TOP output (PIX_OUT_T); the bottom output (PIX_OUT_B) is shorted
          to global ground.
        * Ground-reference ports are shorted to global ground.
        * VIA_<i> ports shorted to ground; SCAP_<i> (shunt-cap) ports tied to
          ground through an ideal capacitor of value shunt_cap_values[i] (F),
          in row-first SCAP index order.

    Returns the written netlist path.
    """
    total_ports = len(port_order)
    node_names  = [p.lower().lstrip("~").replace("~", "_") for p in port_order]
    name_to_node = {p: node_names[i] for i, p in enumerate(port_order)}

    for required in ("PIX_IN_T", "PIX_IN_B", "PIX_OUT_T", "PIX_OUT_B"):
        if required not in name_to_node:
            raise ValueError(
                f"Differential SNP is missing port {required!r}. "
                f"Found ports: {port_order}"
            )

    n_in_t  = name_to_node["PIX_IN_T"]
    n_in_b  = name_to_node["PIX_IN_B"]
    n_out_t = name_to_node["PIX_OUT_T"]
    n_out_b = name_to_node["PIX_OUT_B"]
    gnd_ports = [p for p in port_order if p.endswith("_GND")]

    lines: List[str] = []
    lines.append("// Differential filter S-parameter testbench")
    lines.append(f"// EMX SNP file: {snp_path}")
    lines.append("simulator lang=spectre")
    if tsmc_models_path:
        lines.append(f'include "{tsmc_models_path}" section=tt_rfmim')
    lines.append("")

    node_pairs = " ".join(f"{n} 0" for n in node_names)
    lines.append(f"// EM extraction result ({total_ports} ports)")
    lines.append(f'emx_blk ({node_pairs}) nport file="{snp_path}" N={total_ports}\n')

    if gnd_ports:
        lines.append("// Ground reference ports (shorted to global ground)")
        for gp in gnd_ports:
            node = name_to_node[gp]
            lines.append(f"v_{node} ({node} 0) vsource dc=0")
        lines.append("")

    # Via (type-4) pixels: each VIA_<i> port is a short to global ground.
    via_ports = [p for p in port_order if p.startswith("VIA_")]
    if via_ports:
        lines.append("// Via connections (shorted to global ground)")
        for vp in via_ports:
            node = name_to_node[vp]
            lines.append(f"v_{node} ({node} 0) vsource dc=0")
        lines.append("")

    # Shunt-capacitor (type-5) pixels: each SCAP_<i> port is tied to ground
    # through an ideal capacitor whose value is the GA-tuned shunt_cap_values[i]
    # (row-first SCAP index order).  The pixel's metal layer (AP vs M9) is set in
    # the layout; here only the capacitance matters.
    scap_ports = sorted(
        (p for p in port_order if p.startswith("SCAP_")),
        key=lambda p: int(p.split("_")[1]),
    )
    if scap_ports:
        scv = shunt_cap_values or []
        if len(scv) != len(scap_ports):
            raise ValueError(
                f"shunt_cap_values length ({len(scv)}) does not match number of "
                f"SCAP_* ports ({len(scap_ports)}) in the differential SNP."
            )
        lines.append("// Shunt-capacitor connections (pixel → ideal cap → GND)")
        for i, sp in enumerate(scap_ports):
            node = name_to_node[sp]
            lines.append(f"c_{node} ({node} 0) capacitor c={scv[i]}")
        lines.append("")

    lines.append("// Differential input front end")
    lines.append(f"c_diff ({n_in_t} {n_in_b}) capacitor c={diff_cap}")
    lines.append(f"p_in  ({n_in_t} {n_in_b}) port r={port1_impedance} num=1")
    lines.append("")
    lines.append("// Differential output: read TOP, ground BOTTOM")
    lines.append(f"v_{n_out_b} ({n_out_b} 0) vsource dc=0")
    lines.append(f"p_out ({n_out_t} 0) port r={port2_impedance} num=2")
    lines.append("")

    lines.append("// 2-port S-parameter sweep")
    lines.append(
        f"sp_sim sp ports=[p_in p_out]"
        f" start={sim_cfg['spectre_freq_start']}"
        f" stop={sim_cfg['spectre_freq_stop']}"
        f" step={sim_cfg['spectre_freq_step']}"
        f' file="{output_s2p}" datafmt=touchstone'
    )

    netlist_path = Path(netlist_dir) / "tb_state0.scs"
    netlist_path.write_text("\n".join(lines) + "\n")
    return netlist_path


def run_all_states_differential(
    snp_path: Path,
    cfg: dict,
    emx_dir: Path,
    cell_name: str,
) -> List[Optional[Path]]:
    """
    Post-EMX driver for differential_filter mode.

    The 0/1-only differential filter is a single-state design: parse the SNP
    port order, build the differential testbench, run Spectre, return the .s2p.
    """
    snp_path = Path(snp_path)
    emx_dir  = Path(emx_dir)

    sim = cfg["simulation"]
    p   = cfg.get("paths", {})
    tsmc_models = p.get("tsmc_models", "")

    diff_cap = sim.get("diff_cap", "61e-15")
    r_in     = sim.get("port1_impedance", 60)
    r_out    = sim.get("port2_impedance", 50)
    shunt_cap_values = cfg.get("pixelated", {}).get("shunt_cap_values", []) or []

    print(f"  [spectre_runner] Parsing port order from {snp_path.name} ...")
    port_order = parse_snp_port_order(snp_path)
    print(f"  [spectre_runner] Port order: {port_order}")

    output_s2p = emx_dir / f"{cell_name}_state0.s2p"
    psf_dir    = emx_dir / "psf_state0"
    psf_dir.mkdir(exist_ok=True)

    netlist_path = generate_differential_netlist(
        snp_path        = snp_path,
        port_order      = port_order,
        sim_cfg         = sim,
        output_s2p      = output_s2p,
        netlist_dir     = emx_dir,
        tsmc_models_path = tsmc_models,
        diff_cap        = diff_cap,
        port1_impedance = r_in,
        port2_impedance = r_out,
        shunt_cap_values = shunt_cap_values,
    )

    spectre_log = psf_dir / "spectre.log"
    cmd = ["spectre", "-64", str(netlist_path), "-raw", str(psf_dir), "+aps", "+mt"]
    cmd = wrap_command(cmd, cfg.get("server"))
    print(f"  [spectre_runner] Executing: {' '.join(cmd)}")

    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(emx_dir))
    spectre_log.write_text(proc.stdout + proc.stderr)

    if proc.returncode != 0:
        print(f"  [spectre_runner] WARNING: Spectre returned {proc.returncode}")

    if output_s2p.exists():
        print(f"  [spectre_runner] Differential → {output_s2p.name}")
        return [output_s2p]
    print(f"  [spectre_runner] WARNING: {output_s2p} not produced")
    return [None]
