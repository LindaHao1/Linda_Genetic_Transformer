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

  // Switch connections (state-dependent)
  //  ON : c_sw<i>(sw_<i> sw_mid<i>) + r_on<i>(sw_mid<i> 0)
  //  OFF: c_sw<i>(sw_<i> sw_mid<i>) + r_off<i>(sw_mid<i> 0) + c_off<i>(sw_mid<i> 0)

  p_in  (pix_in  0) port r=50 num=1
  p_out (pix_out 0) port r=50 num=2

  sp_sim sp ports=[p_in p_out] start=... stop=... step=...
         file=<output.s2p> datafmt=touchstone
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

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
    state_vector: List[int],
    cap_values: List[float],
    sw_indices_in_port_order: List[int],
    ind_indices_in_port_order: List[int],
    gnd_indices_in_port_order: List[int],
    r_on: float,
    r_off: float,
    c_off: float,
    inductor_s1p_list: List[str],
    sim_cfg: dict,
    output_s2p: Path,
    netlist_dir: Path,
    cap_pair_indices: Optional[List[tuple]] = None,
    cap_pair_values: Optional[List[float]] = None,
    inductor_state_vector: Optional[List[int]] = None,
) -> Path:
    """
    Write a Spectre .scs testbench for one switch state and return its path.

    Args:
        snp_path                  Path to the EMX .snp file.
        port_order                List of port names from parse_snp_port_order().
        state_index               Index of this state (0-based, used in filename).
        state_vector              Binary list: 1 = switch ON, 0 = switch OFF.
        cap_values                Per-switch series capacitance (same order as SW_*).
        sw_indices_in_port_order  Indices into port_order that are SW_* ports.
        ind_indices_in_port_order Indices into port_order that are IND_* ports.
        gnd_indices_in_port_order Indices into port_order that are GND ref ports
                                  (PIX_IN_GND, PIX_OUT_GND) — shorted to ground.
        r_on                      Switch ON resistance (ohm).
        r_off                     Switch OFF resistance (ohm).
        c_off                     Switch OFF capacitance (F).
        inductor_s1p_list         Per-inductor s1p file paths, in IND_* port order.
                                  Length must equal len(ind_indices_in_port_order).
        sim_cfg                   simulation config dict (spectre_freq_* keys).
        output_s2p                Path where Spectre should write the .s2p file.
        netlist_dir               Directory to write the .scs file into.
        cap_pair_indices          List of (port_idx_a, port_idx_b) tuples for each
                                  capacitor pair (CP_{i}A, CP_{i}B ports).
        cap_pair_values           Per-pair capacitance values (F), same order as
                                  cap_pair_indices.

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
    lines.append("simulator lang=spectre\n")

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

    # --- Inductor connections (state-dependent) ---
    if ind_indices_in_port_order:
        lines.append("// Inductor connections (state-dependent)")
        for local_idx, port_idx in enumerate(ind_indices_in_port_order):
            node = node_names[port_idx]
            is_on = True
            if inductor_state_vector is not None and local_idx < len(inductor_state_vector):
                is_on = bool(inductor_state_vector[local_idx])

            if is_on:
                s1p = inductor_s1p_list[local_idx] if local_idx < len(inductor_s1p_list) else ""
                lines.append(f"nport_ind{local_idx} ({node} 0) nport file=\"{s1p}\" N=1")
            else:
                lines.append(f"// IND_{local_idx}: OFF (open-circuit termination)")
                lines.append(f"r_open_ind{local_idx} ({node} 0) resistor r=1e12")
        lines.append("")

    # --- Switch connections (state-dependent) ---
    lines.append("// Switch connections")
    for local_sw_idx, port_idx in enumerate(sw_indices_in_port_order):
        node   = node_names[port_idx]
        sw_on  = bool(state_vector[local_sw_idx])
        cap_v  = cap_values[local_sw_idx]
        mid    = f"sw_mid{local_sw_idx}"

        if sw_on:
            lines.append(f"// SW_{local_sw_idx}: ON (c_sw → r_on → GND)")
            lines.append(f"c_sw{local_sw_idx} ({node} {mid}) capacitor c={cap_v}")
            lines.append(f"r_on{local_sw_idx} ({mid} 0) resistor r={r_on}")
        else:
            lines.append(f"// SW_{local_sw_idx}: OFF (c_sw → r_off || c_off → GND)")
            lines.append(f"c_sw{local_sw_idx} ({node} {mid}) capacitor c={cap_v}")
            lines.append(f"r_off{local_sw_idx} ({mid} 0) resistor r={r_off}")
            lines.append(f"c_off{local_sw_idx} ({mid} 0) capacitor c={c_off}")
    lines.append("")

    # --- Capacitor pair connections (state-independent) ---
    if cap_pair_indices:
        lines.append("// Capacitor pair connections (always connected)")
        for pair_idx, (idx_a, idx_b) in enumerate(cap_pair_indices):
            node_a = node_names[idx_a]
            node_b = node_names[idx_b]
            cap_v  = (cap_pair_values or [])[pair_idx]
            lines.append(f"c_pair{pair_idx} ({node_a} {node_b}) capacitor c={cap_v}")
        lines.append("")

    # --- I/O ports ---
    lines.append("// Input/output RF ports")
    lines.append("p_in  (pix_in  0) port r=50 num=1")
    lines.append("p_out (pix_out 0) port r=50 num=2\n")

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
    state_list: List[List[int]],
    cap_values: List[float],
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

    # --- Parse port order from SNP header ---
    print(f"  [spectre_runner] Parsing port order from {snp_path.name} ...")
    port_order = parse_snp_port_order(snp_path)
    print(f"  [spectre_runner] Port order: {port_order}")

    # --- Derive per-type port index lists ---
    sw_indices  = [i for i, name in enumerate(port_order) if name.startswith("SW_")]
    ind_indices = [i for i, name in enumerate(port_order) if name.startswith("IND_")]
    gnd_indices = [i for i, name in enumerate(port_order)
                   if name in ("PIX_IN_GND", "PIX_OUT_GND")]

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

    sw  = cfg.get("switch_model", {})
    sim = cfg["simulation"]
    pd  = cfg.get("pixelated", {})

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

    s2p_paths: List[Optional[Path]] = []

    for state_idx, state_vector in enumerate(state_list):
        # Determine inductor ON/OFF vector for this state (None = all-ON default)
        if inductor_state_list is not None and state_idx < len(inductor_state_list):
            inductor_state_vector = inductor_state_list[state_idx]
        else:
            inductor_state_vector = None

        print(f"  [spectre_runner] State {state_idx}: SW={state_vector}  IND={inductor_state_vector}")

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
            r_on                      = sw.get("r_on",  3),
            r_off                     = sw.get("r_off", 260),
            c_off                     = sw.get("c_off", "8e-14"),
            inductor_s1p_list         = inductor_s1p_list,
            sim_cfg                   = sim,
            output_s2p                = output_s2p,
            netlist_dir               = emx_dir,
            cap_pair_indices          = cap_pair_indices,
            cap_pair_values           = cap_pair_values,
            inductor_state_vector     = inductor_state_vector,
        )

        # --- Run Spectre ---
        spectre_log = psf_dir / "spectre.log"
        cmd = ["spectre", str(netlist_path), "-raw", str(psf_dir)]
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
