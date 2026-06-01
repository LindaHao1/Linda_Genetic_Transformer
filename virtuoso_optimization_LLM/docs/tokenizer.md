# Tokenizer

Source: [`src/tokenizer.py`](../src/tokenizer.py). Inspect any design with
[`scripts/inspect_tokens.py`](../scripts/inspect_tokens.py).

The tokenizer turns one filter **design** (the `record` dict produced by
`Design.to_record()`) into a flat integer sequence the LM trains on, and back.
Its hard contract: `decode(encode(record))` reproduces the design (structure
exactly; tunable values within one quantization bin) **and** the decoded dict
is exactly what `world_model_test/encode.py:encode_sample` consumes, so a
generated design round-trips straight into the surrogate.

The conditioning **spec is NOT tokenized** — it is a continuous PV-score vector
fed to the model as a prefix (see `architecture.md`). `SEP_spec` is only a
boundary marker; no spec tokens appear in the stream.

## Sequence layout

```
BOS  SEP_spec
<grid: rows*cols cells in raster row-major; per cell:
   - GROUND / METAL / VIA / PORT                 (1 token)
   - SWITCH         VAL_b                        (2 tokens, INLINE)
   - INDUCTOR       INDVAL_i  VAL_b              (3 tokens, INLINE)
   - CAP_PAIR_{H|V}{L}  VAL_b                    (2 tokens, head cell only)
   - interior/dest cells of an already-started pair emit
     GROUND / METAL respectively (regular pixel tokens)>
SEP_grid
<states: K * (n_sw + n_ind) STATE_*; state-major, switches raster then inductors raster>
SEP_states
EOS
```

- **Grid**: exactly `rows*cols` cells in row-major order. Each cell consumes
  one pixel-family token plus its inline VAL/INDVAL (if applicable). The raster
  cursor advances by **one cell per pixel-family token**; inline VAL/INDVAL
  share the cell of the preceding pixel.
- **Cap pairs** are emitted as **one pixel-family token at the head cell**
  (`CAP_PAIR_{H|V}{L}`), immediately followed by a single `VAL_b` for its
  capacitance. The interior cells along the pair's path are forced to GROUND
  and the dest cell to METAL/PORT by the constraint mask as the raster walk
  continues; for vertical pairs the dest reaches into a later row.
- **States**: `K * (n_sw + n_ind)` binary tokens, state-major; within a state,
  switches in raster order then inductors.

## Token dictionary

IDs are assigned in the order below (contiguous ranges). **Sizes/IDs depend on
config** (`n_bins`, `n_inductor_values`, cap-pair length range);
the table shows the **default config** (21×27, `n_bins=64`,
`n_inductor_values=1`, lengths 2–3) → **vocab_size = 86**. The authoritative
id↔name map is the saved JSON (`Tokenizer.save`), not these numbers.

| family (`get_token_type`) | tokens | count | default IDs | meaning |
|---|---|---|---|---|
| `structural` | `BOS, EOS, PAD, SEP_spec, SEP_grid, SEP_states` | 6 | 0–5 | sequence framing & segment boundaries; `PAD` fills batches (loss-masked) |
| `pixel` | `GROUND, METAL, INDUCTOR, SWITCH, VIA, PORT` | 6 | 6–11 | one per grid cell. Codes 0–4 = ground/metal/inductor/switch/via. `PORT` = a metal pixel in a feed column (col 0 or col W−1); **decodes back to metal (code 1)** but is a distinct token so feed-port location is learnable / position-lockable |
| `pixel_cap_pair` | `CAP_PAIR_H{L}, CAP_PAIR_V{L}` for `L` in `[cap_pair_min_dist .. cap_pair_max_dist]` | 4 | 12–15 | head pixel of a cap pair, names orientation + Chebyshev length. Matrix code at the head is METAL; the pair geometry (dest + interior) is recovered from the orient + length |
| `value` | `VAL_0 … VAL_{n_bins-1}` | 64 | 16–79 | a quantized tunable value. The *physical family* (switch cap / inductor MIM cap / cap-pair cap) is inferred from the preceding pixel token and de-quantized with that family's range |
| `indval` | `INDVAL_0 … INDVAL_{n_inductor_values-1}` | 1 | 80 | categorical inductor-model index (one per inductor pixel) |
| `state` | `STATE_OFF, STATE_ON` | 2 | 81–82 | per-(state, component) on/off bit |

`get_token_type(id)` returns the family string above — used for the
per-token-type loss weighting/diagnostics in `src/training.py`.

## Value (de)quantization

Each `VAL_*` bin maps to a physical range `[lo, hi]` selected by position via
`Tokenizer.value_range(family)`:

| value family | range (config keys) | default |
|---|---|---|
| `switch` (switch series cap) | `cap_min … cap_max` | 100–1000 fF |
| `inductor_cap` (MIM cap) | `inductor_cap_min … inductor_cap_max` | 200–1000 fF |
| `cap_pair` (shunt cap) | `cap_pair_min … cap_pair_max` | 50–1000 fF |

```
encode:  b = clip( int( (v - lo) / (hi - lo) * n_bins ), 0, n_bins-1 )
decode:  v = lo + (b + 0.5) / n_bins * (hi - lo)        # bin centre
```

So a decoded value is the bin **centre**; round-trip error is ≤ one bin width
`(hi-lo)/n_bins`. Larger `tokenizer.n_bins` = finer values, bigger vocab.

## In-grid cap pairs

A capacitor pair is emitted as a **single pixel-family token at the head cell**
(combined orient + length: `CAP_PAIR_H2`, `CAP_PAIR_H3`, `CAP_PAIR_V2`,
`CAP_PAIR_V3` for default `lengths=[2,3]`), immediately followed by one
`VAL_b` for its capacitance.

- **head** = the endpoint with the smaller raster index (`r*cols+c`).
- **orient** = `H` if the two endpoints share a row, `V` if they share a
  column. Pairs must be axis-aligned — a diagonal pair makes `encode` raise.
- **length** = Chebyshev distance, must lie in
  `[cap_pair_min_dist, cap_pair_max_dist]`.
- **interior / dest**: `dest = (r, c+L)` if `H` else `(r+L, c)`; the `L−1`
  interior cells along the path must be GROUND in the matrix, and the dest must
  be METAL. `encode` validates and raises if a pair violates these rules.
- **decode**: when a `CAP_PAIR_{O}{L}` token is consumed at raster cell
  `(r, c)`, the matrix entry is set to METAL and the pair geometry
  (`[[head, dest]]`, value from the next `VAL` bin) is recorded. The matching
  interior GROUND / dest METAL cells are filled in later by the raster walk
  consuming their regular pixel tokens (the constraint mask enforces this).

Constrained decoding (`src/constraints.py`) only lets the model pick a
`CAP_PAIR_*` head whose path interior is all-ground (in the current raster
state — interior cells must still be unwritten and not conflict-forced by a
prior pair) and whose destination cell will be METAL. Pair endpoints and
interior cells may not land on feed columns (see below).

## Feed-column rule

The first column (`c=0`, input feed) and the last column (`c=cols-1`, output
feed) carry a **single feed pixel each**, marked as `PORT` (a metal-coded
pixel-family token distinct from `METAL`). The constraint mask enforces:

- Only `GROUND` or `PORT` may be emitted in a feed column.
- At most one `PORT` per feed column.
- If the raster cursor reaches the last cell of a feed column and no `PORT`
  has been emitted in that column yet, `PORT` is the only allowed token
  (every design ends up with exactly one feed per side).
- Cap-pair heads, interiors, and destinations may not land on feed columns.

Matches the convention in `virtuoso_engine/pixel_matrix.py:get_feed_rows`,
which locates the feedline by finding the unique non-zero row in each feed
column.

## State codes

For each state `s` in `0..K-1`: emit `STATE_ON/OFF` for every switch pixel in
raster order, then for every inductor pixel in raster order, from
`switch_states[s]` / `inductor_switch_states[s]`. An empty
`inductor_switch_states` is treated as all-ON at encode time. Total state
tokens = `K * (n_sw + n_ind)`.

## Persistence

`Tokenizer.save(path)` writes `{config, tokens}` JSON; `Tokenizer.load(path)`
rebuilds it and asserts the vocab list is identical (guards against config
drift). Always ship the JSON with a checkpoint — token IDs are config-derived,
so the JSON is the only authoritative id↔name mapping.

## Inspecting

```bash
PY=/disk/mcao53/conda_env/rfic2/bin/python
$PY scripts/inspect_tokens.py --config configs/default.yaml                # synthetic, instant
$PY scripts/inspect_tokens.py --config configs/default.yaml --source sims --index 0
```

Prints the round-trip check (`structure exact` / `re-encode stable`), family
counts, the full token stream grouped by segment, the grid as a character map,
and the decoded values / cap-pairs / state codes.
