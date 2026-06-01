# GA for 3-State Bandpass — Core Algorithm & PV-Contrast Loss

---

## Slide 1 — GA Core Loop

- **Goal**: evolve a pixelated switchable filter that simultaneously realizes 3 bandpass states
  - chromosome = pixel matrix + switch states (per state) + cap / inductor values

- **Initialization**
  - sample $N_{\text{pop}}$ feasible random designs
  - evaluate every state of every design in parallel (Virtuoso)

- **Per-generation update**
  - **Elitism**: keep top $n_e = \lceil \alpha\,N_{\text{pop}}\rceil$ unchanged
  - **State champions**: keep best design *per state* $s \in \{1,2,3\}$
  - **Selection**: tournament of size $k$ on total cost
  - **Crossover** (prob. $p_c$): zig-zag / weighted-clustering / block
  - **Mutation** (prob. $p_m$): one random action (flip, switch toggle, cap nudge, …)
  - **Immigrants**: a few fresh random designs each generation

- **Fitness** (3-state bandpass, sum aggregation)

$$
\mathcal{L}_{\text{total}}(D) \;=\; \sum_{s=1}^{3} \mathcal{L}_{\text{PV},\,s}\bigl(S_{21}^{(s)}(f),\,S_{11}^{(s)}(f)\bigr)
$$

  - $S_{21}^{(s)}$, $S_{11}^{(s)}$ = simulated response of design $D$ in switch-state $s$
  - lower $\mathcal{L}_{\text{total}}$ ⇒ fitter

---

## Slide 2 — PV-Contrast Loss (per state)

- **Idea**: directly maximize passband-vs-stopband **separation**, not point-wise template error
  - passband region $\mathcal{P}_s$, stopband + cross-stopband region $\mathcal{S}_s$

- **Term 1 — contrast** (push passband up, worst stopband points down)

$$
\text{contrast} \;=\; \overline{S_{21}}\big|_{\mathcal{P}_s} \;-\; \frac{1}{k}\!\!\sum_{f \in \text{top-}k(\mathcal{S}_s)} \!\! S_{21}(f)
$$

  - top-$k$ = $k$ worst (highest) $S_{21}$ samples in $\mathcal{S}_s$
  - enters loss as $-\,\text{contrast}$ (maximize ⇒ minimize negative)

- **Term 2 — passband level** (anchor passband near target $T_{\text{pb}}$)

$$
\text{pb\_dev} \;=\; \max\!\bigl(0,\; T_{\text{pb}} - \overline{S_{21}}\big|_{\mathcal{P}_s}\bigr),
\qquad
\text{pb\_term} \;=\; \text{pb\_dev}^{\,p}
$$

  - one-sided: only penalizes when passband is too low
  - exponent $p$ (default 2) makes large shortfalls dominate

- **Term 3 — flatness** (optional, suppress passband ripple)

$$
\text{pb\_std} \;=\; \mathrm{std}\bigl(S_{21}(f)\bigr),\quad f \in \mathcal{P}_s
$$

- **Combined per-state PV loss**

$$
\mathcal{L}_{\text{PV},\,s} \;=\; w_s \Bigl[
  w_c\,(-\text{contrast}) \;+\; w_p\,\text{pb\_term} \;+\; w_f\,\text{pb\_std}
\Bigr] \;+\; \mathcal{L}_{S_{11},\,s}
$$

  - $w_c, w_p, w_f$ = contrast / passband / flatness weights
  - $w_s$ = per-state fusion weight
  - $\mathcal{L}_{S_{11},\,s}$ = standard worst-$k$ $S_{11}$ penalties (passband / stopband / cross)

- **Why PV instead of template matching?**
  - no rigid $S_{21}$ target ⇒ GA discovers *where* to place peaks
  - top-$k$ worst aggregation makes the loss robust to a few outlier frequencies
  - contrast term naturally separates the 3 bands without hand-tuning thresholds
