# AML Agent Methodology

## Abstract

This appendix describes the acute myeloid leukemia (AML) whole-slide-image
(WSI) agent used for morphology-based slide triage. The system is designed as a
retrieval-guided, morphology-first agent for bone marrow slide triage. Rather
than presenting an entire gigapixel WSI directly to a vision-language model
(VLM), the pipeline first constructs a compact candidate map using deterministic
stain and texture heuristics, foundation-model embeddings, and curated
ROI-quality references. The VLM is then restricted to a bounded microscopy-like
workflow in which it opens candidate fields, selects interpretable high-power
regions of interest (ROIs), and produces a strict morphology-only AML report.

The central methodological claim is that AML detection in this setting is better
posed as agentic evidence acquisition than as direct whole-slide classification.
The backend supplies efficient, auditable priors over candidate tissue regions;
the VLM performs local morphology interpretation and final diagnostic synthesis
from a small number of accepted ROIs.

## Scope And Output

The agent performs morphology-based triage for bone marrow WSI analysis. Its
final output is a constrained report containing per-ROI blast estimates, a
global blast range, a binary decision selected from `Normal marrow` or
`Acute leukemia`, confidence and limitations, and an NPM1 morphology prediction
only when acute leukemia is morphologically established.

The system does not use molecular, flow cytometry, clinical, or laboratory
metadata. The reference tile bank is used as an ROI-quality prior, not as a
disease-label classifier. Consequently, the final AML decision remains an
ROI-based morphology judgment rather than a direct nearest-neighbor label.

## Method Overview

The AML agent follows a cascade of increasingly semantic operations:

```text
WSI input
  -> metadata and microns-per-pixel resolution
  -> foreground supertile scan
  -> optional coarse thumbnail screening
  -> fixed-physical-size tile extraction
  -> deterministic tile quality filtering
  -> foundation-model embedding extraction
  -> curated ROI-quality retrieval
  -> per-view top-K candidate ranking
  -> bounded VLM navigation and ROI marking
  -> morphology-only AML decision report
```

This cascade is intentionally asymmetric. High-throughput slide reduction is
performed by deterministic and embedding-based stages that produce a compact,
reviewable candidate set. The VLM is reserved for the operations that require
visual interpretation: selecting readable cellular ROIs, estimating blast
burden, and synthesizing the final morphology report.

## Notation

| Symbol | Definition |
| --- | --- |
| $S$ | input WSI |
| $m_{\mathrm{pp}}$ | slide resolution in microns per pixel |
| $\theta$ | selected foundation-model feature extractor |
| $\mathcal{T}$ | set of foreground supertiles |
| $\mathcal{P} = \{p_i\}_{i=1}^{n}$ | set of candidate tiles |
| $z_i = \mathrm{norm}_2(f_\theta(p_i))$ | L2-normalized embedding for tile $p_i$ |
| $\mathcal{R}^{+}$ | curated good ROI-quality reference tiles |
| $\mathcal{R}^{-}$ | curated bad ROI-quality reference tiles when enabled |
| $D_i$ | dark/cellularity score for tile $p_i$ |
| $Q_i$ | deterministic tile quality score for $p_i$ |
| $G_i$ | aggregated similarity from $p_i$ to good references |
| $N_i$ | aggregated similarity from $p_i$ to bad references |
| $M_i = G_i - N_i$ | good-minus-bad reference margin |
| $B_i$ | bad-like likelihood inferred from $M_i$ |
| $v$ | current WSI view |
| $\mathcal{C}_v$ | ranked candidate list visible in view $v$ |
| $\mathcal{A}$ | accepted ROI set |
| $K$ | requested number of candidates |
| $N_{\mathrm{target}}$ | target number of accepted ROIs |
| $N_{\max}$ | hard cap on accepted ROIs |

## Algorithmic Specification

### Algorithm 1: AML Agent

**Input:** WSI $S$ and configuration $\Omega$.

**Output:** constrained AML report $\hat{Y}$.

1. Initialize the viewer state:
   $\mathcal{S}_0 \gets \mathrm{ResetState}(S,\Omega)$.

2. Load the embedding extractor and slide scale:
   $\theta \gets \mathrm{LoadExtractor}(\Omega_{\mathrm{extractor}})$ and
   $m_{\mathrm{pp}} \gets \mathrm{ResolveMPP}(S,\Omega)$.

3. Acquire the whole-slide overview:
   $v_0 \gets \mathrm{Overview}(S)$.

4. Construct the ROI candidate index:
   $\mathcal{I} \gets
   \mathrm{EnsureROIIndex}(S,\theta,m_{\mathrm{pp}},\Omega)$.

5. Retrieve the initial candidate list:
   $\mathcal{C}_{v_0} \gets
   \mathrm{RankCandidates}(\mathcal{I},\mathrm{bbox}(v_0),K)$.

6. Initialize accepted and attempted sets:
   $\mathcal{A} \gets \varnothing$ and
   $\mathcal{U} \gets \varnothing$.

7. While
   $|\mathcal{A}| < N_{\mathrm{target}}$ and
   $|\mathcal{A}| < N_{\max}$:

   a. Select the highest-ranked unattempted candidate:
      $c^\star \gets
      \mathrm{NextBest}(\mathcal{C}_{v},\mathcal{U},\mathcal{A})$.

   b. If $c^\star = \varnothing$, compute fallback candidates
      $\mathcal{C}_{v}^{\mathrm{fb}}$ from the current view and repeat the
      selection. Terminate if no candidate is available.

   c. Open the candidate field:
      $v_c \gets \mathrm{OpenCandidate}
      (c^\star,\Omega_{\mathrm{field}})$.

   d. Ask the VLM to inspect $v_c$ and propose a local ROI:
      $\tilde{r} \gets
      \mathrm{VLMInspect}(v_c,c^\star,\Omega)$.

   e. If $\tilde{r} = \varnothing$, update
      $\mathcal{U} \gets \mathcal{U} \cup \{c^\star\}$ and refresh
      $\mathcal{C}_{v}$.

   f. Otherwise, mark the normalized ROI:
      $r \gets \mathrm{MarkROI}(\tilde{r})$.

   g. If $r$ is duplicate, artifact-flooded, or otherwise unusable, reject or
      discard it, update $\mathcal{U}$, and refresh $\mathcal{C}_{v}$.

   h. If $r$ is accepted, update
      $\mathcal{A} \gets \mathcal{A} \cup \{r\}$.

8. Return the final report:
   $\hat{Y} \gets \mathrm{FinalOutput}(\mathcal{A},\Omega)$.

Algorithm 1 formalizes the interactive agent loop. Candidate regions are treated
as navigation priors rather than final ROIs. The VLM must inspect each opened
field and select the most interpretable local ROI before it can contribute to
the final evidence set.

### Algorithm 2: ROI Index Construction

**Input:** WSI $S$, extractor $\theta$, scale $m_{\mathrm{pp}}$, and
configuration $\Omega$.

**Output:** ROI index $\mathcal{I}$.

1. Select foreground supertiles:
   $\mathcal{T} \gets
   \mathrm{ForegroundSupertiles}(S,m_{\mathrm{pp}},\Omega)$.

2. For each $t \in \mathcal{T}$, split $t$ into tiles
   $\mathcal{P}_t$, remove low-texture and edge-dominated tiles, and apply the
   quality filter when $\Omega_{\mathrm{filter}}\in\{\mathrm{quality},
   \mathrm{hybrid}\}$.

3. Aggregate all retained tiles:
   $\mathcal{P} \gets \bigcup_{t\in\mathcal{T}} \mathcal{P}_t$.

4. Extract normalized embeddings:
   $Z \gets \{z_i = \mathrm{norm}_2(f_{\theta}(p_i)) :
   p_i \in \mathcal{P}\}$.

5. Compute dark/cellularity scores:
   $D \gets \{D_i : p_i \in \mathcal{P}\}$.

6. Embed curated reference banks:
   $Z^{+} \gets \{\mathrm{norm}_2(f_{\theta}(r)) :
   r \in \mathcal{R}^{+}\}$ and
   $Z^{-} \gets \{\mathrm{norm}_2(f_{\theta}(r)) :
   r \in \mathcal{R}^{-}\}$.

7. Compute retrieval evidence:
   $(G,N,M,B,h) \gets
   \mathrm{ReferenceRetrieval}(Z,Z^{+},Z^{-},\Omega)$.

8. Construct the candidate index:

$$
\mathcal{I} =
\{(p_i, x_i, y_i, z_i, D_i, G_i, N_i, M_i, B_i, h_i)\}_{i=1}^{|\mathcal{P}|}.
$$

The ROI index summarizes where the slide contains promising, readable marrow
fields. It combines tile coordinates, dark/cellularity scores, reference
retrieval evidence, and quality hints so the VLM can focus on interpretable
regions instead of scanning the whole slide exhaustively.

### Algorithm 3: Tile Quality Function

**Input:** tile set $\mathcal{P}$ and configuration $\Omega$.

**Output:** quality-filtered tile set $\mathcal{P}_{\mathrm{selected}}$.

1. For every $p_i \in \mathcal{P}$, compute the feature vector
   $x_i = \mathrm{TileMetrics}(p_i)$.

2. Define the hard-retention indicator:

$$
H_i =
\mathbb{1}\left[
F^{(i)}_{\mathrm{tissue}} \geq \tau_{\mathrm{tissue}}
\land C^{(i)}_{\mathrm{cell}} = 1
\land F^{(i)}_{\mathrm{rbc}} \leq \tau_{\mathrm{rbc}}
\land A^{(i)}_{\mathrm{artifact}} \leq \tau_{\mathrm{artifact}}
\land A^{(i)}_{\mathrm{gray/black}} \leq \tau_{\mathrm{gray/black}}
\land E^{(i)}_{\mathrm{edge}} = 0
\land A^{(i)}_{\mathrm{stringy}} = 0
\right].
$$

3. Score retained tiles with the deterministic quality function:
   $Q_i \gets Q(x_i)$ for all $i$ such that $H_i=1$.

4. Form the retained pool:
   $\mathcal{P}_{\mathrm{pool}} =
   \{p_i \in \mathcal{P}: H_i=1\}$.

5. Select the top-quality subset and diversity reserve:

$$
\mathcal{P}_{\mathrm{selected}} =
\mathrm{TopK}(\mathcal{P}_{\mathrm{pool}},Q,k_{\mathrm{keep}})
\cup
\mathrm{Reserve}(\mathcal{P}_{\mathrm{pool}},r_{\mathrm{reserve}}).
$$

The quality function rejects non-informative or artifactual image patches before
embedding. A small reserve from the lower-ranked pool is retained to preserve
spatial and morphologic diversity.

### Algorithm 4: Reference Retrieval

**Input:** query embeddings $Z$, good-reference embeddings $Z^{+}$,
bad-reference embeddings $Z^{-}$, and configuration $\Omega$.

**Output:** retrieval tuple $(G,N,M,B,h)$.

For each query tile embedding $z_i \in Z$:

1. Retrieve the top-$k$ good and bad neighbors:

$$
\mathcal{N}^{+}_i =
\mathrm{TopK}_{z \in Z^{+}}\mathrm{sim}(z_i,z), \quad
\mathcal{N}^{-}_i =
\mathrm{TopK}_{z \in Z^{-}}\mathrm{sim}(z_i,z).
$$

2. Aggregate neighbor support:

$$
G_i = \mathrm{Agg}(\mathcal{N}^{+}_i), \quad
N_i = \mathrm{Agg}(\mathcal{N}^{-}_i).
$$

3. Compute reference margin and bad-like likelihood:

$$
M_i = G_i - N_i, \quad
B_i = \sigma(-\lambda M_i).
$$

4. Assign the quality hint:

$$
h_i = \mathrm{Hint}(M_i,B_i,g_i^{(1)},n_i^{(1)}).
$$

### Algorithm 5: Candidate Ranking

**Input:** ROI index $\mathcal{I}$, view bounding box $b_v$, and requested
candidate count $K$.

**Output:** ranked view-local candidate list $\mathcal{C}_v$.

1. Select indexed tiles inside the current view:

$$
\mathcal{P}_v =
\{p_i \in \mathcal{I}: \mathrm{center}(p_i) \in b_v\}.
$$

2. Apply the AML dark/cellularity floor:

$$
\mathcal{P}_v \gets
\{p_i \in \mathcal{P}_v : D_i \geq \tau_{\mathrm{dark}}\}.
$$

3. For every $p_i \in \mathcal{P}_v$, compute:

```math
\begin{aligned}
\Delta_i &= g_i^{(1)} - n_i^{(1)}, \\
P_i^{\mathrm{qual}} &= w_M M_i + w_{\Delta}\Delta_i, \\
L_i^{\mathrm{bad}} &=
  \operatorname{clip}_{[0,1]}
  \left(\frac{B_i-b_0}{1-b_0}\right), \\
L_i^{\mathrm{match}} &=
  \operatorname{clip}_{[0,1]}
  \left(\frac{n_i^{(1)}-g_i^{(1)}+\delta}{\alpha}\right), \\
L_i^{\mathrm{qual}} &=
  w_{\mathrm{bad}}L_i^{\mathrm{bad}}
  + w_{\mathrm{match}}L_i^{\mathrm{match}}, \\
S_i^{\mathrm{cand}} &=
  w_D z(D_i)
  + w_G z(G_i)
  + w_P z(P_i^{\mathrm{qual}})
  - w_L L_i^{\mathrm{qual}}
  + w_S \mathbf{1}[\mathrm{StrongGood}_i]
  + w_R \mathbf{1}[\mathrm{DarkRegion}_i].
\end{aligned}
```

4. Sort candidates by $S^{\mathrm{cand}}_i$ in descending order.

5. Remove duplicate, edge-touching, and low-quality regions; suppress
   bad-like candidates when non-bad candidates exist.

6. Return:

$$
\mathcal{C}_v =
\mathrm{TopK}(\mathrm{SortDesc}(\mathcal{P}_v,S^{\mathrm{cand}}),K).
$$

### Algorithm 6: Final Diagnostic Synthesis

**Input:** accepted ROI set $\mathcal{A}=\{r_j\}_{j=1}^{m}$ and prompt rules
$\Pi$.

**Output:** final constrained report $\hat{Y}$.

1. For each accepted ROI $r_j$, estimate the interpretable nucleated cell count
   $n_j$ and blast-like cell count $b_j$.

2. Compute the ROI blast fraction:

$$
\hat{b}_j = \frac{b_j}{\max(n_j,1)}.
$$

3. Map each $\hat{b}_j$ into the required tier:

$$
\tau_j \in \{<5\%, 5\mathrm{-}9\%, 10\mathrm{-}19\%, 20\mathrm{-}50\%, >50\%\}.
$$

4. Synthesize global blast burden:

$$
\hat{b}_{\mathrm{global}} =
\mathrm{Synthesize}(\{\tau_j\}_{j=1}^{m},\mathcal{A}).
$$

5. Assign the binary morphology decision:

$$
\hat{y} =
\begin{cases}
\mathrm{AcuteLeukemia}, &
  \hat{b}_{\mathrm{global}} \geq 0.20
  \land \mathrm{DiffuseImmaturity}(\mathcal{A}), \\
\mathrm{NormalMarrow}, & \mathrm{otherwise}.
\end{cases}
$$

6. Set:

$$
\mathrm{NPM1Applicable} =
\mathbb{1}[\hat{y}=\mathrm{AcuteLeukemia}].
$$

7. Return $\hat{Y}$ as a constrained output containing ROI-level morphology,
   global blast burden, final decision, limitations, confidence, and gated NPM1
   assessment.

## Slide Scale And Tiling

The pipeline resolves slide scale from available microns-per-pixel metadata and
falls back to a nominal default otherwise. Physical tile size is converted to
level-0 pixels as:

$$
s_0 = \left\lceil \frac{u_{\mathrm{tile}}}{m_{\mathrm{pp}}} \right\rceil ,
$$

where $u_{\mathrm{tile}}$ is the requested physical tile side length and
$s_0$ is the corresponding level-0 tile side in pixels.

Supertile size is selected so that each supertile contains an integer number of
embedding tiles while respecting the configured maximum span:

$$
\begin{aligned}
u_{\max} &= s_{\max} \, m_{\mathrm{pp}}, \\
n_{\mathrm{side}} &= \max \left(
  \left\lfloor \frac{u_{\max}}{u_{\mathrm{tile}}} \right\rfloor, 1
\right), \\
s_{\mathrm{super}} &= s_0 \, n_{\mathrm{side}} .
\end{aligned}
$$

Each selected supertile is read from level 0, resized to
`tile_size_px * tiles_per_supertile_side`, and partitioned into square
`tile_size_px` inputs for the feature extractor. This design keeps the physical
field size consistent across slides with different scanning resolutions.

## Foreground And Coarse Screening

Foreground detection begins with one thumbnail cell per supertile. A supertile
is considered foreground if:

$$
\bar{g}_{\mathrm{thumb}} < \tau_{\mathrm{brightness}} .
$$

OpenSlide non-empty bounds, when present, further constrain the sampling grid.
When coarse or hybrid filtering is enabled, each foreground supertile receives a
low-resolution screening score. With RGB values scaled to `[0, 1]`:

$$
\begin{aligned}
g &= 0.299R + 0.587G + 0.114B, \\
c &= \max(R,G,B) - \min(R,G,B), \\
\mathbb{1}_{\mathrm{tissue}} &=
  \mathbb{1}\left[g < 0.93 \land (c > 0.035 \lor g < 0.82)\right], \\
\phi_{\mathrm{purple}} &= \max(0, 0.55B + 0.35R - 0.75G).
\end{aligned}
$$

The screening score is:

$$
S_screen = clip01(0.34 F_tissue + 0.33 F_purple + 0.23 H_purple - 0.16 F_red - 0.18 F_gray - 0.12 F_dark - 0.10 F_sat - 0.15 A_artifact)
$$

This stage enriches for stained, basophilic, cellular marrow while suppressing
background, smooth red blood cell dominated material, dull gray-black debris,
stain pooling, and saturated artifacts. Spatial separation constraints are used
to maintain coverage rather than repeatedly sampling a single high-scoring
region.

## Dark And Cellularity Model

The AML heuristic deliberately avoids the simplistic assumption that dark tissue
is diagnostic. Darkness is used only when accompanied by chromaticity,
hematoxylin-like signal, local texture, and preserved cellular structure.

The dark/cellularity prior computes a thumbnail-level cellularity score:

$$
P_{\mathrm{blue}} =
\mathrm{normalize}_{95}\left(
\rho_{B/R} \, c_{\mathrm{gate}} \, d_{\mathrm{mid}} \,
(0.35 + 0.65 t_{\mathrm{gate}})
\right)
$$

$$
C_{\mathrm{cell}} =
\mathrm{normalize}_{95}\left(
c_{\mathrm{gate}} \, d_{\mathrm{mid}} \, t_{\mathrm{gate}} \,
\frac{\max(0, \max(R,B) - G + 18)}{60}
\right)
$$

$$
S_{\mathrm{dark}} = 0.56 P_{\mathrm{blue}} + 0.08 C_{\mathrm{cell}} + 0.15 \rho_{\mathrm{density}} + 0.08 t_{\mathrm{gate}} - 0.18 R_{\mathrm{smooth}} - 0.14 A_{\mathrm{dark}} - 0.18 A_{\mathrm{gray/black}} - 0.12 L_{\mathrm{penalty}} .
$$

The score is thresholded by percentile, expanded within tissue, refined by
connected components, and trimmed to tissue support. In the AML candidate path,
per-tile dark/cellularity scores contribute to ranking as a soft prior rather
than as a mandatory diagnostic gate.

## Tile Quality Model

The tile quality model is the primary hand-engineered AML prior. It measures
tissue content, basophilic cellularity, nuclear texture, focus, red blood cell
dominance, edge effects, and artifact burden.

| Feature family | Methodological role |
| --- | --- |
| `tissue_fraction` | excludes background, glass, and pale empty regions |
| `purple_fraction`, `purple_cellular_fraction` | enriches basophilic nucleated marrow |
| `dark_cellular_fraction`, `dark_in_focus_fraction` | enriches dense, readable cellular areas |
| `hematoxylin`, `nuclear_fraction`, `packed_nuclear_fraction` | approximates nucleated-cell abundance |
| `nuclear_detail`, `focus`, `focus_energy` | favors interpretable chromatin and boundaries |
| `rbc_fraction`, `red_dominant_fraction` | suppresses clot and hemodiluted regions |
| `gray_black_fraction`, `artifact_fraction`, `stringy_artifact_score` | suppresses folds, precipitate, crush, and debris |
| `touches_tissue_edge` | avoids partial tissue-border tiles |

The coarse tile score is:

$$
S_{\mathrm{coarse}} = 0.26 F_{\mathrm{tissue}} + 0.28 F_{\mathrm{purple}} + 0.06 F_{\mathrm{eosinophilic}} + 0.15 F_{\mathrm{focus}} - 0.16 F_{\mathrm{rbc}} - 0.15 F_{\mathrm{artifact}} - 0.18 F_{\mathrm{gray/black}} .
$$

After hard rejection, retained feature values are robustly scaled:

$$
\mathrm{robustunit}(x)
= \mathrm{clip}_{[0,1]}
\left(
  \frac{x - P_{10}(x)}{P_{90}(x) - P_{10}(x)}
\right).
$$

The final quality score is:

$$
Q_i = 0.38 S_{\mathrm{coarse}} + 0.08 P_{\mathrm{cell}} + 0.04 D_{\mathrm{cell}} + 0.03 D_{\mathrm{very}} + 0.04 D_{\mathrm{focus}} + 0.09 F_{\mathrm{nuclear}} + 0.10 F_{\mathrm{packed}} + 0.06 F_{\mathrm{focus}} + 0.04 E_{\mathrm{focus}} + 0.02 \rho_{\mathrm{component}} + 0.01 U_{\mathrm{component}} + 0.03 F_{\mathrm{round}} + 0.02 F_{\mathrm{eosinophilic}} + 0.06 R_{\mathrm{nucleated/red}} - 0.12 A_{\mathrm{artifact}} - 0.12 A_{\mathrm{stringy}} - 0.16 A_{\mathrm{gray/black}} - 0.06 F_{\mathrm{red}} - 0.07 F_{\mathrm{empty}} - 0.16 B_{\mathrm{brightness}} .
$$

The selected tile set is:

$$
\begin{aligned}
k_{\mathrm{keep}} &=
\left\lceil |\mathcal{P}_{\mathrm{pool}}|
\, r_{\mathrm{quality}} \right\rceil, \\
\mathcal{P}_{\mathrm{selected}} &=
\mathrm{TopK}(\mathcal{P}_{\mathrm{pool}}, Q, k_{\mathrm{keep}})
\cup
\mathrm{Reserve}(\mathcal{P}_{\mathrm{pool}}, r_{\mathrm{reserve}}).
\end{aligned}
$$

The reserve term reduces sensitivity to imperfect handcrafted thresholds and
helps retain atypical but potentially informative morphology.

## Foundation Embeddings

Each retained tile is transformed and passed through the selected feature
extractor in batches. The extractor output is reshaped to an $[N,D]$ matrix and
L2-normalized:

$$
z_i =
\frac{f_i}{\max(\lVert f_i \rVert_2, 10^{-12})}.
$$

Cosine similarity can then be computed as an inner product:

$$
\mathrm{sim}(z_i, z_j) = z_i^\top z_j .
$$

Supported extractors include UNI2, Virchow2, H-optimus-1, DINO/Bloom-style
models, RedDINO-style models, and compatible ONNX variants.

## Curated Reference Retrieval

In AML mode, the candidate ranker embeds curated good-quality and bad-quality
reference examples. These examples represent ROI-quality prototypes:

| Reference set | Interpretation |
| --- | --- |
| `Good_Tiles` | hypercellular, basophilic, nucleated, in-focus, low-artifact, morphologically informative fields |
| `Bad_Tiles` | background, red-cell or clot dominance, gray-black debris, blur, crush, artifact-dark regions, edges, and smear debris |

For each query embedding `z_i`, the system retrieves the top similarities to
good and bad references:

$$
\begin{aligned}
G_i &= \mathrm{Agg}\left(
  \mathrm{TopK}_{z \in Z^{+}}
  \mathrm{sim}(z_i, z)
\right), \\
N_i &= \mathrm{Agg}\left(
  \mathrm{TopK}_{z \in Z^{-}}
  \mathrm{sim}(z_i, z)
\right), \\
M_i &= G_i - N_i, \\
B_i &= \sigma(-\lambda M_i).
\end{aligned}
$$

Here $Z^{+}$ and $Z^{-}$ denote embedded good and bad reference banks, and
$\lambda$ is a scale parameter controlling the sharpness of the bad-like
likelihood.

The available aggregation operators are:

$$
\begin{aligned}
\mathrm{Agg}_{\mathrm{mean}}(s_{1:k}) &=
\frac{1}{k}\sum_{j=1}^{k}s_j, \\
\mathrm{Agg}_{\max}(s_{1:k}) &=
\max_{1 \leq j \leq k} s_j, \\
\mathrm{Agg}_{\mathrm{weighted}}(s_{1:k}) &=
\sum_{j=1}^{k} a_j s_j, \\
a_j &= \frac{\exp(-j/3)}{\sum_{\ell=1}^{k}\exp(-\ell/3)} .
\end{aligned}
$$

The logistic transform is:

$$
\sigma(x) = \frac{1}{1+\exp(-x)}.
$$

Quality hints are derived from the margin, top-1 good and bad similarities, and
bad-like likelihood:

$$
h_i =
\begin{cases}
\mathrm{goodlike}, &
  M_i > \tau_M^{+}
  \land B_i < \tau_B^{+}
  \land g_i^{(1)} \geq n_i^{(1)} + \tau_{\Delta}^{+}, \\
\mathrm{badlike}, &
  M_i \leq \tau_M^{-}
  \lor \mathrm{BadDominance}(g_i^{(1)}, n_i^{(1)}, B_i), \\
\mathrm{uncertain}, & \mathrm{otherwise}.
\end{cases}
$$

Strong good support can rescue otherwise ambiguous candidates:

$$
\mathrm{StrongGood}_i =
\mathbb{1}\left[
  g_i^{(1)} \geq \tau_g
  \land n_i^{(1)} \leq \tau_n
  \land B_i \leq \tau_{\mathrm{bad}}
\right].
$$

This mechanism converts a small curated visual memory into a fast ROI-quality
prior while preventing the reference bank from becoming the final AML decision
rule.

## Candidate Scoring

For a view `v`, the ranker first selects indexed tiles whose centers lie inside
the level-0 bounding box of `v`. In AML reference mode, very low
dark/cellularity candidates may be excluded:

$$
p_i \in \mathcal{P}_v \quad \mathrm{iff} \quad D_i \geq \tau_{\mathrm{dark}} .
$$

For each remaining tile:

$$
Delta_i = g1_i - n1_i
$$

$$
P_qual_i = w_M M_i + w_Delta Delta_i
$$

$$
L_bad_i = clip01((B_i - b_0) / (1 - b_0))
$$

$$
L_match_i = clip01((n1_i - g1_i + delta) / alpha)
$$

$$
L_qual_i = w_bad * L_bad_i + w_match * L_match_i
$$

Here `g1_i` and `n1_i` are the top-1 good and bad reference
similarities, respectively.

The combined candidate score is:

$$
S_cand_i = w_D * z(D_i) + w_G * z(G_i) + w_P * z(P_qual_i) - w_L * L_qual_i + w_S * I[StrongGood_i] + w_R * I[DarkRegion_i]
$$

where $z(x) = (x - \mu_x)/(\sigma_x + \epsilon)$ is computed within the current
candidate set. Candidate post-processing removes edge-touching and duplicate
regions, orders the remaining candidates by quality hint and retrieval support,
and hides bad-like candidates whenever non-bad alternatives are available. If
every candidate is bad-like, the system may expose a low-confidence best-effort
candidate to avoid an empty navigation state.

## Fallback Candidate Search

When the embedding index cannot provide candidates, the viewer falls back to a
7-by-7 grid over the current rendered field. For each patch:

$$
F_{\mathrm{tissue}} = \mathrm{mean}(\mathbb{1}[g < 0.92])
$$

$$
E_{\mathrm{edge}} = \frac{\mathrm{mean}(|\nabla_x g|) + \mathrm{mean}(|\nabla_y g|)}{2}
$$

$$
S_{\mathrm{fallback}} = 0.70 F_{\mathrm{tissue}} + 0.30 E_{\mathrm{edge}} .
$$

Patches with insufficient tissue or tissue-edge artifacts are skipped. The
remaining patches are ranked by texture-rich tissue content and constrained by
minimum normalized center spacing. This fallback is a navigation aid only and
does not contribute disease evidence by itself.

## Agentic ROI Acquisition

The VLM is exposed to the current view image, a compact candidate list,
ROI-quality examples, progress reminders, and relevant navigation state. The
intended acquisition sequence is:

```text
open whole-slide overview
repeat until target ROI count:
  open the next ranked candidate field
  inspect locally with limited zoom/pan
  mark a readable local ROI
produce the final morphology report
```

The prompt explicitly frames each candidate as a search region rather than a
fixed target. The model should select a readable high-cellularity field inside
the opened view, not blindly mark the candidate center. Guardrails enforce an
overview-first workflow, discourage repeated same-region navigation, warn on
low-tissue fields, prevent duplicate ROIs, and require finalization after the
ROI target or cap is reached. The runtime exposes `target_accepted_rois` and
`max_accepted_rois` as configuration parameters; the tuned AML defaults are 5
and 5 respectively.

## ROI Marking And Evidence Capture

ROI marking uses normalized coordinates in the current view. The center is
mapped to level-0 slide coordinates:

$$
c^{(0)}_x = x^{(0)}_{\mathrm{view}} + \mathrm{round}\left(\frac{c^{(999)}_x}{999} w_{\mathrm{view}}\right)
$$

$$
c^{(0)}_y = y^{(0)}_{\mathrm{view}} + \mathrm{round}\left(\frac{c^{(999)}_y}{999} h_{\mathrm{view}}\right)
$$

The final evidence crop is a fixed square:

$$
\begin{aligned}
s_{\mathrm{roi}} &= s_{\mathrm{fixed}}, \\
x_0 &= \mathrm{clamp}
\left(c^{(0)}_x - \frac{s_{\mathrm{roi}}}{2}, 0, W_S - s_{\mathrm{roi}}\right), \\
y_0 &= \mathrm{clamp}
\left(c^{(0)}_y - \frac{s_{\mathrm{roi}}}{2}, 0, H_S - s_{\mathrm{roi}}\right).
\end{aligned}
$$

Duplicate suppression uses center distance:

$$
\mathrm{duplicate}(r_a, r_b) =
\mathbb{1}\left[
  \lVert c_a - c_b \rVert_2 < 0.5 s_{\mathrm{roi}}
\right].
$$

AML-specific artifact rejection removes saturated dark-blue flood fields:

$$
\mathbb{1}_{\mathrm{flood}} =
\mathbb{1}\left[
  g < 0.22
  \land c > 0.18
  \land B > R + 0.15
  \land B > G + 0.05
  \land E_{\mathrm{lap}} < 0.05
\right],
$$

with rejection when:

$$
\mathrm{mean}(\mathbb{1}_{\mathrm{flood}}) > 0.30 .
$$

In AML modes, accepted and discarded ROI changes are checkpointed immediately to
`roi_collection.json` together with copied ROI images. This allows
`aml_auto` to continue into diagnosis from the saved collection if ROI
acquisition later terminates on the turn budget.

## Diagnostic Decision Rule

Final diagnosis is produced by the VLM from accepted ROI crops under strict
prompt constraints. The model estimates blasts among interpretable nucleated
hematopoietic cells, excluding red blood cells, fat, empty background, and
artifacts. Per-ROI and global blast burden must be reported using the exact
tiers `<5%`, `5-9%`, `10-19%`, `20-50%`, and `>50%`.

The binary morphology rule is:

$$
\hat{y} =
\begin{cases}
\mathrm{AcuteLeukemia}, &
  \hat{b}_{\mathrm{global}} \geq 0.20
  \land \mathrm{DiffuseImmatureMorphology}, \\
\mathrm{NormalMarrow}, &
  \hat{b}_{\mathrm{global}} < 0.05
  \lor \mathrm{HeterogeneousMaturationPreserved}, \\
\mathrm{closestbinarylabel}, &
  0.05 \leq \hat{b}_{\mathrm{global}} < 0.20
  \land \mathrm{SuspiciousMorphology}.
\end{cases}
$$

For the intermediate range, the report must state uncertainty and limitations.

Blast-like cells must be identified by morphology, not by color alone. Relevant
features include high nuclear-to-cytoplasmic ratio, round or oval immature
nuclei, fine chromatin, visible nucleoli when present, scant cytoplasm, and Auer
rods when present.

NPM1 prediction is gated by the AML decision:

$$
\mathrm{NPMOneApplicable} =
\mathbb{1}[\hat{y} = \mathrm{AcuteLeukemia}] .
$$

When applicable, the model classifies `NPM1_mutated` versus `NPM1_wildtype`
using morphology-only supportive evidence such as cup-like nuclear
invaginations, folded or irregular nuclei, monocytic differentiation, and
relatively abundant cytoplasm.

## Methodological Contributions

1. The method formulates AML WSI analysis as retrieval-guided microscopy rather
   than direct whole-slide classification.

2. The reference memory encodes ROI quality rather than disease labels, allowing
   the final AML decision to remain morphology-based and auditable.

3. The dark/cellularity prior is multi-factorial. It combines stain color,
   chroma, texture, nuclear density, focus, and artifact penalties to avoid the
   failure mode that equates any dark region with leukemia.

4. The system separates navigation from diagnosis. Candidate tiles guide the VLM
   toward promising fields, but only accepted ROI crops contribute to the final
   report.

5. The agent loop is bounded and reportable. It produces a concise trail of
   selected ROIs, morphology observations, final decision, confidence, and
   limitations.
