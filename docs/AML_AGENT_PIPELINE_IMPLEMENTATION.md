# AML Agent Pipeline Methodology

This document describes the end-to-end AML detector agent implemented in this
repository. It focuses on the active methodology: how the system opens a bone
marrow WSI, reduces it to candidate high-power regions, guides the VLM through a
bounded ROI collection policy, and produces a morphology-only AML triage report.

The key design idea is not to ask the VLM to scan an entire WSI unaided. The
backend first builds a compact, morphology-aware candidate map from cheap stain
heuristics, foundation-model embeddings, and curated ROI-quality references.
The VLM then acts like a pathologist using a digital slide viewer: it jumps to
promising regions, inspects locally, keeps exactly the configured number of
interpretable ROIs, and makes a final morphology-only decision from those ROIs.

## Source Map

The pipeline is spread across these files:

| Area | Main files |
| --- | --- |
| Runtime entrypoints | `main.py`, `evaluate/run_single_slide.py`, `evaluate/run_batch_aml.sh` |
| Agent selection and tools | `wsi_core_pkg/runtime.py`, `wsi_core_pkg/agents.py`, `wsi_core_pkg/tools.py` |
| AML task prompt | `wsi_core_pkg/prompts.py` |
| Context/image injection | `wsi_core_pkg/context_injection.py` |
| Dark/cellularity overlay | `wsi_core_pkg/dark_regions.py` |
| Tiling and embedding extraction | `wsi_core_pkg/embeddings/tiling.py` |
| Raw tile quality heuristics | `wsi_core_pkg/embeddings/tile_prefilter.py` |
| Coarse thumbnail prefilter | `wsi_core_pkg/embeddings/openslide_prefilter.py` |
| Candidate ranking/retrieval | `wsi_core_pkg/embeddings/roi_ranker.py` |
| Reports | `wsi_core_pkg/reporting.py` |
| Tunable defaults | `configs/config.yaml` |

## End-to-End Flow

1. The user selects a WSI, AML agent, VLM model, embedding extractor, tile size,
   batch size, ROI crop size, and tile prefilter mode in the web UI or CLI.

2. `run_wsi_agent_for_web()` resets global WSI state with the selected slide,
   extractor, tile prefilter, ROI target, and cache settings. It selects
   `WSIAmlDetectorAgent`, whose instructions are `DEFAULT_AML_PROMPT`.

3. The agent must call `wsi_get_overview_view()` first. That renders the
   whole-slide overview and lazily triggers ROI candidate preparation through
   `_attach_roi_candidates()` -> `_refresh_roi_candidates_for_current_view()` ->
   `_ensure_unsupervised_roi_index()`.

4. Candidate preparation checks disk caches. If a matching ROI index or feature
   matrix already exists for the slide/extractor/filter/tile-size setting, it is
   loaded. Otherwise the slide is tiled, filtered, embedded, reference-scored,
   and indexed.

5. The current view receives `roi_candidates`. In AML mode the candidates are
   ranked by dark/cellularity score, reference retrieval support, and bad-quality
   penalties. Bad-like candidates are hidden when non-bad alternatives exist.

6. The VLM opens a candidate with `wsi_open_candidate(rank)`. This jumps to a
   configured candidate navigation field, commonly around 600 um unless
   overridden. The candidate is only a region-level hint, not the final ROI.

7. Inside that field, the VLM locally zooms/pans or directly marks a usable ROI.
   The final ROI crop is created at native level 0 with fixed side length
   `ROI_OUTPUT_SIZE_PX` and stored in `state._roi_marks`.

8. The system blocks duplicate ROIs, warns on empty/background fields, rejects
   saturated dark-blue flood artifacts, auto-saves one representative good tile
   from early kept AML ROIs, and guides the VLM toward the next unattempted
   candidate.

9. The agent repeats candidate opening and local inspection until the configured
   AML target is reached. The default config uses target 5 and hard cap 5; the
   prompt asks for exactly 5 accepted ROIs.

10. Once the target/cap is reached, navigation guards stop further tool calls.
    Context injection shows all kept ROI images and asks the model for strict
    JSON: per-ROI blast range, global blast range, final decision, confidence,
    and NPM1 prediction only if AML is morphologically established.

11. `write_markdown_report()` writes `report.md` and `report.txt`, including the
    prompt, final output, ROI images/metadata, navigation steps, and AML
    retrieval summaries. CLI wrappers also export `summary.json`, `report.json`,
    `final_output.txt`, ROI images, and state.

## Methodology Overview

The AML agent is a cascade:

```text
WSI input
  -> slide metadata and MPP resolution
  -> supertile foreground scan
  -> optional coarse prefilter
  -> tile split at fixed physical size
  -> raw tile texture/stain/cellularity filtering
  -> foundation-model embeddings
  -> curated good/bad reference retrieval
  -> per-view top-K ROI candidates
  -> bounded VLM navigation and ROI marking
  -> strict morphology-only AML JSON report
```

The cascade is deliberately asymmetric. Most of the WSI is handled by cheap,
deterministic heuristics and cached embeddings. The expensive VLM is used only
where it is strongest: visual inspection of a small number of high-power fields
and final morphology reasoning.

## Algorithmic Pseudocode

The following pseudocode summarizes the AML agent as a set of professional
pipeline functions. Symbols are intentionally compact so the method can be
mapped directly onto implementation, experiments, or a manuscript-style methods
section.

### Symbol Table

| Symbol | Meaning |
| --- | --- |
| `S` | input whole-slide image |
| `theta` | selected foundation embedding extractor |
| `mpp` | microns per pixel resolved from slide metadata or fallback |
| `T` | foreground supertile set |
| `P` | candidate tile set |
| `p_i` | tile `i` |
| `z_i = f_theta(p_i)` | L2-normalized embedding for tile `p_i` |
| `R_pos`, `R_neg` | curated good and bad ROI-quality reference tiles |
| `D_i` | dark/cellularity score for tile `p_i` |
| `Q_i` | deterministic tile quality score |
| `M_i` | good-minus-bad reference margin |
| `B_i` | bad-like likelihood |
| `C_v` | ranked candidate list for current view `v` |
| `A` | accepted ROI set |
| `K` | requested candidate count |
| `N_target` | target accepted ROI count |
| `N_max` | hard accepted ROI cap |

### Main AML Agent

```text
function AML_AGENT(S, cfg):
    state <- RESET_WSI_STATE(S, cfg)
    theta <- LOAD_EXTRACTOR(cfg.extractor_name)
    mpp <- RESOLVE_MPP(S, cfg.default_mpp_um)

    v0 <- WSI_GET_OVERVIEW_VIEW(S)
    I <- ENSURE_ROI_INDEX(S, theta, mpp, cfg)
    C_v <- SELECT_CANDIDATES_FOR_VIEW(I, bbox(v0), K=cfg.roi_candidate_top_k)

    A <- empty list
    attempted <- empty set

    while |A| < cfg.N_target and |A| < cfg.N_max:
        c_star <- NEXT_BEST_CANDIDATE(C_v, attempted, A)

        if c_star is None:
            C_v <- FALLBACK_CANDIDATES_FROM_VIEW(v0, cfg)
            c_star <- NEXT_BEST_CANDIDATE(C_v, attempted, A)
            if c_star is None:
                break

        v_c <- WSI_OPEN_CANDIDATE(c_star.rank, field_um=cfg.candidate_nav_field_um)
        r_candidate <- VLM_LOCAL_INSPECT_AND_PROPOSE_ROI(v_c, c_star, cfg)

        if r_candidate is None:
            attempted.add(c_star.rank)
            C_v <- REFRESH_CANDIDATES_FOR_CURRENT_VIEW(I, cfg)
            continue

        r <- WSI_MARK_ROI_NORM(r_candidate.bbox_norm)

        if r.reason == "duplicate_roi":
            attempted.add(r_candidate.bbox_level0)
            C_v <- REFRESH_CANDIDATES_FOR_CURRENT_VIEW(I, cfg)
            continue

        if r.reason == "dark_blue_flood_rejected":
            attempted.add(r_candidate.bbox_level0)
            C_v <- REFRESH_CANDIDATES_FOR_CURRENT_VIEW(I, cfg)
            continue

        if ROI_IS_CLEARLY_UNUSABLE(r):
            maybe_discard <- WSI_DISCARD_LAST_ROI(r)
            if maybe_discard.ok:
                attempted.add(r_candidate.bbox_level0)
                C_v <- REFRESH_CANDIDATES_FOR_CURRENT_VIEW(I, cfg)
                continue

        A.append(r)
        AUTO_SAVE_GOOD_TILE_IF_NEEDED(r)

        if |A| >= cfg.N_target or |A| >= cfg.N_max:
            break

        C_v <- REFRESH_CANDIDATES_FOR_CURRENT_VIEW(I, cfg)

    return FINAL_AML_JSON(A, cfg.prompt_rules)
```

### ROI Index Construction

```text
function ENSURE_ROI_INDEX(S, theta, mpp, cfg):
    cache_key <- HASH(S.path, theta.id, cfg.tile_filter, cfg.tile_size_px, "aml")

    if ROI_INDEX_CACHE_EXISTS(cache_key):
        return LOAD_ROI_INDEX(cache_key)

    T <- SELECT_FOREGROUND_SUPERTILES(S, mpp, cfg)
    P <- empty list

    for each supertile t in T:
        tiles <- SPLIT_SUPERTILE(t, cfg.tile_size_um, cfg.tile_size_px)
        tiles <- FILTER_BY_TEXTURE_AND_EDGE(tiles, cfg)

        if cfg.tile_filter in {"quality", "hybrid"}:
            tiles <- QUALITY_PREFILTER(tiles, cfg)

        P.extend(tiles)

    Z <- EMBED_AND_L2_NORMALIZE(P, theta)
    D <- DARK_CELLULARITY_SCORE(P)

    if cfg.agent_type == "aml":
        R_pos, R_neg <- LOAD_REFERENCE_TILES(cfg.example_tiles_root)
        Z_pos, Z_neg <- EMBED_AND_L2_NORMALIZE(R_pos, theta), EMBED_AND_L2_NORMALIZE(R_neg, theta)
        ref <- REFERENCE_RETRIEVAL_SCORES(Z, Z_pos, Z_neg, cfg)
        base_scores <- ref.retrieval_score
    else:
        base_scores <- KNN_NOVELTY_SCORES(Z, cfg)
        ref <- empty reference evidence

    I <- ROI_INDEX(
        slide_path=S.path,
        coordinates=LEVEL0_COORDS(P, mpp),
        embeddings=Z,
        dark_scores=D,
        base_scores=base_scores,
        reference_evidence=ref
    )

    SAVE_ROI_INDEX(cache_key, I)
    return I
```

### Tile Quality Function

```text
function QUALITY_PREFILTER(P, cfg):
    for each tile p_i in P:
        metrics_i <- TILE_METRICS(p_i)

        hard_keep_i <-
            tissue_fraction_i >= tau_tissue
            and has_chromatic_cellular_signal_i
            and rbc_fraction_i <= tau_rbc
            and artifact_fraction_i <= tau_artifact
            and gray_black_fraction_i <= tau_gray_black
            and not touches_tissue_edge_i
            and not stringy_artifact_i

        if hard_keep_i:
            Q_i <- TILE_QUALITY_SCORE(metrics_i)

    P_pool <- {p_i in P where hard_keep_i = true}
    P_top <- TOP_BY_SCORE(P_pool, Q, ceil(cfg.quality_keep_ratio * |P_pool|))
    P_reserve <- RANDOM_RESERVE(P_pool - P_top, cfg.random_reserve_ratio)

    return P_top union P_reserve
```

The quality score is the weighted AML morphology proxy:

```text
Q_i =
    0.38 * coarse_score_i
  + 0.08 * purple_cellular_i
  + 0.04 * dark_cellular_i
  + 0.03 * very_dark_i
  + 0.04 * dark_in_focus_i
  + 0.09 * nuclear_fraction_i
  + 0.10 * packed_nuclear_i
  + 0.06 * focus_i
  + 0.04 * focus_energy_i
  + 0.02 * component_density_i
  + 0.01 * component_uniformity_i
  + 0.03 * round_nuclei_i
  + 0.02 * eosinophilic_cellular_i
  + 0.06 * nucleated_to_red_i
  - 0.12 * artifact_i
  - 0.12 * stringy_i
  - 0.16 * gray_black_i
  - 0.06 * red_i
  - 0.07 * empty_i
  - 0.16 * brightness_i
```

### Reference Retrieval Function

```text
function REFERENCE_RETRIEVAL_SCORES(Z, Z_pos, Z_neg, cfg):
    for each tile embedding z_i in Z:
        G_i <- TOP_K_SIMILARITIES(z_i, Z_pos, k=cfg.reference_top_k)
        N_i <- TOP_K_SIMILARITIES(z_i, Z_neg, k=cfg.reference_top_k)

        good_i <- AGGREGATE(G_i, cfg.reference_aggregation)
        bad_i <- AGGREGATE(N_i, cfg.reference_aggregation)

        M_i <- good_i - bad_i
        B_i <- SIGMOID(cfg.logit_scale * (-M_i))

        quality_hint_i <- CLASSIFY_REFERENCE_HINT(
            margin=M_i,
            bad_likelihood=B_i,
            good_top1=max(G_i),
            bad_top1=max(N_i),
            cfg=cfg
        )

    return {
        retrieval_score: good,
        margin: M,
        bad_likelihood: B,
        quality_hint: quality_hint
    }
```

Where:

```text
SIGMOID(x) = 1 / (1 + exp(-x))

AGGREGATE(s_1 ... s_k, "mean") = mean(s_1 ... s_k)
AGGREGATE(s_1 ... s_k, "max") = max(s_1 ... s_k)
AGGREGATE(s_1 ... s_k, "weighted") =
    sum_i w_i * s_i, where w_i = exp(-i / 3) / sum_j exp(-j / 3)
```

### Candidate Ranking Function

```text
function SELECT_CANDIDATES_FOR_VIEW(I, bbox_v, K):
    P_v <- {p_i in I.tiles where center(p_i) inside bbox_v}

    if AML_REFERENCE_MODE(I):
        P_v <- {p_i in P_v where D_i >= cfg.absolute_min_dark_score}

    for each p_i in P_v:
        gap_i <- good_top1_i - bad_top1_i

        quality_prior_i <-
            w_margin * M_i
          + w_gap * gap_i

        bad_like_penalty_i <-
            clip((B_i - bad_like_baseline) / (1 - bad_like_baseline), 0, 1)

        bad_match_penalty_i <-
            clip((bad_top1_i - good_top1_i + bad_offset) / bad_scale, 0, 1)

        quality_penalty_i <-
            w_bad_like * bad_like_penalty_i
          + w_bad_match * bad_match_penalty_i

        S_i <-
            w_dark * ZSCORE(D_i)
          + w_base * ZSCORE(retrieval_score_i)
          + w_prior * ZSCORE(quality_prior_i)
          - w_penalty * quality_penalty_i
          + good_support_bonus_i
          + optional_dark_box_prior_i

    C <- SORT_DESC(P_v, key=S)
    C <- REMOVE_EDGE_TOUCHING_AND_DUPLICATE_CANDIDATES(C)
    C <- HIDE_BAD_LIKE_WHEN_BETTER_OPTIONS_EXIST(C)

    return TOP_K(C, K)
```

### Final Diagnosis Function

```text
function FINAL_AML_JSON(A, prompt_rules):
    for each ROI r_j in A:
        cells_j <- interpretable nucleated hematopoietic cells in r_j
        blasts_j <- cells with high N:C ratio, immature nucleus, fine chromatin,
                    visible nucleoli when present, and scant cytoplasm

        blast_fraction_j <- blasts_j / max(cells_j, 1)
        blast_range_j <- BIN_BLAST_FRACTION(blast_fraction_j)

    global_blast_range <- SYNTHESIZE_ROI_RANGES({blast_range_j})

    if global_blast_range indicates >= 20% blasts
       and morphology is diffusely immature/blast-rich:
        final_decision <- "Acute leukemia"
    else:
        final_decision <- "Normal marrow"

    if final_decision == "Acute leukemia":
        npm1_prediction <- MORPHOLOGY_ONLY_NPM1_ASSESSMENT(A)
    else:
        npm1_prediction <- NOT_APPLICABLE()

    return STRICT_JSON(
        morphology_summary,
        accepted_rois,
        discard_summary,
        global_blast_range,
        final_decision,
        limitations_confidence,
        npm1_prediction
    )
```

## Slide Scale And Tiling

The system tries to use slide MPP from OpenSlide metadata. If metadata is
missing, it falls back to `DEFAULT_MPP_UM` from `configs/config.yaml`.

For a requested physical tile size:

```text
tile_size_level0_px = ceil(tile_size_um / slide_mpp_um)
```

Supertile size is chosen so each supertile contains an integer number of tiles
and does not exceed the configured maximum level-0 span:

```text
max_supertile_um = max_supertile_size_slide_px * slide_mpp_um
tiles_per_supertile_side = max(floor(max_supertile_um / tile_size_um), 1)
supertile_size_level0_px = tile_size_level0_px * tiles_per_supertile_side
```

Each supertile is read from level 0, resized to
`tile_size_px * tiles_per_supertile_side`, and split into `tile_size_px` square
tiles for the embedding extractor.

## Foreground And Coarse Screening

The first pass builds one thumbnail cell per supertile. A supertile is
foreground when:

```text
thumbnail_gray < ROI_INDEX_BUILD_BRIGHTNESS_CUTOFF
```

OpenSlide non-empty bounds, when present, further restrict the grid.

When coarse or hybrid filtering is enabled, each foreground supertile receives a
low-resolution screening score from `screening_tile_metrics()` and
`screening_roi_score()`.

Definitions, using RGB channels scaled to `[0, 1]`:

```text
gray = 0.299 R + 0.587 G + 0.114 B
chroma = max(R, G, B) - min(R, G, B)
tissue_mask = (gray < 0.93) and (chroma > 0.035 or gray < 0.82)
purple_signal = max(0, 0.55 B + 0.35 R - 0.75 G)
```

The coarse screening score is:

```text
S_screen = clip(
    0.34 * tissue_fraction
  + 0.33 * purple_fraction
  + 0.23 * purple_strength
  - 0.16 * red_fraction
  - 0.18 * gray_cluster_fraction
  - 0.12 * dark_fraction
  - 0.10 * extreme_saturation_fraction
  - 0.15 * artifact_penalty,
  0, 1
)
```

Intent:

- keep stained tissue, especially purple/basophilic cellular marrow;
- reject smooth red/RBC-heavy material;
- reject dull gray-black clusters, over-dark low-detail regions, and saturated
  artifacts;
- preserve enough spatial diversity by selecting separated supertiles.

There is also a dark-region overlay/gating implementation in
`dark_regions.py` and optional dark-box hooks in the tiler/ranker. In the active
interactive AML candidate path, `_ensure_unsupervised_roi_index()` currently
passes `dark_region_boxes_level0=None`, so explicit dark boxes are not a hard
gate. The active AML ranking still uses per-tile dark/cellularity scores and
the web UI can display the dark-region overlay for human inspection.

## Dark/Cellularity Detection

Darkness alone is intentionally not enough. The AML pipeline uses darkness only
when it co-occurs with chroma, hematoxylin/nuclear signal, texture, and preserved
cell-like structure.

`dark_regions.py` computes a thumbnail cellularity score:

```text
purple_blue_norm = normalize95(
  blue_over_red * chroma_gate * mid_darkness * (0.35 + 0.65 * texture_gate)
)

chromatic_cellular_norm = normalize95(
  chroma_gate * mid_darkness * texture_gate * max(0, max(R, B) - G + 18) / 60
)

score =
    0.56 * purple_blue_norm
  + 0.08 * chromatic_cellular_norm
  + 0.15 * density
  + 0.08 * texture_gate
  - 0.18 * red_smooth_norm
  - 0.14 * artifact_dark
  - 0.18 * gray_black_penalty
  - 0.12 * light_penalty
```

The dark-region module thresholds this score by percentile, grows high-scoring
cores inside tissue, refines connected components, trims boxes to tissue, and
writes both a JPEG overview and a soft alpha mask.

Intent:

- detect nuclei-rich basophilic fields;
- avoid confusing RBC/clot, stain pooling, folds, and charcoal debris with
  cellular marrow;
- provide a human-visible overlay and optional coarse prior.

## Raw Tile Quality Prefilter

The quality prefilter in `tile_prefilter.py` is the main hand-built AML
heuristic layer. It computes tile-level features, hard-rejects obviously bad
tiles, then ranks the remaining pool.

Important feature families:

| Feature | Intent |
| --- | --- |
| `tissue_fraction` | reject background/glass and pale empty regions |
| `purple_fraction`, `purple_cellular_fraction` | find basophilic nucleated marrow |
| `dark_cellular_fraction`, `very_dark_fraction`, `dark_in_focus_fraction` | find dense, dark, in-focus cellular fields |
| `hematoxylin`, `nuclear_fraction`, `packed_nuclear_fraction` | estimate nucleated-cell richness |
| `nuclear_detail`, `focus`, `focus_energy` | prefer readable chromatin/cell boundaries |
| `rbc_fraction`, `red_dominant_fraction` | reject RBC/clot/hemodilution |
| `gray_black_fraction`, `artifact_fraction`, `stringy_artifact_score` | reject folds, debris, smear streaks, precipitate, crush |
| `touches_tissue_edge` | avoid partial edge tiles with background at borders |

The coarse tile score is:

```text
S_coarse =
    0.26 * tissue_fraction
  + 0.28 * purple_fraction
  + 0.06 * eosinophilic_cellular_fraction
  + 0.15 * focus_score
  - 0.16 * rbc_fraction
  - 0.15 * artifact_fraction
  - 0.18 * gray_black_fraction
```

After hard rejection, feature values are robustly scaled:

```text
robust_unit(x) = clip((x - P10(x)) / (P90(x) - P10(x)), 0, 1)
```

The hybrid quality score for retained candidate tiles is:

```text
S_quality =
    0.38 * S_coarse
  + 0.08 * purple_cellular
  + 0.04 * dark_cellular
  + 0.03 * very_dark
  + 0.04 * dark_in_focus
  + 0.09 * nuclear_fraction
  + 0.10 * packed_nuclear
  + 0.06 * focus
  + 0.04 * focus_energy
  + 0.02 * component_density
  + 0.01 * component_uniformity
  + 0.03 * round_nuclei
  + 0.02 * eosinophilic_cellular
  + 0.06 * nucleated_to_red
  - 0.12 * artifact
  - 0.12 * stringy
  - 0.16 * gray_black
  - 0.06 * red
  - 0.07 * empty
  - 0.16 * brightness
```

The selected set is:

```text
keep_top = ceil(pool_size * ROI_QUALITY_PREFILTER_KEEP_RATIO)
selected = top_by_score(keep_top) + random_reserve(rejected_pool)
```

The small random reserve preserves diversity and reduces the chance that a
single imperfect heuristic wipes out a useful atypical region.

## Foundation Model Embeddings

For every selected tile:

1. The extractor transform converts the PIL tile to tensor.
2. The extractor model runs in batches, using CUDA AMP when available.
3. The feature output is normalized to a 2D matrix `[N, D]`.
4. Features are L2-normalized for cosine retrieval:

```text
f_l2 = f / max(||f||_2, 1e-12)
```

Supported extractors include `uni2`, `virchow2`, `h_optimus_1`,
`dinobloom*`, `reddino*`, and ONNX variants when installed. Feature matrices are
cached with a key that includes slide, extractor, tile size, filter settings,
and relevant code/config hashes.

## Curated Reference Retrieval

In AML mode, the candidate ranker embeds curated tiles from
`Selected_Tiles/Good_Tiles` and `Selected_Tiles/Bad_Tiles`.

Current context injection describes these as ROI-quality examples, not direct
AML-vs-normal labels:

- good: hypercellular, basophilic, nucleated, in focus, low artifact,
  morphologically informative;
- bad: background-heavy, RBC/clot-dominant, gray-black junk, blurred, crushed,
  artifact-dark, or edge/debris dominated.

For each slide tile, the ranker computes cosine similarities to good and bad
reference tiles. Search is exact matrix multiplication for smaller banks or HNSW
when enabled and useful.

For each query tile:

```text
sim(q, r) = q_l2 dot r_l2
good_score = aggregate(top_k_good_sims)
bad_score = aggregate(top_k_bad_sims)
margin = good_score - bad_score
bad_likelihood = sigmoid(AML_REFERENCE_LOGIT_SCALE * (-margin))
retrieval_score = good_score
```

The aggregation mode is configurable:

```text
mean:     aggregate = mean(s_1 ... s_k)
max:      aggregate = max(s_1 ... s_k)
weighted: aggregate = sum_i soft_weight_i * s_i
          soft_weight_i = exp(-i / 3) / sum_j exp(-j / 3)
```

Candidate quality hints are derived from margin, top-1 good/bad similarities,
and `bad_likelihood`:

```text
good_like if:
  margin > AML_GOOD_LIKE_MARGIN_THRESHOLD
  and bad_likelihood < AML_GOOD_LIKE_BAD_LIKELIHOOD_MAX
  and good_top1 >= bad_top1 + AML_GOOD_LIKE_TOP1_GAP

bad_like if:
  margin <= AML_BAD_MARGIN_REJECT_THRESHOLD
  or bad_likelihood/bad_top1 thresholds indicate bad-reference dominance

uncertain otherwise
```

Strong good support can rescue a candidate even when bad-reference evidence is
present:

```text
strong_good_support =
  good_top1 >= AML_SUPPORT_GOOD_TOP1_FLOOR
  and bad_top1 <= AML_SUPPORT_BAD_TOP1_CEILING
  and bad_likelihood <= AML_SUPPORT_BAD_LIKE_CEILING
```

Intent:

- turn a small curated visual memory into a fast quality prior;
- keep the VLM away from recurrent failure modes;
- rank diagnostic ROI quality without letting the reference bank make the final
  AML diagnosis.

## Candidate Ranking

For a current view, the ranker selects all indexed tile centers inside the view.
If AML reference mode is active, very low dark/cellularity tiles can be filtered:

```text
keep if dark_roi_score >= AML_ABSOLUTE_MIN_DARK_SCORE
```

For each remaining tile:

```text
similarity_gap = good_top1 - bad_top1
quality_prior =
    AML_QUALITY_PRIOR_BAD_MARGIN_WEIGHT * margin
  + AML_QUALITY_PRIOR_SIMILARITY_GAP_WEIGHT * similarity_gap

bad_like_soft_penalty =
  clip((bad_likelihood - baseline) / (1 - baseline), 0, 1)

bad_match_soft_penalty =
  clip((bad_top1 - good_top1 + offset) / scale, 0, 1)

quality_penalty =
    AML_QUALITY_PENALTY_BAD_LIKE_WEIGHT * bad_like_soft_penalty
  + AML_QUALITY_PENALTY_BAD_MATCH_WEIGHT * bad_match_soft_penalty
```

The final AML candidate score is:

```text
S_candidate =
    AML_COMBINED_RANK_DARK_WEIGHT * z(dark_roi_score)
  + AML_COMBINED_RANK_BASE_SCORE_WEIGHT * z(retrieval_score)
  + AML_COMBINED_RANK_QUALITY_PRIOR_WEIGHT * z(quality_prior)
  - AML_COMBINED_RANK_QUALITY_PENALTY_WEIGHT * quality_penalty
  + AML_COMBINED_RANK_GOOD_SUPPORT_BONUS * I(strong_good_support)
  + AML_DARK_REGION_BOX_PRIOR_WEIGHT * I(inside_dark_region)
```

Here `z(x) = (x - mean(x)) / std(x)` within the current candidate set.

Candidates are sorted by `S_candidate`, then filtered for spacing/overlap and
edge-touching tissue. In `tools.py`, AML-facing post-processing further orders
visible candidates by:

```text
rank_key = (
  quality_hint_bonus,
  good_top1_similarity,
  retrieval_score,
  combined_rank_score
)

quality_hint_bonus =
  1.0 for good_like
  0.5 for uncertain
  0.0 for bad_like
```

Bad-like candidates are hidden when at least one non-bad candidate exists. If
all candidates are bad-like, one low-confidence best-effort candidate may be
shown to avoid an empty list, with warnings.

## Fallback Heuristic Candidate Search

If the embedding index cannot provide candidates, the system falls back to a
7-by-7 grid over the current rendered view.

For each grid patch:

```text
tissue = mean(gray < 0.92)
edge = (mean(abs(diff_x(gray))) + mean(abs(diff_y(gray)))) / 2
S_fallback = 0.70 * tissue + 0.30 * edge
```

Patches touching tissue edges or with `tissue < 0.05` are skipped. The remaining
patches are sorted by score with a minimum normalized center spacing.

Intent:

- keep navigation possible when the learned index is unavailable;
- prefer tissue and local texture over empty background;
- avoid making the fallback a diagnostic decision-maker.

## VLM Navigation Policy

The VLM sees:

- the current view image;
- example good/bad ROI-quality images at the start of a run;
- progress reminders such as `Current progress: 3/5 ROIs marked`;
- a concise candidate list with rank, score, and quality hint;
- current-view and overview images depending on context budget.

Core tool flow:

```text
wsi_get_overview_view
repeat until target ROI count:
  wsi_open_candidate(rank)
  inspect locally with at most a small number of zoom/pan steps
  wsi_mark_roi_norm(...) or wsi_mark_candidate(...)
final JSON
```

The prompt and tool feedback intentionally say that a candidate is a search
region, not a fixed target. The agent should find readable high-cellularity
tissue inside the opened field and mark the best local ROI rather than blindly
marking the candidate center.

Important guardrails:

- `wsi_get_overview_view()` must be first.
- `wsi_open_candidate()` is preferred for candidate jumps.
- `wsi_get_view_info()` should be rare.
- Repeated same-region navigation triggers a loop warning.
- Consecutive low-tissue views trigger a low-tissue warning.
- When ROI target/cap is reached, tool calls return a finalization-required
  response.
- `wsi_discard_last_roi()` is blocked near target except for clearly bad ROIs,
  to prevent over-discarding borderline but interpretable AML evidence.

## ROI Marking And Evidence Capture

`wsi_mark_roi_norm()` accepts normalized coordinates `[0, 999]` relative to the
current view. The requested box center is converted to level-0 coordinates:

```text
cx_level0 = current_x0 + round((cx_999 / 999) * current_w)
cy_level0 = current_y0 + round((cy_999 / 999) * current_h)
```

The final crop is a fixed square:

```text
roi_side_level0_px = ROI_OUTPUT_SIZE_PX
x0 = clamp(cx_level0 - roi_side / 2, 0, slide_w - roi_side)
y0 = clamp(cy_level0 - roi_side / 2, 0, slide_h - roi_side)
```

Duplicate suppression rejects ROIs whose centers are closer than half the ROI
side length to an existing kept ROI:

```text
duplicate if distance(center_new, center_existing) < 0.5 * roi_side
```

AML-specific artifact rejection checks for saturated dark-blue flood:

```text
flood_mask =
  gray < 0.22
  and chroma > 0.18
  and B > R + 0.15
  and B > G + 0.05
  and laplacian_edge < 0.05

reject if mean(flood_mask) > 0.30
```

For early kept AML ROIs, the system auto-saves one centered good tile. This can
later be folded into the dynamic reference index with
`wsi_rebuild_reference_index(include_saved_tiles=True)`.

## Final AML Decision Logic

The final decision is made by the VLM from kept ROI images under strict prompt
constraints.

Primary morphology task:

- estimate blasts among interpretable nucleated hematopoietic cells;
- ignore RBCs, fat, empty/background, and artifacts;
- use exact tiers: `<5%`, `5-9%`, `10-19%`, `20-50%`, `>50%`;
- choose final decision from `Normal marrow` or `Acute leukemia`.

Morphology-only AML rule:

```text
Acute leukemia if blasts >= 20% with widespread immature morphology.
Normal marrow if heterogeneous maturation or blasts < 5%.
For 5-20% suspicious morphology, choose the closest binary label and state
uncertainty/limitations.
```

Blast-like cells require morphology, not color alone:

- high nuclear-to-cytoplasmic ratio;
- round/oval nucleus;
- fine chromatin;
- visible nucleoli, at least in a subset;
- scant cytoplasm;
- Auer rod flags myeloid blast immediately.

NPM1 prediction is strictly gated:

```text
if final_decision != "Acute leukemia":
  npm1_prediction.applicable = false
else:
  classify NPM1_mutated vs NPM1_wildtype using morphology-only supportive
  features such as cup-like nuclear invaginations, folded/irregular nuclei,
  monocytic differentiation, and relatively abundant cytoplasm.
```

## Caching Strategy

The pipeline uses several caches because WSI embedding is the most expensive
stage.

| Cache | Purpose |
| --- | --- |
| Tile cache ZIP | Persist selected image tiles for reuse |
| Feature cache NPZ | Persist tile embeddings, coordinates, and dark ROI scores |
| Reference embedding cache | Persist good/bad reference embeddings |
| Reference HNSW cache | Persist HNSW index for reference lookup |
| ROI index pickle | Persist full slide ROI ranker index |

The ROI index cache key includes:

```text
slide_path
extractor_name
tile_prefilter_method
tile_size_px
aml_mode
```

Feature cache keys include tiler settings, extractor id, AMP mode, cache format,
and code hash. Reference embedding caches are invalidated by reference paths and
file mtimes.

## Novel Approach

The AML agent is novel in how it combines deterministic pathology priors with an
agentic VLM:

1. It treats AML WSI navigation as retrieval-guided microscopy, not as whole
   image classification. The backend proposes compact candidate regions; the VLM
   performs local inspection and final morphology reasoning.

2. It uses ROI-quality references rather than direct disease labels. Good/bad
   tiles teach the system what is diagnostically interpretable, while the final
   AML call remains morphology-only and ROI-based.

3. It avoids the "dark equals AML" trap. Every dark/cellularity score combines
   color, chroma, nuclear signal, edge texture, packing, focus, and artifact
   penalties.

4. It uses a cascade with explicit failure controls. Cheap color/texture
   filters reduce the WSI, embeddings capture semantic similarity, retrieval
   labels quality, and navigation guards prevent wandering, duplicate ROIs, and
   premature stopping.

5. It makes candidate tiles region-level priors rather than final answers. The
   VLM must still inspect the opened field and choose a final ROI with readable
   single-cell morphology.

6. It records an auditable trail: candidate-prep metadata, ROI images, retrieval
   similarities, navigation steps, final JSON, and report artifacts are all
   preserved.

## Practical Configuration Defaults

Important defaults from `configs/config.yaml`:

| Setting | Default | Meaning |
| --- | --- | --- |
| `TILE_FILTER` | `hybrid` | coarse plus raw-tile quality prefilter |
| `TILE_SIZE_PX` | `224` | extractor input tile side |
| `BATCH_SIZE` | `512` | embedding batch size in suite defaults |
| `ROI_SIZE_PX` | `2048` | final ROI crop/output side |
| `ROI_CANDIDATE_TOP_K` | `72` | raw candidates per view |
| `ROI_CANDIDATE_TOP_K_AML` | `30` | AML-facing candidate cap |
| `TARGET_ACCEPTED_ROIS` | `5` | preferred AML ROI count |
| `MAX_ACCEPTED_ROIS` | `5` | hard AML ROI cap |
| `CANDIDATE_NAV_FIELD_UM` | `600.0` | field width for candidate jump |
| `AML_REFERENCE_TOP_K` | `7` | reference neighbors per tile |
| `AML_REFERENCE_AGGREGATION` | `mean` | neighbor similarity aggregation |
| `AML_ABSOLUTE_MIN_DARK_SCORE` | `0.04` | hard dark/cellularity floor |

## Implementation Note

The checked-in code contains documentation strings and UI text that describe
"dark-guided supertile coarse filtering." The lower-level tiling and ranker do
support explicit dark-region boxes as a prior. However, in the main interactive
AML path in `tools.py`, the current call to `build_unsupervised_roi_index()`
passes `dark_region_boxes_level0=None`, and the current call to
`select_topk_candidates_for_view()` passes `focus_boxes_level0=None`. Therefore,
the active pipeline should be understood as:

```text
foreground/coarse/quality tile filtering
  -> per-tile dark/cellularity scores
  -> embedding/reference retrieval
  -> morphology-aware AML candidate ranking
```

with explicit dark-region boxes available as a separate overlay and optional
hook rather than an active hard gate in the main path.
