# Reference Embeddings Guide

This guide explains how to pre-build reference embeddings from curated tile images for fast retrieval during AML agent inference.

## Overview

The AML agent uses reference-based retrieval to rank ROI candidates by similarity to curated "good" (AML-like) and "bad" (normal marrow) tile examples. Pre-building embeddings avoids re-extracting them every session, significantly reducing startup time.

**Benefits:**
- **Fast startup**: Load embeddings in seconds instead of minutes
- **Consistent results**: Same embeddings across sessions
- **Extractor flexibility**: Use different extractors (reddino, dinobloom, uni2)

## Quick Start

### Step 1: Organize Your Tiles

Place curated tiles in the following directory structure:

```
Selected_Tiles/
├── Good_Tiles/       # AML-like tiles (high blast %)
│   ├── tile_001.jpg
│   ├── tile_002.jpg
│   └── ...
└── Bad_Tiles/        # Normal marrow tiles
    ├── tile_001.jpg
    ├── tile_002.jpg
    └── ...
```

Supported formats: `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`

### Step 2: Pre-Build Embeddings

```bash
# Using reddino (recommended - fast and lightweight)
python -m wsi_core_pkg.embeddings.prebuild_reference_embeddings \
    --tiles-root ./Selected_Tiles \
    --output-dir ./outputs/cache/reference_hnsw \
    --extractor reddino

# Using uni2 (slower but more accurate)
python -m wsi_core_pkg.embeddings.prebuild_reference_embeddings \
    --tiles-root ./Selected_Tiles \
    --output-dir ./outputs/cache/reference_hnsw \
    --extractor uni2
```

### Step 3: Run the AML Agent

The agent automatically loads pre-built embeddings if available:

```bash
# Start your AML agent session
# It will use the cached embeddings from ./outputs/cache/reference_hnsw
```

## Command Reference

### prebuild_reference_embeddings

**Usage:**
```bash
python -m wsi_core_pkg.embeddings.prebuild_reference_embeddings [OPTIONS]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--tiles-root` | `./Selected_Tiles` | Root directory containing Good_Tiles and Bad_Tiles folders |
| `--output-dir` | `./outputs/cache/reference_hnsw` | Directory to save embeddings and metadata |
| `--batch-size` | `32` | Batch size for embedding extraction |
| `--device` | `cuda` (if available) | torch device (e.g., `cuda`, `cuda:0`, `cpu`) |
| `--extractor` | `reddino` | Extractor to use: `reddino`, `dinobloom`, `uni2` |

## Output Files

The script produces:

1. **`{extractor_id}_prebuilt_embeddings.npy`**: L2-normalized embeddings array of shape `(n_tiles, embedding_dim)`
2. **`{extractor_id}_prebuilt_meta.json`**: Metadata including:
   - `extractor_id`: Extractor identifier
   - `count`: Number of tiles
   - `dim`: Embedding dimension
   - `fingerprint`: Cache invalidation hash
   - `paths`: List of tile file paths
   - `labels`: List of labels ("good" or "bad")
   - `good_count`, `bad_count`: Tile counts per class

Example output:
```
outputs/cache/reference_hnsw/
├── RedDino-Small_prebuilt_embeddings.npy    # 43 tiles x 384 dims
├── RedDino-Small_prebuilt_meta.json         # Metadata
├── RedDino-Small_35476a7c10ecf455_embeddings.npy  # Fingerprinted copy
├── RedDino-Small_35476a7c10ecf455_labels.npy
└── RedDino-Small_35476a7c10ecf455_meta.json
```

## Environment Variables

Configure retrieval behavior via environment variables:

```bash
# HNSW indexing
export AML_REFERENCE_USE_HNSW=true          # Enable HNSW (default: auto)
export AML_REFERENCE_HNSW_M=32              # HNSW M parameter
export AML_REFERENCE_HNSW_EF_CONSTRUCTION=200
export AML_REFERENCE_HNSW_EF_SEARCH=100

# Retrieval
export AML_REFERENCE_TOP_K=5                # Number of neighbors to retrieve
export AML_REFERENCE_AGGREGATION=mean       # Aggregation: mean, max, weighted
export AML_REFERENCE_LOGIT_SCALE=4.0        # Sigmoid scale for bad_likelihood

# Caching
export AML_REFERENCE_CACHE_DIR=./outputs/cache/reference_hnsw

# Tissue filtering
export AML_MIN_DARK_SCORE_PERCENTILE=30.0   # Minimum cellularity threshold
export AML_DARK_PRIOR_WEIGHT=0.55           # Weight on dark_roi_scores
```

## Extractor Comparison

| Extractor | Dimension | Speed | Accuracy | VRAM Usage |
|-----------|-----------|-------|----------|------------|
| **reddino** | 384 | Fast | Good | Low |
| dinobloom | 768 | Medium | Good | Medium |
| uni2 | 1024 | Slow | Best | High |

**Recommendation:** Use `reddino` for development and rapid iteration. Switch to `uni2` for production if accuracy is critical.

## Cache Invalidation

The cache is automatically invalidated when:
- Tile file paths change
- New tiles are added to `Selected_Tiles/`
- Different extractor is used

To manually clear the cache:
```bash
rm -rf ./outputs/cache/reference_hnsw/
```

## Dynamic Prototype Updates

During a session, you can incorporate newly saved tiles into the reference bank:

```python
# In the AML agent, after saving tiles:
wsi_rebuild_reference_index(include_saved_tiles=True)
```

This rebuilds the HNSW index with tiles saved via `wsi_save_tile_norm`.

## Troubleshooting

### "No image tiles found"
- Check that `Selected_Tiles/Good_Tiles` and `Selected_Tiles/Bad_Tiles` directories exist
- Ensure tile images have supported extensions (`.jpg`, `.png`, etc.)

### "CUDA out of memory"
- Reduce batch size: `--batch-size 16`
- Use CPU: `--device cpu`
- Switch to reddino extractor: `--extractor reddino`

### Slow embedding extraction
- Use reddino instead of uni2
- Pre-build embeddings once and reuse the cache
- Ensure GPU is being used: check `--device cuda`

### Cache not loading
- Verify the cache directory: `ls ./outputs/cache/reference_hnsw/`
- Check that extractor_id matches (e.g., `RedDino-Small` vs `uni2`)
- Clear cache and rebuild: `rm -rf ./outputs/cache/reference_hnsw/ && python -m ...`

## See Also

- [README.md](README.md) - Main project documentation
- [wsi_core_pkg/embeddings/roi_ranker.py](wsi_core_pkg/embeddings/roi_ranker.py) - Retrieval implementation
- [wsi_core_pkg/tools.py](wsi_core_pkg/tools.py) - `wsi_rebuild_reference_index` tool
