# Slide Agent

![Slide Agent overview](static/assets/overview.png)

## Install

### Environment

```bash
uv sync

source .venv/bin/activate
```

### Configure `.env` to your needs

```bash
cp .env.example .env
```

### Configure model name

In `wsi_core.py` and `main.py`, set `MODEL_NAME` and `ALLOWED_MODEL_NAMES` to the models you want exposed in the UI.

### Configure server-side slide roots

The Explorer modal can browse server-local slide roots directly, so large HPC WSIs do not need to be uploaded through the browser.

By default it exposes:

```text
/mnt/copernicus3/PATHOLOGY/others/private/haemadata/ALL_WSIs/
```

To add more roots, set `SERVER_SLIDE_ROOTS` as a colon-separated list before starting the app:

```bash
export SERVER_SLIDE_ROOTS="/mnt/copernicus3/PATHOLOGY/others/private/haemadata/ALL_WSIs/:/some/other/root"
```

The Explorer supports:

- Standard slide files: `.svs`, `.tif`, `.tiff`, `.ndpi`
- MIRAX files: `.mrxs`, `.mrsx`
- MIRAX companion folders: select the folder whose sibling `.mrxs`/`.mrsx` has the same stem

## Run

```bash
python main.py
```
