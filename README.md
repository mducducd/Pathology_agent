# Slide Agent
WSI agent using embedding retrieval + VLM reasoning for robust, explainable ROI ranking.

## Install:

### Environment:

```bash
uv sync

source .venv/bin/activate
```

### Configure .env to your needs:

```bash
cp .env.example .env
```


### Configure Model Name:

In the `wsi_core.py` and `main.py` files, set the `MODEL_NAME` and update `ALLOWED_MODEL_NAMES` variable to your desired VLM.


### Run:

```bash
python main.py
```
