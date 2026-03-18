# Slide Agent

![Slide Agent overview](static/assets/overview.png)

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

In the `wsi_core.py` and `amin.py` file, set the `MODEL_NAME` and `ALLOWED_MODEL_NAMES` variable to your desired model name.


### Run:

```bash
python main.py
```
