import json
import os

BASEDIR = os.path.dirname(os.path.abspath(__file__))
# personal data (config.json, history, /input, /model, outputs) can live in a separate folder
DATADIR = os.path.abspath(os.environ.get("BANKSTATEMENT_DATA") or BASEDIR)


def load_config():
    # personal settings live in config.json (not versioned), config.example.json holds the defaults
    with open(os.path.join(BASEDIR, "config.example.json"), encoding="utf-8") as f:
        config = json.load(f)
    local = os.path.join(DATADIR, "config.json")
    if os.path.exists(local):
        with open(local, encoding="utf-8") as f:
            config.update(json.load(f))
    return config


def path(name):
    return os.path.join(DATADIR, name)


CONFIG = load_config()
