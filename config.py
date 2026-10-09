import json
import os

BASEDIR = os.path.dirname(os.path.abspath(__file__))


def load_config():
    # personal settings live in config.json (not versioned), config.example.json holds the defaults
    with open(os.path.join(BASEDIR, "config.example.json"), encoding="utf-8") as f:
        config = json.load(f)
    local = os.path.join(BASEDIR, "config.json")
    if os.path.exists(local):
        with open(local, encoding="utf-8") as f:
            config.update(json.load(f))
    return config


def path(name):
    return os.path.join(BASEDIR, name)


CONFIG = load_config()
