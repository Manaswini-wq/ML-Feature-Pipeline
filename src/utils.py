import os
import yaml
from dotenv import load_dotenv

load_dotenv()

def load_config(path="config/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

def get_env(key):
    val = os.environ.get(key)
    if not val:
        raise EnvironmentError(f"Missing env variable: {key}")
    return val
