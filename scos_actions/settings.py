import os
from os import path

BASE_DIR = path.dirname(path.abspath(__file__))
REPO_ROOT = path.dirname(BASE_DIR)

SCOS_ACTIONS_CONFIG_DIR = path.join(REPO_ROOT, "configs")

ACTION_DEFINITIONS_DIR = path.join(SCOS_ACTIONS_CONFIG_DIR, "actions")
