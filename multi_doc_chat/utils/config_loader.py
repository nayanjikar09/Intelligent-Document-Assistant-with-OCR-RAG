"""
Configuration loader for the Intelligent Document Assistant project.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any

import os
import yaml


def get_project_root() -> Path:
    """
    Returns the project root directory.

    Example:
    multi_doc_chat/
        config/
        utils/
    """
    return Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def load_config(config_path: str | None = None) -> dict[str, Any]:
    """
    Load YAML configuration.

    Priority:
    1. config_path argument
    2. CONFIG_PATH environment variable
    3. config/config.yaml

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """

    path = (
        Path(config_path)
        if config_path
        else Path(
            os.getenv(
                "CONFIG_PATH",
                get_project_root() / "config" / "config.yaml",
            )
        )
    )

    if not path.is_absolute():
        path = get_project_root() / path

    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found:\n{path}"
        )

    try:
        with path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}

        if not isinstance(config, dict):
            raise ValueError(
                "Configuration file must contain a YAML dictionary."
            )

        return config

    except yaml.YAMLError as e:
        raise ValueError(
            f"Invalid YAML configuration:\n{e}"
        ) from e