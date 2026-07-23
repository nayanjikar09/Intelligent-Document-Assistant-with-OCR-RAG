from __future__ import annotations

import logging
from pathlib import Path
from datetime import datetime
import structlog


class CustomLogger:
    """Creates a reusable structured logger."""

    _configured = False

    def __init__(self, log_dir: str = "logs"):
        self.logs_dir = Path(log_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = (
            self.logs_dir
            / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
        )

    def _configure_structlog(self) -> None:
        if CustomLogger._configured:
            return

        structlog.configure(
            processors=[
                structlog.processors.TimeStamper(fmt="iso", utc=True, key="timestamp"),
                structlog.processors.add_log_level,
                structlog.processors.EventRenamer("event"),
                structlog.processors.JSONRenderer(),
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
        CustomLogger._configured = True

    def get_logger(self, name: str = "app"):
        self._configure_structlog()
        logger = logging.getLogger(name)

        if not logger.handlers:
            logger.setLevel(logging.INFO)
            logger.propagate = False
            formatter = logging.Formatter("%(message)s")

            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)

            file_handler = logging.FileHandler(self.log_file, encoding="utf-8")
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)

            logger.addHandler(console_handler)
            logger.addHandler(file_handler)

        return structlog.get_logger(name)


# Global logger instance
GLOBAL_LOGGER = CustomLogger().get_logger("global")