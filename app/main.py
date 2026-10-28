"""Entry point for the Local AI Interview Assistant."""

from __future__ import annotations

import logging
import sys

from PySide6.QtWidgets import QApplication

from app.config import APP_TITLE
from app.ui.main_window import MainWindow


def setup_logging() -> None:
    """Configure application-wide logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    # Quieten noisy libraries
    logging.getLogger("faster_whisper").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def main() -> int:
    """Launch the application."""
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting %s", APP_TITLE)

    app = QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)

    window = MainWindow()
    window.show()

    logger.info("Application window shown — entering event loop")
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
