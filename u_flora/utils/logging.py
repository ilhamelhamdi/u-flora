import logging

_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """Configure the u_flora package logger.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ...).
        log_file: Optional path to a dedicated log file. When provided, log
            records are written to both stderr and this file, allowing the
            caller to have its own log separate from the inherited stderr stream.
    """
    pkg_logger = logging.getLogger("u_flora")
    if pkg_logger.handlers:
        return  # already configured in this process

    fmt = logging.Formatter(_LOG_FORMAT)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    pkg_logger.addHandler(stream_handler)

    if log_file:
        import os

        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(fmt)
        pkg_logger.addHandler(file_handler)

    pkg_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    pkg_logger.propagate = False  # don't double-emit if root is later configured
