import logging
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "flirtio",
    log_file: Optional[str] = "logs/run.log",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Create and configure a logger that logs to console and (optionally) to file.
    Call this once per module: logger = setup_logger(__name__).
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # avoid duplicate handlers
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # console
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # file
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger