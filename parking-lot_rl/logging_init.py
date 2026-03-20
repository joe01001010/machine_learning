import logging
from pathlib import Path


def configure_logging(source_file, log_dir='logs', level=logging.INFO):
    """
    Configure a shared file logger for the current process and return a
    module-specific logger.
    """
    source_path = Path(source_file).resolve()
    module_name = source_path.stem
    logs_path = Path(log_dir).resolve()
    logs_path.mkdir(parents=True, exist_ok=True)
    log_file = logs_path / f"{module_name}.log"

    logger = logging.getLogger(module_name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger