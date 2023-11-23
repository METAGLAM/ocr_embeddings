import logging
import os
from typing import Dict

import sys
import yaml


def load_settings() -> Dict:
    """
    Read settings file from config/settings.yaml and it returns as dict.

    Returns:
        a dict with settings params.
    """
    settings_file_path = os.path.join("/config", "settings.yaml")
    if not os.path.exists(settings_file_path):
        settings_file_path = os.path.join("config", "settings.yaml")

    with open(settings_file_path, encoding="utf8") as file:
        params = yaml.safe_load(file)

    return params


def initialize_logger(params: Dict) -> logging:
    """
    Given settings dict, this function initialize logging system.

    Args:
        params (Dict): settings dict

    Returns:
        logger
    """
    log_params = params["logging"]

    _logger = logging.getLogger(__name__)
    hdlr_out = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        log_params["formatter"]["format"],
        log_params["formatter"]["time_format"])
    hdlr_out.setFormatter(formatter)
    _logger.addHandler(hdlr_out)
    _logger.setLevel(getattr(logging, log_params["level"]))
    _logger.propagate = False

    # If it contains log file, initialize it
    if log_params.get('file') is not None:
        log_filename = log_params['file']
        hdlr_file = logging.FileHandler(log_filename)
        hdlr_file.setFormatter(formatter)
        _logger.addHandler(hdlr_file)

    _logger.info("Logger initialized")

    return _logger


# Startup functions
settings = load_settings()
logger = initialize_logger(settings)
