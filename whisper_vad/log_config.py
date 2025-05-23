import logging.config
import json
import os
from pathlib import Path

LOGGING_EXAMPLE = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '{asctime}:{name}:{levelname}:{message}',
            'style':"{",
            'datefmt':"%Y-%m-%d %H:%M"
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
            'level': 'DEBUG',
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': 'app.log',
            'formatter': 'standard',
            'level': 'INFO',
            'backupCount': 5,
            'maxBytes': 10485760,  # 10 MB
        },
    },
    'root': {
        'handlers': ['console', 'file'],
        'level': 'DEBUG',
    },
    'loggers': {
        'tasks': {
            'level': 'DEBUG',
            'handlers': ['console'],
            'propagate': False,
        },
        'multiproc': {
            'level': 'DEBUG',
            'handlers': ['console'],
            'propagate': False,
        }
    }
}

def setup_logging(config_dir: str = "config/"):
    """Setup logging configuration from a JSON file

    :param config_dir: Directory with log_config.json file, defaults to "config/"
    :type config_dir: str, optional
    """
    config_path = Path(config_dir)
    if not Path("log/").exists():
        os.makedirs("log")
    with open(config_path.joinpath("log_config.json"), mode="r") as f:
        LOGGING_CONFIG = json.load(f)
    logging.config.dictConfig(LOGGING_CONFIG)
