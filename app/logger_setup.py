import logging
from logging.handlers import RotatingFileHandler

def setup_custom_logger(name, log_file='app/log', log_level=logging.DEBUG):
    """
    Set up a custom logger with a console and file handler.

    :param name: Name of the logger
    :param log_file: File to save the logs
    :param log_level: Logging level
    :return: Configured logger
    """
    # Create a custom logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Create formatters
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s - %(lineno)d - %(message)s'
    )

    # Create console handler and set level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)

    # Create file handler with rotation and set level
    file_handler = RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=3
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(file_formatter)

    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger