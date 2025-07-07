import logging
from logging.handlers import RotatingFileHandler
import os

from config.settings import settings

# env = os.getenv('ENV', 'DEV')
env = settings.env
# log_dir_path = os.path.join(os.getenv('LOG_DIR'), os.getenv('LOG_FILE'))
log_dir_path = os.path.join(settings.log_dir, settings.log_file)

def get_loggers(name:str='reas') -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG if env == 'DEV' else logging.INFO)
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        file_handler = RotatingFileHandler(
            filename=log_dir_path,
            maxBytes=(settings.log_size * 1024 * 1024),
            backupCount=3
        )
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    return logger