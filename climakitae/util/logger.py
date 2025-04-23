import logging
import sys
from functools import wraps

# TODO this should be built as a singleton to avoid global variables

# Define global variables to control logging
app_log_enabled = False  # For users

# Let user see what the current logging status is
current_logging_status = lambda: app_log_enabled

# Instantiating loggers
logger = logging.getLogger("Climakitae Back-end Logger")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(message)s"))

### Functions to enable or disable logging in notebooks


def enable_app_logging():
    """
    Enables application logging by adding a handler.
    """
    global app_log_enabled
    if not app_log_enabled:
        app_log_enabled = True
        logger.addHandler(handler)


def disable_app_logging():
    """
    Disables logging by removing application logger handler.
    """
    global app_log_enabled
    if app_log_enabled:
        app_log_enabled = False
        logger.removeHandler(handler)
