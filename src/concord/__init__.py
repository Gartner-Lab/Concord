__version__ = "0.9.4"
import logging
import sys

logger = logging.getLogger("Concord")
# check if logger has been initialized
if not logger.hasHandlers() or len(logger.handlers) == 0:
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def set_verbose_mode(verbose):
    if verbose:
        logger.setLevel(logging.INFO)
        for handler in logger.handlers:
            handler.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)
        for handler in logger.handlers:
            handler.setLevel(logging.WARNING)


def lazy_import(module_name, install_instructions=None):
    """
    Lazily import a module. If the module is not installed, log an error or raise an ImportError.

    Parameters:
    - module_name (str): The name of the module to import.
    - install_instructions (str): Optional string to provide install instructions if the module is not found.

    Returns:
    - module: The imported module, if found.
    """
    try:
        return __import__(module_name)
    except ImportError:
        message = f"'{module_name}' is required but not installed."
        if install_instructions:
            message += f" Please install it with: {install_instructions}"
        raise ImportError(message)
        
from . import ml, pl, ul
from .concord import Concord
