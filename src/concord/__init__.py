__version__ = "1.0.7"
import logging
import sys

PKG_LOGGER_NAME = __name__                # "concord"
logger = logging.getLogger(PKG_LOGGER_NAME)

if not logger.handlers:                   # configure only once
    logger.setLevel(logging.INFO)         # default = verbose
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "%(name)s - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S"
        )
    )
    logger.addHandler(handler)
    logger.propagate = False              

def get_logger(name: str | None = None) -> logging.Logger:
    """
    Return the package root logger (`concord`) or one of its children.

    >>> from concord import get_logger
    >>> logger = get_logger(__name__)
    """
    if name is None or name == PKG_LOGGER_NAME:
        return logger
    return logger.getChild(name.split("concord.", 1)[-1])

def set_verbose_mode(verbose: bool = True):
    """
    Toggle INFO/DEBUG messages for the whole package.

    Parameters
    ----------
    verbose : bool
        True  → log level INFO (default)  
        False → log level WARNING
    """
    level = logging.INFO if verbose else logging.WARNING
    logger.setLevel(level)
    for h in logger.handlers:
        h.setLevel(level)
        

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
        
from . import ml, pl, ul, bm, sm
from .concord import Concord
