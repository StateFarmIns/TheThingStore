"""Initialization for the first time!"""
import logging
from thethingstore.thing_store_base import ThingStore
from thethingstore.thing_store_pa_fs import FileSystemThingStore
from typing import Dict, Type


logger = logging.getLogger(__name__)

####################################################################
#                      Conditional Imports                         #
# Todo - Move these into their independent loading components.     #
####################################################################

try:
    import sklearn  # noqa: F401
except ImportError:
    logger.warning(
        """SKLearn is not installed.
    If you wish to save and load scikit-learn models please run
    `pip install thethingstore[models]`.
    """
    )

try:
    import joblib  # noqa: F401
except ImportError:
    logger.warning(
        """Joblib is not installed.
    If you wish to save and load scikit-learn models please run
    `pip install thethingstore[models]`.
    """
    )

try:
    import geopandas  # noqa: F401
except ImportError:
    logger.warning(
        """GeoPandas is not installed.
    If you wish to save and load shapes / layers please run
    `pip install thethingstore[shapes]`.
    """
    )

try:
    import mlflow  # noqa: F401
    from thethingstore.thing_store_mlflow import MLFlowThingStore

    _mlflow = True
except ImportError:
    logger.warning(
        """MLFlow is not installed.
    If you wish to save and load information from an MLFlow Thing Store
    please run
    `pip install thethingstore[mlflow]`.
    """
    )
    _mlflow = False


####################################################################
#                      Implemented TS Data                         #
####################################################################

_implemented_ts: Dict[str, Type] = {
    "FileSystemThingStore": FileSystemThingStore,
}

if _mlflow:

    _implemented_ts.update(MLFlowThingStore=MLFlowThingStore)
    __all__ = [
        "FileSystemThingStore",
        "MLFlowThingStore",
    ]
else:
    __all__ = [
        "FileSystemThingStore",
    ]

__all__ += ["api", "ThingStore", "_implemented_ts", "file_id"]
