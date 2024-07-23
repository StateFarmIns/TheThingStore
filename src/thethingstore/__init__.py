"""First time init and high level docs.

The Thing Store API is implemented on top of a FileSystem, but
can describe arbitrary information effectively in a manner that
allows for it to be considered within the same semantic scope.

The layout of this code is as follows:

* `api`: Saving and loading utilities.
* `file_id.py`: This exposes the FILE_ID schema definition and simple utilities.
* `thing_components.py`: This exposes the mathematical and software definition of what a Thing is.
* `thing_node.py`: This exposes the 'ThingNode'; a class which can be dropped onto a fileid to produce
    a lazy-load ticket style representation of a Thing.
* `thing_store_elements.py`: This exposes a default metadata standard implemented in TheThingStore.
    All metadata datasets have this information at minimum.
* `thing_store_log.py`: This exposes the methods by which the ThingStore 'saves' things into a layer.
* `types.py`: Contains extra typing information (deprecate in place of components?)
"""
import logging
from thethingstore.thing_store_base import ThingStore
from thethingstore.thing_store_pa_fs import FileSystemThingStore
from typing import Dict, Type


logger = logging.getLogger(__name__)

####################################################################
#                      Implemented TS Data                         #
####################################################################

_implemented_ts: Dict[str, Type] = {
    "FileSystemThingStore": FileSystemThingStore,
}

__all__ = [
    "FileSystemThingStore",
]

__all__ += ["api", "ThingStore", "_implemented_ts", "file_id"]
