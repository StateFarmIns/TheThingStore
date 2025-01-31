"""Perform basic pointer operations."""

from typing import Union, Optional, Type, Mapping, Callable

from thethingstore._types import Address, FileId, Parameter, Metric, Dataset, Metadata
from thethingstore.api import error as tse
from typing import TYPE_CHECKING
import urllib

if TYPE_CHECKING:
    from thethingstore.thing_store_base import ThingStore


def dereference(  # noqa: C901 - this function does what it needs to
    ts: "ThingStore",  # noqa: F821 - ThingStore isn't defined.
    file_identifier: FileId,
    component: str,
    version: str,
    artifact_identifier: Optional[str] = None,
    artifacts_folder: Optional[str] = None,
) -> Union[
    Dataset, Callable, Mapping[str, Union[Metadata, Parameter, Metric]], Type[None]
]:
    """Dereference the given file ID until the data is found.

    This function performs the following operations iteratively to resolve pointer information to corresponding data:
        * Get the address from the pointer file.
        * Parse that address into meaningful information (file ID, version, TS implementation, params).
        * Spin up a TS instance from that information.
        * Check if the pointed-to file is, itself, a pointer.
            * If yes, loop to the start.
            * If no, get the component requested.
            * Return the component information in the same form as would the corresponding getter function.

    The component argument must be a str corresponding to one of the implemented ThingPointer components:
        * data
        * parameters
        * metrics
        * artifacts
        * function

    The following components are not implemented as ThingPointers:
        * metadata - Every Thing should have it's own set of metadata. Metadata cannot, nor should, be inherited.
        * embeddings - Every Thing should have a unique embedded representation - even pointers.

    If "artifacts" is passed as the component:
        * If artifacts_folder is not given, raise an Exception.
        * Else if artifacts_folder and artifacts_identifier are given, copy that artifact into the artifacts_folder.
        * Else if artifacts_folder and not artifacts_identifier, copy all artifacts into the artifacts_folder.

    Parameters
    ----------
    ts: 'ThingStore'
        This is a ThingStore compliant data layer.
    file_identifier: FileId
        This is the file ID that contains the pointer.
    component: str
        This is the pointer component.
    version: str
        This is the file version.
    artifact_identifier: Optional[str]
        This is artifact to copy into the artifacts_folder.
    artifacts_folder: Optional[str]
        This is the target folder to copy artifacts into.

    Returns
    -------
    new_data: Union[Dataset, Callable, Mapping[str, Union[Metadata, Parameter, Metric]], Type[None]]
        This is the data pointed to by the pointer chain. If the data is a set of artifacts, they are copied
        into the artifacts_folder and None is returned.
    """
    if not ts._check_file_id(file_identifier):
        raise tse.ThingStoreFileNotFoundError(file_identifier)
    if not ts._is_pointer(file_identifier, component, version):
        raise tse.ThingStorePointerError(
            file_identifier, component, "This file is not a pointer"
        )

    # Get address stored at file ID
    _new_addr = ts._get_pointer(file_identifier, component, version)
    _parsed = urllib.parse.urlparse(_new_addr)
    _new_fileid = _parsed.netloc
    _new_version = _parsed.fragment
    # Get TS from that address
    _new_ts = ts.address_to_ts(_new_addr)

    _new_data = ""
    while isinstance(_new_data, Address):
        # If it's pointing at another pointer, return the new pointer
        if _new_ts._is_pointer(_new_fileid, component, _new_version):
            _new_data = _new_ts._get_pointer(_new_fileid, component, _new_version)
            _parsed = urllib.parse.urlparse(_new_data)
            _new_fileid = _parsed.netloc
            _new_version = _parsed.fragment
            _new_ts = ts.address_to_ts(_new_data)
        # Otherwise get the data
        elif component == "data":
            return _new_ts.get_dataset(_new_fileid, version=_new_version)
        elif component == "parameters":
            return _new_ts.get_parameters(_new_fileid, version=_new_version)
        elif component == "metadata":
            raise tse.ThingStoreNotAllowedError(
                "dereferencing metadata", "Metadata cannot be a pointer"
            )
        elif component == "metrics":
            return _new_ts.get_metrics(_new_fileid, version=_new_version)
        elif component == "artifacts":
            if not artifacts_folder:
                raise tse.ThingStoreGeneralError(
                    "Dereferencing artifacts missing a target folder"
                )
            if not artifact_identifier:
                _new_ts.get_artifacts(
                    _new_fileid, artifacts_folder, version=_new_version
                )
                return None
            else:
                _new_ts.get_artifact(
                    _new_fileid,
                    artifact_identifier,
                    artifacts_folder,
                    version=_new_version,
                )
                return None
        elif component == "embedding":
            raise tse.ThingStoreNotAllowedError(
                "dereferencing embedding", "Embedding cannot be a pointer"
            )
        elif component == "function":
            return _new_ts.get_function(_new_fileid, version=_new_version)
        else:
            raise NotImplementedError(
                f"The component type is not currently implemented: {component}"
            )
    return None
