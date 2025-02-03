"""File ID schema utilities.

Utilities
~~~~~~~~~
parse_fileid: A small utility to unpack a fileid and run it through
    a pydantic data model.

Data Models
~~~~~~~~~~~
FileId: This is a pydantic data model.
"""

import urllib.parse
from thethingstore.thing_store_base import _implemented_ts
from thethingstore.api.error import FileIDError, ThingStoreGeneralError as TSGE
from pyarrow.fs import LocalFileSystem, S3FileSystem
from pydantic import BaseModel, Field, validator, root_validator
from typing import Any, Dict, Optional, Union


def parse_fileid(fileid: str) -> Dict[str, Union[str, Dict[str, str]]]:
    """Extract FileID Components.

    A FileID can be sufficiently self-descriptive such that an
    instance of a ThingStore may be dynamically created to
    load the FileID.

    FileID uses standard URL description in order to extract
    components and these are described here, along with their
    Thing Store components, for documentation.

    Structure: `<scheme>//<netloc>/<path>/#<fragment>/?<query>`

    Example of an HTTP file schema: `https://www.example.com/path/to/page#section2?param1=value1&param2=value2`
    Example of a FILEID file schema:
        `fileid://MyMadeUpFile/FileSystemThingStore?metadata_filesystem=LocalFileSystem&managed_location=.#1`

    More examples of qualified fileids may be found in the examples
    and within the tests.

    1. scheme: The scheme represents the protocol or application
        layer protocol that the URL uses. Examples of schemes
        include "http," "https," "ftp," "file," etc. It is the
        part before the colon (":") in the URL. For a fileid the
        scheme should be 'fileid'.
        Example: `https://www.example.co` -> `https`

    2. netloc: The netloc refers to the network location of the URL,
        which typically includes the domain name and, optionally,
        the port number. It is the part between the double slashes ("//")
        and the next forward slash ("/") or the end of the URL.
        Currently this only holds the FILEID and is expected to
        be case sensitive and unique.
        Example: `https://www.example.com:8080/path/to/page` -> `www.example.com:8080`

    3. path: The path represents the specific resource or location
        on the server identified by the URL. It comes after the
        netloc and typically begins with a forward slash ("/").
        For a FILEID this will contain the full class name of the
        implemented Thing Store.
        Example: `https://www.example.com/path/to/page` -> `/path/to/page`

    4. params: This is not used.

    5. query: The query component contains a set of parameters or
        data passed to the server as part of the URL. It follows a
        question mark ("?") and consists of key-value pairs
        separated by ampersands ("&"). For a FILEID this contains
        the parameters used to 'turn on' that specific Thing Store.
        Example: `https://www.example.com/path/to/page?param1=value1&param2=value2`
        -> `param1=value1;param2=value2`

    6. fragment: The fragment component represents a specific
        section or anchor within the document that the URL points
        to. It is preceded by a hash symbol ("#") and is typically
        used in web pages to navigate to a specific section.
        Example: `https://www.example.com/path/to/page#section2` -> `section2`
        For a FILEID this is used to specify a file version.

    Parameters
    ----------
    fileid: FileId
        This is a fileid, which is a unique 'name' for a process
        node component.

    Returns
    -------
    componentized_fileid: Dict[str, Union[str, Dict[str, str]]]
        This returns a componentized FileId

    Examples
    --------
    Simple examples
    >>> parse_fileid('fileid://bobby')
    {'fileid': 'bobby'}
    >>> parse_fileid('fileid://ExampleFileId/FileSystemThingStore')
    {'fileid': 'ExampleFileId', 'implementation': 'FileSystemThingStore'}
    >>> parse_fileid('fileid://ExampleFileId/FileSystemThingStore#2')
    {'fileid': 'ExampleFileId', 'implementation': 'FileSystemThingStore', 'fileversion': '2'}

    Complex example!
    >>> import tempfile
    >>> from thethingstore import FileSystemThingStore as FSTS
    >>> import os
    >>> with tempfile.TemporaryDirectory() as t:
    ...     ts = FSTS(
    ...         metadata_filesystem=LocalFileSystem(),
    ...         managed_location=t,
    ...     )
    ...     flid = ts.log(metadata={'silly': 'rabbit'})
    ...     file_id_bits = [
    ...         'fileid://',  # Schema
    ...         flid,  # File ID
    ...         '/',  # Separator
    ...         'FileSystemThingStore',  # Implementation
    ...         '?',  # Separator for params
    ...         'metadata_filesystem=LocalFileSystem',
    ...         f'&managed_location={t}',
    ...         '#1',  # File version
    ...     ]
    ...     components = parse_fileid(''.join(file_id_bits))
    ...     assert components['fileid'] == flid
    ...     assert components['implementation'] == 'FileSystemThingStore'
    ...     assert isinstance(components['implementation_params'], dict)
    ...     _par = components['implementation_params']
    ...     assert set(_par.keys()) == {'metadata_filesystem', 'managed_location'}
    ...     assert components['fileversion'] == '1'
    """
    output: Dict[str, Union[str, Dict[str, str]]] = {}
    parse_result = urllib.parse.urlparse(fileid)
    # This has a scheme, a netloc (filepath), a path,
    #   params, a query, and a fragment.
    # The scheme *should* say that it's a file id.
    if not parse_result.scheme.lower() == "fileid":
        raise FileIDError(
            fileid, "Not appropriate schema. FileId should lead with fileid://"
        )
    # The netloc should be the fileid.
    if not parse_result.netloc:
        raise FileIDError(fileid, "No FileId present.")
    output["fileid"] = parse_result.netloc
    # The path should be the Thing Store implementation
    if parse_result.path:
        output["implementation"] = parse_result.path[1:]
    # The query should be unpacked.
    if parse_result.query:
        # This is a dictionary of keys with list values.
        # For now these are all atomic.
        _unpacked_params = urllib.parse.parse_qs(parse_result.query)
        unpacked_params: Dict[str, str] = {}
        for key, value in _unpacked_params.items():
            if len(value) == 1:
                unpacked_params[key] = value[0]
            else:  # pragma: no cover
                raise NotImplementedError("Please reach out to the dev team.")
        output["implementation_params"] = unpacked_params
    # The fragment should be unpacked
    if parse_result.fragment:
        output["fileversion"] = parse_result.fragment
    return output


_fsmap = {
    "LocalFileSystem": LocalFileSystem,
    "S3FileSystem": S3FileSystem,
}


class FileID(BaseModel):
    """DataModel representing a FileID promise.

    This data model intelligently validates and represents a FileID promise.

    For an understanding of what this is intended to accomplish, see
    `parse_fileid`; this is the bit that's identifying the
    components and doing some *very* minor checking.
    """

    fileid: str = Field(..., description="The unique FILEID")
    fileversion: Optional[Union[str, int]] = Field(
        None, description="An optional file version."
    )
    implementation: Optional[Any] = Field(
        ..., description="The ThingStore class to use."
    )
    implementation_params: Optional[Dict[str, Union[str, Any]]] = Field(
        ..., description="Parameters to instantiate the Thing Store class."
    )

    @validator("fileversion")
    def convert_fileversion(
        cls: BaseModel, val: Optional[str] = None
    ) -> Optional[Union[int, str]]:
        """Convert FILE_VERSION."""
        if val is None:  # pragma: nocover
            return val
        try:
            return int(float(val))
        except ValueError:  # This accounts for non-numeric string.
            return val

    @validator("implementation_params")
    def convert_implementation_params(
        cls: BaseModel, val: Optional[Dict[str, Union[str, type]]]
    ) -> Optional[Dict[str, Union[str, Any]]]:
        """Convert implementation parameters."""
        if val is None:  # pragma: nocover
            return val
        for key, value in val.items():
            if key == "metadata_filesystem":
                if value not in _fsmap.keys():
                    raise NotImplementedError(value)
                assert isinstance(val["metadata_filesystem"], str)  # nosec
                val["metadata_filesystem"] = _fsmap[val["metadata_filesystem"]]()
        return val

    @validator("implementation")
    def convert_implementation(cls: BaseModel, val: str) -> Any:
        """Convert implementation."""
        if val not in _implemented_ts:
            raise TSGE(
                f"""Not an accessible Thing Store implementation: {val}

            If this type of Thing Store exists, please ensure that the
            requirements are installed.

            Available Flavors of Thing Store
            -----------------------------\n{list(_implemented_ts.keys())}
            """
            )
        return _implemented_ts[val]

    # This mypy validation is brought to you by 'Adding twenty lines of code to
    #   clarify things is not worth it.' This is *pydantic* code, not mine.
    # All this does is simply try to grab the bits that can make an TS.
    # It does zero validation to determine whether or not they're actually
    #   going to work.
    @root_validator(pre=True)
    def validate_ts(cls: BaseModel, values: Dict[str, Any]):  # type: ignore
        """Validate implementation."""
        if "implementation_params" not in values:
            raise NotImplementedError(
                "FileID Schema must have `implementation_params`."
            )
        if "implementation" not in values:
            raise NotImplementedError("FileID Schema must have `implementation`.")

        return values
