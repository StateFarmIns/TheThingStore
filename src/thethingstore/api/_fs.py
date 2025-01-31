"""Filesystem hidden utilities."""

import urllib
from pyarrow.fs import LocalFileSystem, FileSystem, FileSelector
from typing import Any, Dict, List, Tuple, Union


def get_fs(
    filepath: Union[str, List[str], Dict[str, Any]]
) -> Tuple[Union[str, List[str], Dict[str, Any]], FileSystem]:
    """Map filesystem if possible.

    This takes a url with a schema and unpacks that.
    This can understand things like 's3://path' and
    'file://path' and others.

    Parameters
    ----------
    filepath: Union[str, List[str], Dict[str]]
        Filepaths to investigate.

    Returns
    -------
    filepath, filesystem: Tuple[Union[str, List[str], Dict[str, Any]], Union[FileSystem, Type[None]]]
        This returns a *potentially* modified filepath (if a file
        system is identified via schema the schema is removed) and
        a filesystem if it can be interpreted, None otherwise.
    """
    if isinstance(filepath, str):
        parsed = urllib.parse.urlparse(filepath)
        if not parsed.scheme:
            return filepath, LocalFileSystem()
        else:
            fs, newpath = FileSystem.from_uri(filepath)
            return newpath, fs
    elif isinstance(filepath, Dict):
        return filepath, LocalFileSystem()
    elif isinstance(filepath, List):
        return filepath, LocalFileSystem()
    else:
        raise NotImplementedError


def ls_dir(path: str, filesystem: FileSystem) -> Union[str, List[str]]:
    """List one level deep.

    If this is a filepath for a file you just get that path back,
    but if you call this on a directory with multiple paths
    underneath it you will get those back.

    This is equivalent in behavior to `ls dir`.

    Please see the pyarrow filesystem documentation for examples on
    paths to use within specific filesystems. This was built primarily
    to accomodate a single api for various filesystem with consistent
    behavior.

    If you explicitly pass a filesystem of None this will attempt
    to infer the filesystem.

    Parameters
    ----------
    path: str
        A filepath.
    filesystem: FileSystem
        The filesystem for the path.

    Returns
    -------
    filepath(s): Union[str, List[str]]
        The filepath, or paths subordinate to.
    """
    file_info = filesystem.get_file_info(path)
    if not file_info.type.name == "Directory":
        return path
    return [_.path for _ in filesystem.get_file_info(FileSelector(path))]
