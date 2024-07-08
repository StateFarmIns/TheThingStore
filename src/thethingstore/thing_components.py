"""What is a Thing?

A Thing can, quite simply and apologies for the tongue-in-cheek, be *anything*.

Problem is, that makes it *mighty* hard to describe in math, sometimes.
See: Deep learning and multi-modal input.

There's a lot of value in being a little more p(y)edantic about
how we describe data which we shove into automated processes.

(Note that `types.py` is more up to date if you're looking for static
typing, but the general thoughts and specific components here
are appropriate.)

Thing Components
~~~~~~~~~~~~~~~~
FileId: str
    This is a string, albeit one with expected formatting.
    Please see `file_id.py` for more information.
Parameters:
    A potentially arbitrary key-value mapping structure.
Metadata:
    A key-value mapping structure expected to have *atomic* elements.
Metric:
    A key-value mapping structure expected to have *measure* elements.
Dataset:
    This represents a Table of information.
Function:
    A python file with a workflow function.
Embedding:
    A matrix of arbitrary dimensionality
Artifacts:
    Look... stuff!

"""
from pydantic import BaseModel, Extra


class Atomic(BaseModel):
    """DataModel representing an integer, a float, or a string.

    An atomic element is the fundamentally lowest level of information
    representable within a system.
    """

    pass


class Complex(BaseModel):
    """DataModel representing an arbitrary container of Atomic elements."""

    pass


class Dataset(BaseModel, extra=Extra.allow):
    """DataModel representing a tabular dataset.

    A Thing can have a dataset component.
    A dataset can be described in many ways, though the most
    convenient ways of describing data lie below:

    * A string which is representative of a file to be loaded. This is an atomic value.
    * A complex data structure which is representative of labeled partitions of data.

    Here are some concrete examples which you could use in Python.
    Note that all these examples use parquet.
    This is intentional.
    Please reach out to data stewards for more information.

    Examples
    --------
    Examples with atomic values:

    >>> 'myfolder/myfile.parquet'
    'myfolder/myfile.parquet'
    >>> 's3://my/most/favoritest/prefix/whateverfolderfullofparquet'
    's3://my/most/favoritest/prefix/whateverfolderfullofparquet'
    >>> 'fileid://whateverstuff'
    'fileid://whateverstuff'


    Complex examples:

    ```
    >>> {'2028 loss data': 'myfolder/loss1.parquet', '2029 loss data': 'myfolder/loss2.parquet'}
    {'2028 loss data': 'myfolder/loss1.parquet', '2029 loss data': 'myfolder/loss2.parquet'}
    >>> {'2028 loss data': 'fileid://whateverstuff#1', '2029 loss data': 'fileid://whateverstuff#2'}
    {'2028 loss data': 'fileid://whateverstuff#1', '2029 loss data': 'fileid://whateverstuff#2'}

    ```

    Pandas DataFrames, pyarrow datasets, e.g.
    """


class Metrics(BaseModel, extra=Extra.allow):
    """DataModel representing measurement.

    A Thing can have a metrics component.
    Metrics for any given process take the form of a key-value
    mapping of atomic elements.

    If a single metric is represented as a fileid that metric is
    assumed to have a compatible function which is intended to be
    automatically run to measure the output of this fileid.

    If a single metric is represented as a float or integer value
    that metric is assumed to be direct measurement and is recorded as-is.

    Examples
    --------
    >>> metrics = {
    ...     'silly': 1,
    ...     'example': 'fileid://whateverfile',
    ... }
    """

    pass


class Metadata(BaseModel, extra=Extra.allow):
    """DataModel representing a set of metadata.

    A Thing can have a metadata component.

    Metadata for any given process take the form of a key-value mapping
    of atomic elements or a single atomic element indicating a static
    set of inputs already available.

    If a set of metadata is represented as a fileid that fileid is
    assumed to have compatible metadata structure. The metadata for
    that fileid are used as-is.

    Metadata may not be nested and must be atomic.

    Appropriately designed atomic workflows do not use nested parameters.
    Workflows which reuse separate components highly probably do use
    nested parameters, as each workflow represents a point of potential
    parameter update.

    Parameters are used to modify the behavior of execution and there
    is an assumption that modifying a parameter potentially necessitates
    change flowing from the parameters through all downstream nodes
    which consume the parameters either implicitly or explicitly.

    Examples
    --------
    >>> metadata = {'favorite_color': 'red', 'business_process': 'auto'}
    >>> metadata = 'fileid://whateverfile'
    """

    pass


class Parameters(BaseModel, extra=Extra.allow):
    """DataModel representing Parameters.

    A Thing can have a parameters component.
    Parameters for any given process take the form of a key-value
    mapping of atomic elements or a single atomic element indicating
    a static set of inputs already available.

    If a **set** of parameters is represented as a fileid that fileid is
    assumed to have a compatible parameters structure. The parameters
    for that fileid are used as-is.

    If a **single** parameter is represented as a fileid that parameter
    is assumed to have a compatible subordinate parameter structure.
    The parameters for that file-id are imported and used.

    This allows for potentially deeply nested hierarchical sets of
    workflow parameters.

    Appropriately designed atomic workflows do not use nested parameters.
    Workflows which reuse separate components highly probably do use
    nested parameters, as each workflow represents a point of potential
    parameter update.

    Parameters are used to modify the behavior of execution and there
    is an assumption that modifying a parameter potentially necessitates
    change flowing from the parameters through all downstream nodes
    which consume the parameters either implicitly or explicitly.

    Examples
    --------
    >>> parameters = {
    ...     'silly': 1,
    ...     'example': 'two',
    ...     'configuration_file': 'fileid://whateverfile'
    ... }

    >>> parameters = 'fileid://whateverfile'
    """

    pass


class Artifacts(BaseModel, extra=Extra.allow):
    """DataModel representing artifacts.

    A Thing can have an artifacts component.

    What is an artifact?
    Everything that doesn't fit neatly somewhere else.

    Ball trees, html reports, logs, provisioning artifacts, etc.
    These are all valid files for retention that should be attached
    to inform processes.

    Artifacts may be identified by a string; that string is assumed to
    *either* be an accessible folderpath *or* a file identifer which
    contains a representative set of artifacts to reuse.

    Examples
    --------
    >>> artifacts = 'existing_folder_of_artifacts'
    >>> artifacts = 'fileid://whateverfile'
    """

    pass


class Function(BaseModel, extra=Extra.allow):
    """DataModel representing a reusable function.

    A Thing can have a function component.
    That function component *must* be stored and reused as a Thing.

    It's relatively simple to add a function in the ThingStore

    A function component is defined by a Python file with 'workflow' function defined within it.
    Please see examples: TODO - Link docs

    Examples
    --------
    function = 'fileid://whateverfile'
    """

    pass


class Embedding(BaseModel, extra=Extra.allow):
    """DataModel representing an Embedding.

    A Thing can have an embedded representation of arbitrary dimensionality.

    This is anything that may be represented as a Tensor.

    Examples
    --------
    >>> import numpy as np
    >>> np.array([0, 1, 2])
    array([0, 1, 2])
    >>> np.array([[0, 1, 2], [3, 4, 5]])
    array([[0, 1, 2],
           [3, 4, 5]])
    """

    pass


class Thing(BaseModel, extra=Extra.allow):
    """DataModel representing a Thing.

    A 'thing' can be anything, but *might* have primary components.

    Each of these component pieces can be *atomic* or *complex*.

    An *atomic* element is a string, an integer, or a float.

    A *complex* element is a sequence, or mapping, of atomic values.

    String can be atomic or *ticket* representation of aforesaid stuff.

    Potential Components
    --------------------
    * Metadata
    * Function
    * Parameters
    * Metrics
    * Dataset
    * Artifacts
    * Embedding
    """

    pass
