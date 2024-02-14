"""Test default metadata"""
from thethingstore.thing_store_elements import Metadata


def test_add_default_metadata():
    """Demonstrate what adding metadata looks like.

    The Thing Store has strong feelings about *some* Metadata.
    This illustrates what it looks like and how it works.

    Creating an element of Metadata adds information, as below.
    """
    # In the beginning the data was null and void.
    _metadata = {}
    # And PyDantic said 'let there be information'
    _ = Metadata(**_metadata)
    assert isinstance(_, Metadata)
    assert set(_.dict().keys()) == {
        "FILE_ID",
        "FILE_VERSION",
        "DATASET_DATE",
        "DATASET_VALID",
        "TS_HAS_DATASET",
        "TS_HAS_PARAMETERS",
        "TS_HAS_METADATA",
        "TS_HAS_ARTIFACTS",
        "TS_HAS_METRICS",
        "TS_HAS_FUNCTION",
        "TS_HAS_EMBEDDING",
    }, _.dict().keys()
    # When you log something to the Thing Store it has
    #   this information appended by default.
    # For more information, please read the documentation in the Metadata object.
