"""Test save functions."""
import os
import pytest
import tempfile
from thethingstore.api.save import save


class Stupid:
    a = 1


def test_save_fringe_items():
    """Tests edge cases for save.

    This is tested pretty thoroughly in test_item_save_and_load, so
    only the edge cases are called out explicitly here.
    """
    # 1. This tests that save won't overwrite an existing directory.
    with tempfile.TemporaryDirectory() as t:
        with pytest.raises(RuntimeError, match="Cannot save on top of an existing dir"):
            save("anything", location=t)
    # 2. This tests that when given a list of non atomic things that
    #   it makes that dir and saves everything underneath it.
    # 3. This is a random object that needs to be saved as a pickle.
    with tempfile.TemporaryDirectory() as t:
        objects_to_save = [["1", "2", "3"], {"a": "thingy"}, Stupid()]
        save(objects_to_save, location=f"{t}/example")
        assert set(os.listdir(f"{t}/ts-list-example")) == {
            "ts-dict-1.parquet",
            "ts-list-0.parquet",
            "ts-thing-2.pickle",
        }
