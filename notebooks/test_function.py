"""Example Testing Function.

This has every component.
"""
import pandas as pd
from thethingstore.types import FileId, Thing

workflow_metadata = {"example": "label", "FILE_ID": "EXTREMELY_SPECIFIC"}
workflow_artifacts_dir = "localthingstore"
# workflow_embedding = [1., 2., 3.]  # Not implemented just yet.
# workflow_embedding = [[1., 2., 3.], [4., 5., 6.]]  # Not implemented just yet.
workflow_dataset = pd.DataFrame({"stupid": [1.0, 2.0, 3.0]})
workflow_metrics = {"metric": "SPECIFIC_MEASUREMENT_FUNCTION"}


def workflow(number_fileid: FileId = "asd") -> Thing:
    return {"metadata": {"mew": "labels"}}
