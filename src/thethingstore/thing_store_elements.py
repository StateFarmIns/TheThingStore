"""Standard metadata elements.

This exposes standard metadata.
"""
from datetime import datetime
from thethingstore.types import FileId
from pydantic import BaseModel, Extra, Field


class Metadata(BaseModel, extra=Extra.allow):
    """DataModel representing a set of individual metadata."""

    FILE_ID: FileId = Field(default=None, description="The unique FileID")
    FILE_VERSION: int = Field(default=1, description="The File Version")
    DATASET_DATE: datetime = Field(
        default=datetime.now(), description="The date and time of creation / update."
    )
    DATASET_VALID: bool = Field(
        default=True, description="Whether this element should be queued for removal."
    )
    TS_HAS_DATASET: bool = Field(
        default=False, description="Whether or not the node has dataset."
    )
    TS_HAS_PARAMETERS: bool = Field(
        default=False, description="Whether or not the node has parameters."
    )
    TS_HAS_METADATA: bool = Field(
        default=False, description="Whether or not the node has metadata."
    )
    TS_HAS_ARTIFACTS: bool = Field(
        default=False, description="Whether or not the node has artifacts."
    )
    TS_HAS_METRICS: bool = Field(
        default=False, description="Whether or not the node has metrics."
    )
    TS_HAS_FUNCTION: bool = Field(
        default=False, description="Whether or not the node has function."
    )
    TS_HAS_EMBEDDING: bool = Field(
        default=False, description="Whether or not the node has embedding."
    )
