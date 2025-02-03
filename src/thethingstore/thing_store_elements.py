"""Standard metadata elements.

This exposes standard metadata.
"""

from datetime import datetime
from thethingstore._types import FileId
from pydantic import BaseModel, Extra, Field
from typing import Optional


class Metadata(BaseModel, extra=Extra.allow):
    """DataModel representing a set of individual metadata for a thing."""

    FILE_ID: Optional[FileId] = Field(default=None, description="The unique FileID")
    FILE_VERSION: int = Field(default=1, description="The File Version")
    DATASET_DATE: datetime = Field(
        default=datetime.now(), description="The date and time of creation / update."
    )
    DATASET_VALID: bool = Field(
        default=True, description="Whether this element should be queued for removal."
    )
    TS_HAS_DATASET: bool = Field(
        default=False, description="Whether or not the thing has dataset."
    )
    TS_HAS_PARAMETERS: bool = Field(
        default=False, description="Whether or not the thing has parameters."
    )
    TS_HAS_METADATA: bool = Field(
        default=False, description="Whether or not the thing has metadata."
    )
    TS_HAS_ARTIFACTS: bool = Field(
        default=False, description="Whether or not the thing has artifacts."
    )
    TS_HAS_METRICS: bool = Field(
        default=False, description="Whether or not the thing has metrics."
    )
    TS_HAS_FUNCTION: bool = Field(
        default=False, description="Whether or not the thing has function."
    )
    TS_HAS_EMBEDDING: bool = Field(
        default=False, description="Whether or not the thing has embedding."
    )
