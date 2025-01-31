"""Data hashing utility."""

import hashlib
import pandas as pd
import boto3
import tempfile
import logging
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Union, Optional


logger = logging.getLogger(__name__)


def dataset_digest(  # noqa: C901
    df: Union[pd.DataFrame, pa.Table, str],
    s3_client: Optional[boto3.client] = None,
    bucket: Optional[str] = None,
    key: Optional[str] = None,
    return_type: str = "hex",
) -> Union[str, bytes, int]:
    """Return md5 digest given Pandas DataFrame, PyArrow Table, or filepath.

    Alternatively, this function accepts a boto3 client, bucket, and prefix (key) that points to a
    parquet document on s3.

    NOTE: the current s3 implementation does not necessarily return the md5 hash of the parquet document.
    The hash that is returned is the ETag of the s3 object, which *might* or *might not* be the md5 hash depending on
    how the object was originally uploaded.
    See more documentation here: https://docs.aws.amazon.com/AmazonS3/latest/API/API_Object.html
    Also here: https://docs.aws.amazon.com/AmazonS3/latest/API/API_PutObject.html#API_PutObject_Example_11

    For a pandas dataframe, this writes the dataset out to disk as a parquet document in a temporary directory.
    Then, a recursive call is made, and the path to the parquet document is passed in.

    For a pyarrow table, this writes the dataset out to disk as a parquet document in a temporary directory.
    Then, a recursive call is made, and the path to the parquet document is passed in.

    For a filepath, this reads the parquet in 8MB chunks, uses hashlib.md5 to digest those hashes into a
    single hash, and returns in the format requested (if possible).

    Treats all bytes as big-endian.

    Parameters
    ----------
    df: Union[pd.DataFrame, pa.Table, str]
        This is the dataset in RAM or a filepath to a parquet to hash
    s3_client: Optional[boto3.client]
        This is a client to the location where the parquet file lives whose hash is being requested
    bucket: Optional[str]
        In the case that an s3_client was provided, this is the name of the bucket where the parquet lives
    key: Optional[str]
        In the case that an s3_client was provided, this is the prefix where the parquet lives
    return_type: str
        This is the return type for the hash.
        Currently implemented types:
        - hex
        - bytes
        - int | integer

    Returns
    -------
    hash: Union[str, bytes, int]
        The dataset hash
    """
    hashobj = None
    # Are we working with a Pandas DataFrame?
    if isinstance(df, pd.DataFrame):
        # Write dataframe out as parquet and make recursive call
        with tempfile.TemporaryDirectory() as tmpdir:
            logger.warning(f"Writing dataframe out to disk: '{tmpdir}/df.parquet'")
            df.to_parquet(f"{tmpdir}/df.parquet", index=False)
            return dataset_digest(df=f"{tmpdir}/df.parquet", return_type=return_type)

    # Are we working with a PyArrow Table?
    elif isinstance(df, pa.Table):
        # Write table out as parquet and make recursive call
        with tempfile.TemporaryDirectory() as tmpdir:
            logger.warning(f"Writing PyArrow table out to disk: {tmpdir}/df.parquet")
            pq.write_table(df, f"{tmpdir}/df.parquet")
            return dataset_digest(df=f"{tmpdir}/df.parquet", return_type=return_type)

    # Are we working with a filepath?
    elif isinstance(df, str):
        # File is local
        if not s3_client:
            with open(df, "rb") as f:
                hashobj = hashlib.md5(usedforsecurity=False)
                while chunk := f.read(8192):
                    hashobj.update(chunk)
        # File is remote
        else:
            if not bucket:
                raise Exception(
                    "An s3 client was provided to dataset_digest(), but no bucket was provided."
                )
            if not key:
                raise Exception(
                    "An s3 client and bucket were provided to dataset_digest(), but no key was provided."
                )
            s3_resp: dict = s3_client.head_object(Bucket=bucket, Key=key)
            s3obj_etag: str = s3_resp["ETag"].strip('"')
            return s3obj_etag

    # Otherwise, the dataset type is unimplemented
    else:
        raise Exception(
            f"Cannot create digest of type {type(df)}.\nMust be of type {pd.DataFrame} or {str}"
        )
    # Do you want the hash in hex (default)?
    if return_type == "hex":
        return hashobj.hexdigest()
    # Do you want the hash in bytes?
    elif return_type == "bytes":
        return hashobj.digest()
    # Do you want the hash as an integer?
    elif return_type == "int" or return_type == "integer":
        hash_bytes = hashobj.digest()
        # Interpret bytes as big-endian
        return int.from_bytes(hash_bytes, "big")
    # Otherwise, the return_type is unimplemented
    else:
        return_types = ["hex", "bytes", "int|integer"]
        raise Exception(
            f"Requested return_type not supported: {return_type}"
            + f"\nThe following return types are supported:{return_types}"
        )
