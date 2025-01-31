"""Reusable Error for the Thing Store Framework."""

from typing import Any, Optional, Type


class FileIDError(Exception):
    """Raise malformed FileID Error.

    Reusable Error for the Thing Store Framework.
    """

    def __init__(self, fileid: str, message: str = ""):
        self.message = f"FileID Cannot be interpreted: [{fileid}]\n"
        super().__init__(self.message + message)


class ThingStoreLoadingError(Exception):
    """Raise Loading Error.

    Reusable Error for the Thing Store Framework.
    """

    def __init__(self, message: str):
        message_header = "Thing Store Loading Error:\n"
        self.message = message_header + message
        super().__init__(self.message)


class ThingStoreFileNotFoundError(FileNotFoundError):
    """Raise FileNotFound Error.

    This dataset does not exist within the Thing Store.
    """

    def __init__(self, file_identifier: str, version: Optional[str] = "LATEST"):
        message_header = "ThingStore FileNotFound:\n"
        self.message = (
            message_header
            + f"""
        Your file identifier ({file_identifier}) at version ({version}) could not be
        found in the Thing Store.
        """
        )
        super().__init__(self.message)


class ThingStoreNotAllowedError(Exception):
    """Raise NotAllowed Error.

    This operation is not allowed in the Thing Store.
    """

    def __init__(self, operation: str, additional_message: str = ""):
        message_header = "ThingStore NotAllowed:\n"
        self.message = (
            message_header
            + f"""
        Your desired operation ({operation}) could not be performed
        in the Thing Store because it is not allowed.
        """
        )
        self.message += additional_message
        super().__init__(self.message)


class ThingStoreGeneralError(Exception):
    """Raise General Error.

    It broke...
    """

    def __init__(self, additional_message: Optional[str] = None):
        message_header = "ThingStore Error:\n"
        self.message = message_header
        if additional_message is not None:
            self.message += additional_message
        super().__init__(self.message)


class ThingStoreTypeError(Exception):
    """Raise Type Error.

    The ThingStore cannot represent this type.
    """

    def __init__(self, bad_type: Type[Any]):
        message_header = "ThingStore Type Error:\n"
        message_header = f"ThingStore Cannot Represent Type: {bad_type}"
        self.message = message_header
        super().__init__(self.message)


class ThingStorePointerError(Exception):
    """ThingPointer had an unresolvable exception."""

    def __init__(
        self, file_id: str, component: str, additional_message: Optional[str] = None
    ):
        message_header = "ThingStore Pointer Error:\n"
        self.message = message_header
        self.message += f"""
        The pointer ({file_id}) for component ({component}) has encountered an exception
        """
        if additional_message is not None:
            self.message += additional_message
        super().__init__(self.message)
