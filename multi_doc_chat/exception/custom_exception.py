import sys
import traceback
from typing import Optional


class DocumentPortalException(Exception):
    """
    Custom exception class for the Intelligent Document Assistant project.

    Captures:
    - Error message
    - File name
    - Line number
    - Complete traceback
    """

    def __init__(
        self,
        error_message: str,
        error_details: Optional[object] = None,
        error_code: Optional[str] = None,
    ):
        self.error_message = str(error_message)
        self.error_code = error_code

        # Get exception information
        if error_details is None:
            _, exc_value, exc_tb = sys.exc_info()

        elif hasattr(error_details, "exc_info"):
            _, exc_value, exc_tb = error_details.exc_info()

        elif isinstance(error_details, BaseException):
            exc_value = error_details
            exc_tb = error_details.__traceback__

        else:
            _, exc_value, exc_tb = sys.exc_info()

        # Find last traceback frame
        last_tb = exc_tb
        while last_tb and last_tb.tb_next:
            last_tb = last_tb.tb_next

        self.file_name = (
            last_tb.tb_frame.f_code.co_filename
            if last_tb
            else "<unknown>"
        )

        self.line_number = (
            last_tb.tb_lineno
            if last_tb
            else -1
        )

        # Save traceback
        if exc_tb:
            self.traceback = "".join(
                traceback.format_exception(
                    type(exc_value),
                    exc_value,
                    exc_tb,
                )
            )
        else:
            self.traceback = ""

        super().__init__(self.__str__())

    def __str__(self):

        message = (
            f"\nError Code : {self.error_code or 'N/A'}"
            f"\nFile       : {self.file_name}"
            f"\nLine       : {self.line_number}"
            f"\nMessage    : {self.error_message}"
        )

        if self.traceback:
            message += f"\n\nTraceback:\n{self.traceback}"

        return message

    def __repr__(self):
        return (
            f"DocumentPortalException("
            f"error_code={self.error_code!r}, "
            f"file={self.file_name!r}, "
            f"line={self.line_number}, "
            f"message={self.error_message!r})"
        )