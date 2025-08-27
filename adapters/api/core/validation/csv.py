import os
from fastapi import UploadFile
from urllib import parse
from unicodedata import normalize
import re

from loggers.log import get_loggers
from domain.exceptions import FileTooLarge, UnsupportedFileType, SecurityError
from config.settings import settings

logger = get_loggers("reas.api.csv.fileupload")


def normalize_filename(filename: str) -> str:
    """Normalize filename to remove path traversal vuln, special string, suspicious whitespace, and homoglyhps file.

    Args:
        filename (str): Actual filename received

    Returns:
        str: Normalized lowercased filename
    """
    filename = parse.unquote(filename)
    filename = normalize("NFKC", filename)
    filename = filename.lower()
    filename = filename.strip()
    filename = os.path.basename(filename)
    filename = re.sub(r"\.+", ".", filename)
    filename = re.sub(r"[^a-z0-9._-]", "_", filename)
    return filename


async def validate_csv_metadata(file: UploadFile):
    """Validate expected csv file metadata, including size, extension, and MIME validation.

    Args:
        file (UploadFile): _description_

    Raises:
        UnsupportedFileType: _description_
        FileTooLarge: _description_

    Returns:
        _type_: _description_
    """
    valid_content_type = ["text/csv", "application/vnd.ms-excel"]
    sus_file_name = [".exe", ".bat", ".cmd", ".php", ".js", ".jsp"]
    ext_tamper_pattern = re.compile(
        r"(" + "|".join([rf"\.{ext}" for ext in sus_file_name]) + r")"
    )
    valid_ext = [".csv"]
    valid_content_type_str = ", ".join([ct for ct in valid_content_type])
    try:
        content_type = getattr(file, "content_type", None) or ""
        filename = getattr(file, "filename", "placeholder.csv") or "placeholder.csv"
        filename = normalize_filename(filename)
        if _match := ext_tamper_pattern.search(filename):
            raise SecurityError(
                possible_attack="extension_tampering", filename=filename
            )
        file_ext = os.path.splitext(filename)[1]
        if content_type not in valid_content_type and file_ext not in valid_ext:
            raise UnsupportedFileType(
                expected_type=f"`{valid_content_type_str}`", received_type=content_type
            )

        file.file.seek(0, 2)  # * move pointer to end of the file
        file_size = file.file.tell()
        file.file.seek(0)  # * reset pointer

        if file_size > settings.max_size_bytes:
            raise FileTooLarge(max_size=settings.max_size_mb)

        return file.file.read()
    except (UnsupportedFileType, FileTooLarge, SecurityError):
        raise
