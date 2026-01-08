
from typing import List, Union
import os
import re
import shutil
import logging
from pathlib import Path
import requests
import zipfile

logger = logging.getLogger('energy_fault_detector')

API_BASE = "https://zenodo.org/api"


def parse_record_id(identifier: str) -> str:
    """Extract a Zenodo record ID from an ID, DOI, or URL.

    Accepts:
      - Numeric ID (e.g., "15846963")
      - DOI (e.g., "10.5281/zenodo.15846963")
      - Record URL (e.g., "https://zenodo.org/records/15846963")

    Args:
        identifier: Input string containing an ID, DOI, or URL.

    Returns:
        The numeric record ID as a string.

    Raises:
        ValueError: If a record ID cannot be parsed from the input.
    """
    identifier = identifier.strip()
    m = re.search(r'/records?/(\d+)', identifier)
    if m:
        return m.group(1)
    m = re.search(r'zenodo\.(\d+)$', identifier)  # DOI like 10.5281/zenodo.12345
    if m:
        return m.group(1)
    if identifier.isdigit():
        return identifier
    raise ValueError(f"Could not parse a record ID from: {identifier}")


def fetch_record(session: requests.Session, record_id: str) -> dict:
    """Fetch record metadata from Zenodo's REST API.

    Args:
        session: A requests.Session (may include auth header for restricted files).
        record_id: Numeric Zenodo record ID.

    Returns:
        Parsed JSON payload of the record.

    Raises:
        requests.HTTPError: If the HTTP request fails.
    """
    url = f"{API_BASE}/records/{record_id}"
    r = session.get(url, timeout=60)
    r.raise_for_status()
    return r.json()


def list_files(session: requests.Session, record_json: dict) -> list[dict]:
    """Return a list of file descriptors for a Zenodo record.

    Supports both embedded 'files' in the record JSON and the 'links.files'
    endpoint (newer API).

    Args:
        session: A requests.Session to use for any follow-up call.
        record_json: Record JSON as returned by fetch_record().

    Returns:
        A list of file dicts with at least 'links' and 'key'/'filename'.

    Raises:
        RuntimeError: If no files are found.
        requests.HTTPError: If loading the files listing endpoint fails.
    """
    files = []
    f = record_json.get("files")
    if isinstance(f, dict) and "entries" in f:
        files = f["entries"]
    elif isinstance(f, list):
        files = f
    else:
        files_link = record_json.get("links", {}).get("files")
        if files_link:
            r = session.get(files_link, timeout=60)
            r.raise_for_status()
            data = r.json()
            files = data.get("entries", []) or data.get("files", [])
    if not files:
        raise RuntimeError("No files found in this record.")
    return files


def download_file(session: requests.Session, url: str, dest: Path):
    """Download a single file to disk using streaming.

    Args:
        session: A requests.Session to perform the download.
        url: Direct file URL (e.g., links.content/download/self).
        dest: Destination path to write the file to.

    Raises:
        requests.HTTPError: If the download fails.
        OSError: If writing to disk fails.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    with session.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def safe_extract_zip(zip_path: Path, dest_dir: Path):
    """Extract a ZIP archive safely, preventing path traversal (zip-slip).

    Validates that each member will extract under dest_dir before extraction.

    Args:
        zip_path: Path to the .zip archive.
        dest_dir: Directory to extract into (created if missing).

    Raises:
        RuntimeError: If an unsafe member path is detected.
        zipfile.BadZipFile: If the archive is invalid or corrupted.
        OSError: If filesystem operations fail.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        base = dest_dir.resolve()
        for member in zf.infolist():
            extracted = (dest_dir / member.filename).resolve()
            if base != extracted and not str(extracted).startswith(str(base) + os.sep):
                raise RuntimeError(f"Unsafe path in zip: {member.filename}")
        zf.extractall(dest_dir)


def prepare_output_dir(out_dir: Path, overwrite: bool) -> None:
    """Ensure the output directory is ready.

    If the directory exists and overwrite is True, its contents are removed and the
    directory is recreated empty. If it exists and overwrite is False, it is left as is.
    If it does not exist, it is created.

    Args:
        out_dir: Target output directory path.
        overwrite: Whether to clear and recreate the directory if it exists.

    Raises:
        OSError: If filesystem operations fail.
        RuntimeError: If out_dir points to an unsafe path to remove.
    """
    if out_dir.exists():
        if not overwrite:
            return
        # Safety check for extremely dangerous targets
        resolved = out_dir.resolve()
        if resolved == Path("/") or str(resolved) == "":
            raise RuntimeError(f"Refusing to remove dangerous path: {resolved}")
        if out_dir.is_dir() and not out_dir.is_symlink():
            shutil.rmtree(out_dir)
        else:
            out_dir.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)


def download_zenodo_data(identifier: str = "10.5281/zenodo.15846963", dest: Path = "./downloads",
                         overwrite: bool = False, flatten_file_structure: bool = True,
                         expected_file_types: Union[List[str], str] = "*.csv") -> Path:
    """ Download a Zenodo record via API and unzip any .zip files.

    Downloads all files associated with a given Zenodo record (by ID, DOI, or URL),
    saves them to a local directory, and optionally flattens nested directories
    that result from extracting ZIP archives.

    Args:
        identifier (str): Zenodo record ID, DOI (e.g., 10.5281/zenodo.15846963), or record URL.
            Defaults to the CARE2Compare dataset.
        dest (Path): Local output directory to save downloaded files. (default: downloads)
        overwrite (bool): If True and dest already exists, contents of dest will be overwritten.
            Default is False.
        flatten_file_structure (bool): If True and unzipping results in a single top-level folder
            with no conflicting root-level files matching `expected_file_types`,
            moves its contents up one level. Default is True.
        expected_file_types (Union[List[str], str]): Glob pattern(s) used to detect existing relevant files
            at the root. If any match, flattening is skipped.
            Can be a string like '*.csv' or list like ['*.csv', '*.json']. Default is '*.csv'.

    Returns:
        Path: The absolute path to the directory containing the downloaded and unzipped data.
    """
    if isinstance(expected_file_types, str):
        expected_file_types = [expected_file_types]

    session = requests.Session()
    try:
        record_id = parse_record_id(identifier)
    except ValueError as e:
        logger.error(e)
        raise

    out_dir = Path(dest)

    try:
        if out_dir.exists() and overwrite:
            logger.info(f"Clearing output directory: {out_dir}")
        prepare_output_dir(out_dir, overwrite)
    except Exception as e:
        logger.error(f"Failed to prepare output directory: {e}")
        raise

    logger.info(f"Fetching record {record_id} metadata...")
    record = fetch_record(session, record_id)

    try:
        files = list_files(session, record)
    except RuntimeError as e:
        logger.error(e)
        raise

    downloaded = []
    for f in files:
        links = f.get("links", {})
        url = links.get("content") or links.get("download") or links.get("self")
        if not url:
            logger.error(f"Skipping file without download link: {f}")
            continue
        name = f.get("key") or f.get("filename") or Path(url.split("?", 1)[0]).name
        dest = out_dir / name
        logger.info(f"Downloading: {name}")
        download_file(session, url, dest)
        downloaded.append(dest)

    logger.info(f"Download of data from {identifier} successful.")
    # Unzip any downloaded .zip files
    for p in downloaded:
        if p.suffix.lower() == ".zip":
            extract_target = out_dir # Extract directly into dest
            logger.info(f"Unzipping: {p.name} -> {extract_target}")
            try:
                safe_extract_zip(p, extract_target)
            except Exception as e:
                logger.error(f"Unzipping failed for {p.name}: {e}")
            else:
                try:
                    p.unlink()
                    logger.info(f"Removed archive: {p.name}")
                except OSError as e:
                    logger.warning(f"Could not remove {p}: {e}")

    if flatten_file_structure:
        logger.info(f"Flattening file structure.")
        # Standardize structure: If unzipping created a single subfolder, move its contents up
        # This often happens with Zenodo zips.
        subdirs = [d for d in out_dir.iterdir() if d.is_dir()]
        if len(subdirs) == 1 and not any(out_dir.glob(pattern) for pattern in expected_file_types):
            redundant_dir = subdirs[0]
            logger.info(f"Flattening directory structure from {redundant_dir}")
            for item in redundant_dir.iterdir():
                shutil.move(str(item), str(out_dir / item.name))
            redundant_dir.rmdir()

    return out_dir
