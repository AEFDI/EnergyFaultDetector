
from typing import List, Union
import os
import re
import shutil
import logging
from pathlib import Path
import requests
import zipfile
import tempfile

from urllib.parse import urljoin

logger = logging.getLogger('energy_fault_detector')

API_BASE = "https://zenodo.org/api"
BACKBLAZE_PAGE = "https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data"


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


def recursive_safe_extract(zip_path: Path, dest_dir: Path, remove_archives: bool = True):
    """
    Recursively extracts ZIP files, including those found inside other ZIPs.

    Args:
        zip_path: Path to the .zip archive.
        dest_dir: Directory to extract into.
        remove_archives: Whether to delete the .zip file after successful extraction.
    """
    logger.info(f"Extracting {zip_path.name} to {dest_dir}")
    safe_extract_zip(zip_path, dest_dir)

    if remove_archives:
        try:
            zip_path.unlink()
        except OSError as e:
            logger.warning(f"Could not remove archive {zip_path}: {e}")

    # After extraction, check if any new .zip files were created in the dest_dir
    for item in list(dest_dir.rglob("*.zip")):
        recursive_safe_extract(item, item.parent, remove_archives=remove_archives)



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
                         remove_zip: bool = True, overwrite: bool = False, flatten_file_structure: bool = True,
                         expected_file_types: Union[List[str], str] = "*.csv") -> Path:
    """ Download a Zenodo record via API and unzip any .zip files.

    Downloads all files associated with a given Zenodo record (by ID, DOI, or URL),
    saves them to a local directory, and optionally flattens nested directories
    that result from extracting ZIP archives.

    Args:
        identifier (str): Zenodo record ID, DOI (e.g., 10.5281/zenodo.15846963), or record URL.
            Defaults to the CARE2Compare dataset.
        dest (Path): Local output directory to save downloaded files. (default: downloads)
        remove_zip (bool): If True, ZIP archives will be removed after extraction.
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
            extract_target = out_dir  # Extract directly into dest
            logger.info(f"Unzipping: {p.name} -> {extract_target}")
            try:
                recursive_safe_extract(p, extract_target, remove_archives=remove_zip)
            except Exception as e:
                logger.error(f"Unzipping failed for {p.name}: {e}")

    if flatten_file_structure:
        logger.info(f"Flattening file structure.")
        # Standardize structure: If unzipping created a single subfolder, move its contents up
        # This often happens with Zenodo zips.
        subdirs = [d for d in out_dir.iterdir() if d.is_dir()]
        if len(subdirs) == 1 and not any(next(out_dir.glob(pattern), None) for pattern in expected_file_types):
            redundant_dir = subdirs[0]
            logger.info(f"Flattening directory structure from {redundant_dir}")
            for item in redundant_dir.iterdir():
                shutil.move(str(item), str(out_dir / item.name))
            redundant_dir.rmdir()

    return out_dir


def _build_backblaze_zip_url_map(session: requests.Session) -> dict[str, str]:
    """Build a mapping of Backblaze archive identifiers to their ZIP URLs.

    Scans the Backblaze drive stats page for links of the form 'data_*.zip' and
    maps the identifier between 'data_' and '.zip' to the absolute ZIP URL.

    Args:
        session: A requests.Session used to fetch the Backblaze index page.

    Returns:
        A dictionary mapping archive identifiers (e.g., 'Q1_2020') to absolute ZIP URLs.

    Raises:
        requests.HTTPError: If the Backblaze index page cannot be fetched.
        RuntimeError: If no suitable ZIP links are found.
    """
    logger.info(f"Fetching Backblaze index page: {BACKBLAZE_PAGE}")
    resp = session.get(BACKBLAZE_PAGE, timeout=60)
    resp.raise_for_status()
    html = resp.text

    href_pattern = re.compile(
        r'href=[\'"]([^\'"]*data_[^\'"]*\.zip)[\'"]',
        re.IGNORECASE
    )

    url_map: dict[str, str] = {}
    for match in href_pattern.finditer(html):
        href = match.group(1)
        code_match = re.search(r"data_([^/]+)\.zip", href)
        if not code_match:
            continue
        code = code_match.group(1)  # e.g. "Q1_2020", "2019"
        url_map[code] = urljoin(BACKBLAZE_PAGE, href)

    if not url_map:
        raise RuntimeError("No Backblaze 'data_*.zip' links found on index page.")

    return url_map


def _normalize_backblaze_names(names: Union[List[str], str]) -> list[str]:
    """Normalize Backblaze archive identifiers.

    Removes an optional 'data_' prefix and '.zip' suffix, strips whitespace,
    and removes duplicates while preserving order.

    Args:
        names: Single identifier or list of identifiers, e.g. 'Q1_2020',
            'data_Q1_2020.zip', or ['Q1_2020', 'Q2_2020'].

    Returns:
        A list of normalized identifiers such as ['Q1_2020', 'Q2_2020'].
    """
    if isinstance(names, str):
        raw = [names]
    else:
        raw = list(names)

    result: list[str] = []
    seen: set[str] = set()
    for n in raw:
        s = n.strip()
        if not s:
            continue
        s = re.sub(r"^data_", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\.zip$", "", s, flags=re.IGNORECASE)
        if s and s not in seen:
            seen.add(s)
            result.append(s)
    return result


def _remove_macosx_dirs(root_dir: Path) -> None:
    """Remove macOS metadata directories (_MACOSX / __MACOSX) under a root directory.

    Args:
        root_dir: Directory tree to clean.

    Raises:
        OSError: If removal of a directory fails (logged as a warning).
    """
    for path in root_dir.rglob("*"):
        if path.is_dir() and path.name in ("_MACOSX", "__MACOSX"):
            try:
                shutil.rmtree(path)
            except OSError as e:
                logger.warning(f"Could not remove macOS metadata directory {path}: {e}")


def _move_csv_files_to_dest(source_root: Path, dest_dir: Path, overwrite: bool) -> list[Path]:
    """Move all CSV files under source_root into dest_dir.

    Flattens any directory structure by placing all CSV files directly in dest_dir.

    Args:
        source_root: Directory tree containing extracted files.
        dest_dir: Destination directory where CSV files will be placed.
        overwrite: If True, existing files with the same name in dest_dir are overwritten.
            If False, existing files are left unchanged and skipped.

    Returns:
        A list of Paths to the CSV files in dest_dir that were moved or overwritten.

    Raises:
        OSError: If moving files fails.
    """
    moved: list[Path] = []
    for csv_path in source_root.rglob("*.csv"):
        destination = dest_dir / csv_path.name
        if destination.exists() and not overwrite:
            logger.info(f"File exists and overwrite=False, skipping: {destination.name}")
            continue
        try:
            if destination.exists():
                logger.info(f"Overwriting existing file: {destination.name}")
            shutil.move(str(csv_path), str(destination))
            moved.append(destination)
        except OSError as e:
            logger.error(f"Failed to move CSV {csv_path} -> {destination}: {e}")
    return moved


def download_backblaze_data(names: Union[List[str], str],
                            dest: Path = "./downloads",
                            overwrite: bool = False) -> Path:
    """Download Backblaze drive stats archives and collect CSV files.

    Downloads specified 'data_*.zip' archives from the Backblaze drive stats page,
    extracts them to a temporary location, removes macOS metadata directories,
    and moves all CSV files into a single destination directory.

    The final destination directory contains only the CSV files (no ZIP files or
    _MACOSX directories generated by this function).

    Args:
        names: One or more archive identifiers corresponding to 'data_<name>.zip'
            on the Backblaze page (e.g., "Q1_2020", "Q2_2020", or "data_Q1_2020.zip").
        dest (Path): Destination directory where CSV files will be stored.
        overwrite (bool): If True, existing CSV files in dest with the same name
            will be overwritten. If False, existing files are left unchanged.

    Returns:
        Path: The absolute path to the destination directory containing the CSV files.

    Raises:
        ValueError: If no valid archive identifiers are provided.
        RuntimeError: If no CSV files could be downloaded from the requested archives.
        requests.HTTPError: If fetching the Backblaze index page or a ZIP file fails.
        zipfile.BadZipFile: If an archive is invalid.
        OSError: If filesystem operations fail.
    """
    session = requests.Session()

    normalized_names = _normalize_backblaze_names(names)
    if not normalized_names:
        raise ValueError("No valid Backblaze archive names provided.")

    dest_dir = Path(dest)
    dest_dir.mkdir(parents=True, exist_ok=True)

    zip_url_map = _build_backblaze_zip_url_map(session)

    downloaded_csvs: list[Path] = []
    missing_archives: list[str] = []

    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)

        for code in normalized_names:
            zip_url = zip_url_map.get(code)
            if not zip_url:
                logger.warning(f"Backblaze archive not found on index page: data_{code}.zip")
                missing_archives.append(code)
                continue

            zip_path = tmpdir / f"data_{code}.zip"
            logger.info(f"Downloading Backblaze archive data_{code}.zip from {zip_url}")
            download_file(session, zip_url, zip_path)

            extract_dir = tmpdir / f"extracted_{code}"
            logger.info(f"Extracting Backblaze archive {zip_path.name} to {extract_dir}")
            try:
                safe_extract_zip(zip_path, extract_dir)
            except Exception as e:
                logger.error(f"Extraction failed for {zip_path.name}: {e}")
                continue

            _remove_macosx_dirs(extract_dir)
            csvs = _move_csv_files_to_dest(extract_dir, dest_dir, overwrite=overwrite)
            downloaded_csvs.extend(csvs)

    if missing_archives:
        logger.warning(f"The following Backblaze archives were not found: {', '.join(missing_archives)}")

    if not downloaded_csvs:
        raise RuntimeError("No CSV files were downloaded from the requested Backblaze archives.")

    logger.info(f"Download of Backblaze data successful. CSV files available in {dest_dir.resolve()}")

    return dest_dir.resolve()
