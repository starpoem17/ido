from __future__ import annotations

import sys
from pathlib import Path
from urllib.parse import quote

from tqdm import tqdm

from src.collect import namu


# =========================
# User configuration block
# =========================
SOURCE_DIR = Path("data/korean_raw/namu")
ALLOWED_SUFFIXES = {
    ".json",
    ".txt",
}
SORT_DISCOVERED_TITLES = True
PREVIEW_URL_COUNT = 10
VERBOSE = True
LOG_PREFIX = "[re_namu]"


def log(message: str) -> None:
    if VERBOSE:
        print(f"{LOG_PREFIX} {message}")


def stable_sorted_strings(values: set[str]) -> list[str]:
    return sorted(values)


def discover_titles() -> tuple[list[str], int, int]:
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"source dir does not exist: {SOURCE_DIR}")

    discovered: set[str] = set()
    matched_files = 0
    duplicate_titles = 0
    entries = list(SOURCE_DIR.iterdir())

    for path in tqdm(entries, file=sys.stdout):
        if not path.is_file():
            continue
        if path.suffix.lower() not in ALLOWED_SUFFIXES:
            continue
        matched_files += 1
        title = path.stem.strip()
        if not title:
            continue
        if title in discovered:
            duplicate_titles += 1
            continue
        discovered.add(title)

    titles = list(discovered)
    if SORT_DISCOVERED_TITLES:
        titles = stable_sorted_strings(discovered)
    return titles, matched_files, duplicate_titles


def build_url_from_title(title: str) -> str:
    return f"{namu.NAMU_BASE_URL}/w/{quote(title, safe='')}"


def build_urls(titles: list[str]) -> list[str]:
    return [build_url_from_title(title) for title in titles]


def log_preview(urls: list[str]) -> None:
    if not urls:
        return
    preview = urls[:PREVIEW_URL_COUNT]
    log(f"preview_urls={preview}")


def main() -> None:
    titles, matched_files, duplicate_titles = discover_titles()
    if not titles:
        raise ValueError(f"no titles discovered from {SOURCE_DIR}")

    urls = build_urls(titles)
    log(
        (
            "discovered matched_files=%d unique_titles=%d duplicate_titles=%d "
            "source_dir=%s"
        )
        % (matched_files, len(titles), duplicate_titles, SOURCE_DIR.resolve())
    )
    log_preview(urls)

    namu.TARGET_URLS = urls
    namu.OUTPUT_DIR = SOURCE_DIR
    log(f"rerun target_count={len(urls)}")
    namu.main()


if __name__ == "__main__":
    main()
