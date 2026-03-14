from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urljoin, urlparse, urlunparse

import requests
from tqdm import tqdm


# =========================
# User configuration block
# =========================
TARGET_URLS = [
    "https://namu.wiki/w/태평양 전쟁",
]
EXCLUDED_TITLE_SUBSTRINGS = [  # 수집하지 않을 목차/하위 문서 제목
    "관련 문서",
    "관련 자료",
    "작품",
    "유명인",
    "매체에서",
    "게임",
    "저서",
    "서비스",
    "관련 항목",
    "추천 도서",
    "참고 문헌",
    "자매결연",
    "국제행사",
    "관련 단체",
    "둘러보기",
    "대중매체",
    "기타",
    "관련 사료 목록",
    "연표",
    "인물",
    "신자",
    "외부 링크",
    "문화재",
    "사료",
    "창작물",
    "왕실",
    "배경으로",
    "소재로",
    "같이 보기",
    "유적",
    "역사서",
    "사진",
    "유물",
    "주요 전선",
    "전쟁 범죄",
    "소재 게임",
    "서적",
    "커뮤니티",
    "대중 매체",
    "각종 매체",
    "참고 자료",
    "시리즈"
]
FOLLOW_TOC_SUBPAGES = True
MAX_TOC_LINK_DEPTH = 1
ALLOW_SUBDOC_PATH_PREFIX_ONLY = True
INCLUDE_SECTION_PATH_HEADERS = True
OUTPUT_DIR = Path("data/korean_raw/namu")
REQUEST_TIMEOUT = 20
REQUEST_HEADERS: dict[str, str] = {}
OVERWRITE = True
PARAGRAPH_SEPARATOR = "\n\n"
SECTION_PATH_SEPARATOR = "."
OUTPUT_JSON_INDENT = 2
OUTPUT_JSON_ENSURE_ASCII = False
ROW_SPLIT = None
ROW_MESSAGES = None
ROW_TOKEN_COUNT = None
VERBOSE = True
LOG_PREFIX = "[namu]"
CONTINUE_ON_SUBPAGE_ERROR = True
NAMU_BASE_URL = "https://namu.wiki"

# Skip rules are intentionally grouped here so future pattern edits only touch
# this block.
SKIP_TEXT_RULES: dict[str, list[str]] = {
    "exact": [
        "[ 펼치기 · 접기 ]",
    ],
    "starts_with": [
        "다른 뜻에 대한 내용은",
        "출처 :",
        "출처:",
        "참고 서적:",
        "편집 보호된",
        "상위 문서:",
        "관련 문서:",
        "이 문서의 내용 중 전체 또는 일부는",
        "주의"
    ],
    "contains": [
        "정리한 문서",
        "자세한 내용은",
        "참고",
        "참조"
    ],
}

SKIP_HTML_TAGS = {
    "details",
    "script",
    "style",
    "summary",
    "table",
}
VOID_HTML_TAGS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}
SKIP_HTML_CLASSES = {
    "wiki-edit-section",
    "wiki-folding",
    "wiki-fn-content",
    "wiki-macro-toc",
    "wiki-table-wrap",
}
TEXT_CAPTURE_CLASSES = {
    "wiki-paragraph",
}


SPACE_RE = re.compile(r"[^\S\n]+")
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
FOOTNOTE_RE = re.compile(r"\[\d+\]")
LINE_TRAILING_HASH_RE = re.compile(r"\s*#+\s*$")
INLINE_HASH_TOKEN_RE = re.compile(r"\s+#\s+")
INVALID_FILENAME_RE = re.compile(r'[<>:"/\\\\|?*\x00-\x1f]')


@dataclass(slots=True)
class ParagraphRecord:
    text: str
    section_title: str | None
    section_number: str | None
    section_path_titles: tuple[str, ...]


@dataclass(slots=True)
class ParseResult:
    title: str
    paragraphs: list[ParagraphRecord]
    raw_paragraph_count: int
    excluded_paragraph_count: int
    excluded_section_titles: list[str]


@dataclass(slots=True)
class DocumentResult:
    url: str
    title: str
    rows: list[dict[str, Any]]
    raw_paragraph_count: int
    excluded_paragraph_count: int
    text_skipped_count: int
    kept_paragraph_count: int
    excluded_section_titles: list[str]


@dataclass(slots=True)
class SectionBlock:
    source: str
    paragraphs: list[str]


@dataclass(slots=True)
class TocLinkRecord:
    url: str
    text: str


@dataclass(slots=True)
class TocExtractionResult:
    total_links: int
    candidate_links: list[TocLinkRecord]
    filtered_out_links: list[TocLinkRecord]


def log(message: str) -> None:
    if VERBOSE:
        print(f"{LOG_PREFIX} {message}")


def validate_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.netloc != "namu.wiki":
        raise ValueError(f"unsupported url: {url}")
    if not parsed.path.startswith("/w/"):
        raise ValueError(f"unsupported namuwiki path: {url}")


def fetch_html(url: str) -> str:
    response = requests.get(url, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.text


def normalize_namu_url(url: str) -> str:
    absolute = urljoin(NAMU_BASE_URL, url)
    parsed = urlparse(absolute)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))


def dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def dedupe_toc_links_keep_order(items: list[TocLinkRecord]) -> list[TocLinkRecord]:
    seen_urls: set[str] = set()
    out: list[TocLinkRecord] = []
    for item in items:
        normalized_url = normalize_namu_url(item.url)
        if normalized_url in seen_urls:
            continue
        seen_urls.add(normalized_url)
        out.append(item)
    return out


def is_subpage_url(main_url: str, candidate_url: str) -> bool:
    main_path = unquote(urlparse(main_url).path.rstrip("/"))
    candidate_path = unquote(urlparse(candidate_url).path.rstrip("/"))
    if candidate_path == main_path:
        return False
    if not ALLOW_SUBDOC_PATH_PREFIX_ONLY:
        return True
    return candidate_path.startswith(f"{main_path}/")


def should_exclude_title(title: str | None) -> bool:
    if title is None:
        return False
    stripped = SPACE_RE.sub(" ", title).strip()
    if not stripped:
        return False
    for pattern in EXCLUDED_TITLE_SUBSTRINGS:
        if pattern and pattern in stripped:
            return True
    return False


class NamuWikiContentParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self._in_title_tag = False
        self._heading_tag: str | None = None
        self._pending_section_title: str | None = None
        self._pending_section_number: str | None = None
        self._current_section_title: str | None = None
        self._current_section_number: str | None = None
        self._skip_depth = 0
        self._capture_depth = 0
        self._current_parts: list[str] = []
        self.paragraphs: list[ParagraphRecord] = []
        self.raw_paragraph_count = 0
        self.excluded_paragraph_count = 0
        self.excluded_section_titles: list[str] = []
        self._excluded_section_numbers: list[str] = []
        self._section_titles_by_number: dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = {key: value or "" for key, value in attrs}
        classes = set(attr_map.get("class", "").split())

        if tag == "meta" and attr_map.get("property") == "og:title":
            content = attr_map.get("content", "").strip()
            if content:
                self.title = content

        if self._skip_depth > 0:
            if tag not in VOID_HTML_TAGS:
                self._skip_depth += 1
            return

        if tag == "title":
            self._in_title_tag = True

        if tag in {"h2", "h3", "h4", "h5", "h6"} and "wiki-heading" in classes:
            self._heading_tag = tag
            self._pending_section_title = None
            self._pending_section_number = None
            return

        if self._heading_tag is not None:
            if tag == "a":
                anchor_id = attr_map.get("id", "").strip()
                if anchor_id.startswith("s-"):
                    self._pending_section_number = anchor_id[2:]
            elif tag == "span":
                section_title = attr_map.get("id", "").strip()
                if section_title:
                    self._pending_section_title = section_title
            return

        if tag in SKIP_HTML_TAGS or classes.intersection(SKIP_HTML_CLASSES):
            if tag not in VOID_HTML_TAGS:
                self._skip_depth = 1
            return

        if tag == "div" and classes.intersection(TEXT_CAPTURE_CLASSES):
            self._capture_depth += 1
            if self._capture_depth == 1:
                self._current_parts = []
            return

        if tag == "br" and self._capture_depth > 0:
            self._current_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if self._skip_depth > 0:
            self._skip_depth -= 1
            return

        if tag == "title":
            self._in_title_tag = False
            return

        if self._heading_tag is not None:
            if tag == self._heading_tag:
                self._finalize_heading()
            return

        if tag == "div" and self._capture_depth > 0:
            self._capture_depth -= 1
            if self._capture_depth == 0:
                paragraph = "".join(self._current_parts).strip()
                if paragraph:
                    self.raw_paragraph_count += 1
                    if self._is_current_section_excluded():
                        self.excluded_paragraph_count += 1
                    else:
                        self.paragraphs.append(
                            ParagraphRecord(
                                text=paragraph,
                                section_title=self._current_section_title,
                                section_number=self._current_section_number,
                                section_path_titles=self._build_section_path_titles(
                                    self._current_section_number,
                                    self._current_section_title,
                                ),
                            )
                        )
                self._current_parts = []

    def handle_data(self, data: str) -> None:
        if self._in_title_tag and not self.title:
            title = data.replace("- 나무위키", "").strip()
            if title:
                self.title = title
        if self._skip_depth == 0 and self._capture_depth > 0:
            self._current_parts.append(data)

    def handle_startendtag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        self.handle_starttag(tag, attrs)

    def _finalize_heading(self) -> None:
        self._heading_tag = None
        self._current_section_title = self._pending_section_title
        self._current_section_number = self._pending_section_number
        if self._current_section_number and self._current_section_title:
            self._section_titles_by_number[self._current_section_number] = (
                self._current_section_title
            )

        if (
            should_exclude_title(self._current_section_title)
            and self._current_section_number
        ):
            if self._current_section_title not in self.excluded_section_titles:
                self.excluded_section_titles.append(self._current_section_title)
            if self._current_section_number not in self._excluded_section_numbers:
                self._excluded_section_numbers.append(self._current_section_number)

        self._pending_section_title = None
        self._pending_section_number = None

    def _is_current_section_excluded(self) -> bool:
        if not self._current_section_number:
            return False

        for excluded_prefix in self._excluded_section_numbers:
            if self._current_section_number == excluded_prefix:
                return True
            if self._current_section_number.startswith(f"{excluded_prefix}."):
                return True
        return False

    def _build_section_path_titles(
        self, section_number: str | None, section_title: str | None
    ) -> tuple[str, ...]:
        if not section_number:
            if section_title:
                return (section_title,)
            return ()

        path_titles: list[str] = []
        parts = section_number.split(".")
        for index in range(1, len(parts) + 1):
            key = ".".join(parts[:index])
            title = self._section_titles_by_number.get(key)
            if title:
                path_titles.append(title)

        if not path_titles and section_title:
            path_titles.append(section_title)
        return tuple(path_titles)


class NamuWikiTocParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._toc_depth = 0
        self._active_href: str | None = None
        self._active_text_parts: list[str] = []
        self.links: list[TocLinkRecord] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = {key: value or "" for key, value in attrs}
        classes = set(attr_map.get("class", "").split())

        if self._toc_depth > 0:
            if tag == "a":
                href = attr_map.get("href", "").strip()
                if href and not href.startswith("#") and href.startswith("/w/"):
                    self._active_href = normalize_namu_url(href)
                    self._active_text_parts = []
            if tag not in VOID_HTML_TAGS:
                self._toc_depth += 1
            return

        if tag == "div" and "wiki-macro-toc" in classes:
            self._toc_depth = 1

    def handle_endtag(self, tag: str) -> None:
        if self._active_href is not None and tag == "a":
            text = SPACE_RE.sub(" ", "".join(self._active_text_parts)).strip()
            if text:
                self.links.append(TocLinkRecord(url=self._active_href, text=text))
            self._active_href = None
            self._active_text_parts = []
        if self._toc_depth > 0:
            self._toc_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._active_href is not None:
            self._active_text_parts.append(data)


def parse_document(html: str) -> ParseResult:
    parser = NamuWikiContentParser()
    parser.feed(html)
    parser.close()

    title = parser.title.replace("- 나무위키", "").strip()
    if not title:
        raise ValueError("failed to extract page title")
    if not parser.paragraphs:
        raise ValueError("failed to extract raw paragraphs")

    return ParseResult(
        title=title,
        paragraphs=parser.paragraphs,
        raw_paragraph_count=parser.raw_paragraph_count,
        excluded_paragraph_count=parser.excluded_paragraph_count,
        excluded_section_titles=parser.excluded_section_titles,
    )


def extract_toc_subpage_urls(html: str, main_url: str) -> TocExtractionResult:
    parser = NamuWikiTocParser()
    parser.feed(html)
    parser.close()

    subpage_links = [
        link
        for link in parser.links
        if is_subpage_url(main_url=main_url, candidate_url=link.url)
    ]
    deduped_links = dedupe_toc_links_keep_order(subpage_links)
    filtered_out_links = [
        link for link in deduped_links if should_exclude_title(link.text)
    ]
    candidate_links = [
        link for link in deduped_links if not should_exclude_title(link.text)
    ]
    return TocExtractionResult(
        total_links=len(parser.links),
        candidate_links=candidate_links,
        filtered_out_links=filtered_out_links,
    )


def normalize_paragraph(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = FOOTNOTE_RE.sub("", text)
    text = text.replace("[편집]", " ")
    text = text.replace("&#91;편집&#93;", " ")
    text = INLINE_HASH_TOKEN_RE.sub(" ", text)
    normalized_lines: list[str] = []
    for raw_line in text.splitlines():
        line = SPACE_RE.sub(" ", raw_line).strip()
        line = LINE_TRAILING_HASH_RE.sub("", line).strip()
        if line:
            normalized_lines.append(line)

    normalized = "\n".join(normalized_lines)
    normalized = MULTI_NEWLINE_RE.sub("\n\n", normalized).strip()
    return normalized


def should_skip_paragraph(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True

    for pattern in SKIP_TEXT_RULES.get("exact", []):
        if stripped == pattern:
            return True

    for pattern in SKIP_TEXT_RULES.get("starts_with", []):
        if stripped.startswith(pattern):
            return True

    for pattern in SKIP_TEXT_RULES.get("contains", []):
        if pattern in stripped:
            return True

    return False


def clean_paragraphs(
    paragraphs: list[ParagraphRecord],
) -> tuple[list[ParagraphRecord], int]:
    cleaned: list[ParagraphRecord] = []
    skipped = 0

    for paragraph in paragraphs:
        normalized = normalize_paragraph(paragraph.text)
        if should_skip_paragraph(normalized):
            skipped += 1
            continue
        cleaned.append(
            ParagraphRecord(
                text=normalized,
                section_title=paragraph.section_title,
                section_number=paragraph.section_number,
                section_path_titles=paragraph.section_path_titles,
            )
        )

    return cleaned, skipped


def build_section_source(title: str, section_path_titles: tuple[str, ...]) -> str:
    if not INCLUDE_SECTION_PATH_HEADERS:
        return title
    if not section_path_titles:
        return title
    return SECTION_PATH_SEPARATOR.join((title, *section_path_titles))


def group_paragraphs_by_section(
    title: str, paragraphs: list[ParagraphRecord]
) -> list[SectionBlock]:
    blocks: list[SectionBlock] = []

    for paragraph in paragraphs:
        source = build_section_source(title, paragraph.section_path_titles)
        if not blocks or blocks[-1].source != source:
            blocks.append(SectionBlock(source=source, paragraphs=[paragraph.text]))
            continue
        blocks[-1].paragraphs.append(paragraph.text)

    return blocks


def build_json_rows_from_blocks(blocks: list[SectionBlock]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for block in blocks:
        body = PARAGRAPH_SEPARATOR.join(block.paragraphs).strip()
        if not body:
            continue
        rows.append(
            {
                "source": block.source,
                "split": ROW_SPLIT,
                "content": body,
                "messages": ROW_MESSAGES,
                "token_count": ROW_TOKEN_COUNT,
            }
        )
    return rows


def build_document_rows(title: str, paragraphs: list[ParagraphRecord]) -> list[dict[str, Any]]:
    blocks = group_paragraphs_by_section(title, paragraphs)
    rows = build_json_rows_from_blocks(blocks)
    if not rows:
        raise ValueError("no body text remained after filtering")
    return rows


def build_combined_rows(documents: list[DocumentResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for document in documents:
        rows.extend(document.rows)
    if not rows:
        raise ValueError("no documents remained after filtering")
    return rows


def sanitize_filename(name: str) -> str:
    sanitized = INVALID_FILENAME_RE.sub("_", name).strip(" .")
    return sanitized or "namu_page"


def make_output_path(title: str) -> Path:
    return OUTPUT_DIR / f"{sanitize_filename(title)}.json"


def save_json(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists() and not OVERWRITE:
        log(f"skip existing file: {path}")
        return
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            rows,
            handle,
            ensure_ascii=OUTPUT_JSON_ENSURE_ASCII,
            indent=OUTPUT_JSON_INDENT,
        )


def collect_document(url: str, html: str | None = None) -> tuple[DocumentResult, str]:
    validate_url(url)
    if html is None:
        html = fetch_html(url)
    parse_result = parse_document(html)
    cleaned_paragraphs, skipped_count = clean_paragraphs(parse_result.paragraphs)
    return (
        DocumentResult(
            url=url,
            title=parse_result.title,
            rows=build_document_rows(parse_result.title, cleaned_paragraphs),
            raw_paragraph_count=parse_result.raw_paragraph_count,
            excluded_paragraph_count=parse_result.excluded_paragraph_count,
            text_skipped_count=skipped_count,
            kept_paragraph_count=len(cleaned_paragraphs),
            excluded_section_titles=parse_result.excluded_section_titles,
        ),
        html,
    )


def log_document_result(document: DocumentResult) -> None:
    log(
        (
            "document url=%s title=%s raw_paragraphs=%d section_excluded=%d "
            "text_skipped=%d kept=%d excluded_sections=%s"
        )
        % (
            document.url,
            document.title,
            document.raw_paragraph_count,
            document.excluded_paragraph_count,
            document.text_skipped_count,
            document.kept_paragraph_count,
            document.excluded_section_titles,
        )
    )


def process_url(url: str) -> Path:
    log(f"fetch start: {url}")

    main_document, main_html = collect_document(url)
    log_document_result(main_document)

    documents = [main_document]
    visited_urls = {normalize_namu_url(url)}
    failed_subpages: list[str] = []
    followed_subpages = 0
    toc_links_found = 0
    deduped_links = 0
    toc_filtered_out_count = 0
    toc_filtered_out_texts: list[str] = []

    if FOLLOW_TOC_SUBPAGES and MAX_TOC_LINK_DEPTH >= 1:
        toc_result = extract_toc_subpage_urls(main_html, main_url=url)
        toc_links_found = toc_result.total_links
        deduped_links = (
            toc_result.total_links
            - len(toc_result.candidate_links)
            - len(toc_result.filtered_out_links)
        )
        toc_filtered_out_count = len(toc_result.filtered_out_links)
        toc_filtered_out_texts = [link.text for link in toc_result.filtered_out_links]

        for toc_link in toc_result.candidate_links:
            subpage_url = toc_link.url
            normalized_subpage_url = normalize_namu_url(subpage_url)
            if normalized_subpage_url in visited_urls:
                continue
            visited_urls.add(normalized_subpage_url)
            try:
                subpage_document, _ = collect_document(subpage_url)
                documents.append(subpage_document)
                followed_subpages += 1
                log_document_result(subpage_document)
            except Exception as exc:  # noqa: BLE001
                failed_subpages.append(subpage_url)
                log(f"failed subpage={subpage_url} error={type(exc).__name__}: {exc}")
                if not CONTINUE_ON_SUBPAGE_ERROR:
                    raise

    output_path = make_output_path(main_document.title)
    output_rows = build_combined_rows(documents)
    save_json(output_path, output_rows)

    log(
        (
            "saved title=%s combined_documents=%d output_rows=%d toc_links_found=%d "
            "toc_subpages_followed=%d deduped_links=%d toc_filtered_out=%d "
            "failed_subpages=%d path=%s"
        )
        % (
            main_document.title,
            len(documents),
            len(output_rows),
            toc_links_found,
            followed_subpages,
            deduped_links,
            toc_filtered_out_count,
            len(failed_subpages),
            output_path,
        )
    )
    if toc_filtered_out_texts:
        log(f"toc_filtered_out_texts={toc_filtered_out_texts}")
    if failed_subpages:
        log(f"failed_subpage_urls={failed_subpages}")
    return output_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    failed_urls: list[str] = []
    for url in tqdm(TARGET_URLS, file=sys.stdout):
        try:
            saved_paths.append(process_url(url))
        except Exception as exc:  # noqa: BLE001
            failed_urls.append(url)
            log(f"failed url={url} error={type(exc).__name__}: {exc}")

    log(
        "done files=%d failed=%d output_dir=%s"
        % (len(saved_paths), len(failed_urls), OUTPUT_DIR.resolve())
    )
    if failed_urls:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
