from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# =========================
# User Settings (edit here)
# =========================
MODEL_ID = "google/translategemma-4b-it"
DTYPE = "bfloat16"
DEVICE = "cuda"
ATTN_IMPLEMENTATION = "sdpa"
FLASH_BACKEND_NAME = "torch_builtin_sdpa_flash"
FALLBACK_ATTENTION_BACKEND = "sdpa_auto"
PREFER_TORCH_FLASH_ATTN = True
ALLOW_FLASH_FALLBACK = True
REQUIRE_HF_AUTH = True

MAX_INPUT_TOKENS = 2048
MIN_RESERVED_OUTPUT_TOKENS = 256

MAX_OUTPUT_TOKENS_PROBLEM = 768
MAX_OUTPUT_TOKENS_THINKING = 1536
MAX_OUTPUT_TOKENS_SOLUTION = 1024

TEMPERATURE = 0.0
DO_SAMPLE = False
RETRY_LIMIT = 3
ROW_BATCH_SIZE = 4
TARGET_FIELDS_PER_ROW = 3
MAX_REQUESTS_PER_BATCH = ROW_BATCH_SIZE * TARGET_FIELDS_PER_ROW

WRITE_META_EVERY_ROW = True
LOG_EVERY_N_ROWS = 10
META_SUFFIX = ".meta.json"
REPORT_SUFFIX = ".report.json"

ALLOWED_TARGET_FIELDS = ("problem", "thinking", "solution")


@dataclass(frozen=True)
class TranslationJob:
    job_name: str
    input_path: Path
    output_path: Path
    target_fields: tuple[str, ...]
    source_lang_code: str
    target_lang_code: str
    enabled: bool

    def meta_path(self) -> Path:
        return self.output_path.with_name(f"{self.output_path.name}{META_SUFFIX}")

    def report_path(self) -> Path:
        return self.output_path.with_name(f"{self.output_path.name}{REPORT_SUFFIX}")

    def validate(self) -> None:
        if not self.target_fields:
            raise ValueError(f"[{self.job_name}] target_fields must not be empty")
        invalid_fields = tuple(field for field in self.target_fields if field not in ALLOWED_TARGET_FIELDS)
        if invalid_fields:
            raise ValueError(f"[{self.job_name}] unsupported target_fields: {invalid_fields}")
        if self.input_path == self.output_path:
            raise ValueError(f"[{self.job_name}] input_path and output_path must differ")
        if not self.source_lang_code:
            raise ValueError(f"[{self.job_name}] source_lang_code must not be empty")
        if not self.target_lang_code:
            raise ValueError(f"[{self.job_name}] target_lang_code must not be empty")


def build_translation_jobs() -> list[TranslationJob]:
    jobs = [
        TranslationJob(
            job_name="nohurry_opus_reasoning_ko",
            input_path=Path(
                "data/english_raw/nohurry-Opus-4.6-Reasoning-3000x-filtered/"
                "distilled_corpus_400k_with_cot-filtered.jsonl"
            ),
            output_path=Path(
                "data/korean_raw/nohurry-Opus-4.6-Reasoning-3000x-filtered/"
                "distilled_corpus_400k_with_cot-filtered.jsonl"
            ),
            target_fields=("problem", "thinking", "solution"),
            source_lang_code="en",
            target_lang_code="ko",
            enabled=True,
        )
    ]
    for job in jobs:
        job.validate()
    return jobs


def get_enabled_jobs() -> list[TranslationJob]:
    enabled_jobs = [job for job in build_translation_jobs() if job.enabled]
    enabled_jobs.sort(key=lambda item: item.job_name)
    return enabled_jobs
