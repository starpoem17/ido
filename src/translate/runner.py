from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from tqdm import tqdm

from .engine import (
    FieldTranslationPlan,
    LoadedTranslationEngine,
    TranslationBatchResult,
    load_translation_engine,
    plan_field_requests,
    translate_requests_batch,
)
from .jobs import LOG_EVERY_N_ROWS, ROW_BATCH_SIZE, WRITE_META_EVERY_ROW, TranslationJob, get_enabled_jobs


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", file=sys.stdout, flush=True)


@dataclass
class JobStats:
    translated_fields: dict[str, int] = field(default_factory=dict)
    chunked_fields: dict[str, int] = field(default_factory=dict)
    null_fields: dict[str, int] = field(default_factory=dict)
    empty_fields: dict[str, int] = field(default_factory=dict)
    missing_fields: dict[str, int] = field(default_factory=dict)
    retry_count: int = 0

    @classmethod
    def create(cls, target_fields: tuple[str, ...]) -> "JobStats":
        zeros = {field_name: 0 for field_name in target_fields}
        return cls(
            translated_fields=zeros.copy(),
            chunked_fields=zeros.copy(),
            null_fields=zeros.copy(),
            empty_fields=zeros.copy(),
            missing_fields=zeros.copy(),
        )


@dataclass
class ResumeState:
    completed_rows: int
    retry_count: int
    started_at: str


@dataclass
class PendingRow:
    row_index: int
    translated_row: dict[str, Any]
    field_plans: dict[str, FieldTranslationPlan] = field(default_factory=dict)


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def prepare_resume_state(job: TranslationJob) -> ResumeState:
    output_exists = job.output_path.exists()
    meta_exists = job.meta_path().exists()

    if output_exists != meta_exists:
        raise RuntimeError(f"[{job.job_name}] output/meta existence mismatch")

    if not output_exists:
        return ResumeState(
            completed_rows=0,
            retry_count=0,
            started_at=datetime.now().isoformat(),
        )

    output_line_count = count_lines(job.output_path)
    with job.meta_path().open("r", encoding="utf-8") as handle:
        meta = json.load(handle)

    completed_rows = int(meta["completed_rows"])
    if output_line_count != completed_rows:
        raise RuntimeError(
            f"[{job.job_name}] output lines ({output_line_count}) != meta completed_rows ({completed_rows})"
        )

    return ResumeState(
        completed_rows=completed_rows,
        retry_count=int(meta.get("retry_count", 0)),
        started_at=str(meta.get("started_at", datetime.now().isoformat())),
    )


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as tmp:
        json.dump(payload, tmp, ensure_ascii=False, indent=2)
        tmp.write("\n")
        temp_path = Path(tmp.name)
    temp_path.replace(path)


def write_meta(job: TranslationJob, engine: LoadedTranslationEngine, state: ResumeState, stats: JobStats) -> None:
    payload = {
        "job_name": job.job_name,
        "model_id": engine.model_id,
        "auth_mode": engine.auth_mode,
        "attention_backend": engine.attention_backend,
        "flash_attention_available": engine.flash_attention_available,
        "flash_attention_requested": engine.flash_attention_requested,
        "flash_attention_enabled_for_runtime": engine.flash_attention_enabled_for_runtime,
        "fallback_attention_backend": engine.fallback_attention_backend,
        "input_path": str(job.input_path),
        "output_path": str(job.output_path),
        "target_fields": list(job.target_fields),
        "source_lang_code": job.source_lang_code,
        "target_lang_code": job.target_lang_code,
        "completed_rows": state.completed_rows,
        "skipped_rows": sum(stats.null_fields.values())
        + sum(stats.empty_fields.values())
        + sum(stats.missing_fields.values()),
        "retry_count": state.retry_count + stats.retry_count,
        "started_at": state.started_at,
        "updated_at": datetime.now().isoformat(),
    }
    write_json_atomic(job.meta_path(), payload)


def write_report(
    job: TranslationJob,
    engine: LoadedTranslationEngine,
    state: ResumeState,
    stats: JobStats,
    total_input_rows: int,
    status: str,
) -> None:
    payload = {
        "job_name": job.job_name,
        "model_id": engine.model_id,
        "auth_mode": engine.auth_mode,
        "attention_backend": engine.attention_backend,
        "flash_attention_available": engine.flash_attention_available,
        "flash_attention_requested": engine.flash_attention_requested,
        "flash_attention_enabled_for_runtime": engine.flash_attention_enabled_for_runtime,
        "fallback_attention_backend": engine.fallback_attention_backend,
        "total_input_rows": total_input_rows,
        "total_output_rows": state.completed_rows,
        "source_lang_code": job.source_lang_code,
        "target_lang_code": job.target_lang_code,
        "translated_field_counts": stats.translated_fields,
        "chunked_field_counts": stats.chunked_fields,
        "skipped_null_or_empty_counts": {
            field_name: stats.null_fields[field_name] + stats.empty_fields[field_name]
            for field_name in job.target_fields
        },
        "missing_field_counts": stats.missing_fields,
        "retry_count": state.retry_count + stats.retry_count,
        "status": status,
        "updated_at": datetime.now().isoformat(),
    }
    write_json_atomic(job.report_path(), payload)


def process_row_batch(
    raw_lines: list[str],
    batch_start_index: int,
    job: TranslationJob,
    engine: LoadedTranslationEngine,
    stats: JobStats,
) -> list[str]:
    pending_rows: list[PendingRow] = []
    requests = []

    for offset, raw_line in enumerate(raw_lines):
        row_index = batch_start_index + offset
        row = json.loads(raw_line)
        if not isinstance(row, dict):
            raise ValueError(f"[{job.job_name}] row {row_index} is not a JSON object")

        pending_row = PendingRow(
            row_index=row_index,
            translated_row={key: value for key, value in row.items()},
        )

        for field_name in job.target_fields:
            if field_name not in row:
                stats.missing_fields[field_name] += 1
                continue

            value = row[field_name]
            if value is None:
                stats.null_fields[field_name] += 1
                continue
            if not isinstance(value, str):
                raise TypeError(f"[{job.job_name}] row {row_index} field {field_name} is not a string/null")
            if value == "":
                stats.empty_fields[field_name] += 1
                continue

            field_plan = plan_field_requests(
                engine=engine,
                row_index=row_index,
                field_name=field_name,
                source_text=value,
                source_lang_code=job.source_lang_code,
                target_lang_code=job.target_lang_code,
            )
            pending_row.field_plans[field_name] = field_plan
            requests.extend(field_plan.requests)

        pending_rows.append(pending_row)

    results_by_row_field: dict[tuple[int, str], list[TranslationBatchResult]] = {}
    for batch_result in translate_requests_batch(engine=engine, requests=requests):
        key = (batch_result.request.row_index, batch_result.request.field_name)
        results_by_row_field.setdefault(key, []).append(batch_result)

    output_lines: list[str] = []
    for pending_row in pending_rows:
        for field_name, field_plan in pending_row.field_plans.items():
            key = (pending_row.row_index, field_name)
            if key not in results_by_row_field:
                raise RuntimeError(f"missing translated field result for row {pending_row.row_index} field {field_name}")

            field_results = results_by_row_field[key]
            field_results.sort(key=lambda item: item.request.chunk_index)
            pending_row.translated_row[field_name] = "".join(result.translated_text for result in field_results)
            stats.retry_count += sum(result.retry_count for result in field_results)
            stats.translated_fields[field_name] += 1
            if field_plan.used_chunking:
                stats.chunked_fields[field_name] += field_plan.chunk_count

        output_lines.append(json.dumps(pending_row.translated_row, ensure_ascii=False) + "\n")

    return output_lines


def run_job(job: TranslationJob, engine: LoadedTranslationEngine) -> None:
    job.validate()
    if not job.input_path.exists():
        raise FileNotFoundError(f"[{job.job_name}] missing input file: {job.input_path}")
    if ROW_BATCH_SIZE < 1:
        raise ValueError("ROW_BATCH_SIZE must be >= 1")

    job.output_path.parent.mkdir(parents=True, exist_ok=True)
    total_input_rows = count_lines(job.input_path)
    state = prepare_resume_state(job)
    stats = JobStats.create(job.target_fields)

    log(
        f"[job:{job.job_name}] input={job.input_path} output={job.output_path} "
        f"resume_rows={state.completed_rows}/{total_input_rows} "
        f"row_batch_size={ROW_BATCH_SIZE} "
        f"lang={job.source_lang_code}->{job.target_lang_code} mode=official-chat-template "
        f"auth={engine.auth_mode} attn={engine.attention_backend} "
        f"flash_requested={engine.flash_attention_requested} "
        f"flash_available={engine.flash_attention_available} "
        f"flash_runtime={engine.flash_attention_enabled_for_runtime} "
        f"fallback={engine.fallback_attention_backend}"
    )

    try:
        if state.completed_rows >= total_input_rows:
            write_report(job, engine, state, stats, total_input_rows=total_input_rows, status="completed")
            log(f"[job:{job.job_name}] already complete")
            return

        with (
            job.input_path.open("r", encoding="utf-8") as input_handle,
            job.output_path.open("a", encoding="utf-8") as output_handle,
            tqdm(total=total_input_rows, file=sys.stdout, mininterval=1.0) as progress,
        ):
            progress.update(state.completed_rows)
            raw_batch: list[str] = []
            batch_start_index = state.completed_rows

            def flush_current_batch() -> None:
                nonlocal raw_batch, batch_start_index
                if not raw_batch:
                    return

                translated_lines = process_row_batch(
                    raw_lines=raw_batch,
                    batch_start_index=batch_start_index,
                    job=job,
                    engine=engine,
                    stats=stats,
                )
                for translated_line in translated_lines:
                    output_handle.write(translated_line)
                    output_handle.flush()

                    state.completed_rows += 1
                    if WRITE_META_EVERY_ROW:
                        write_meta(job, engine, state, stats)
                    progress.update(1)

                    if state.completed_rows % LOG_EVERY_N_ROWS == 0:
                        log(
                            f"[job:{job.job_name}] rows={state.completed_rows}/{total_input_rows} "
                            f"retry_count={state.retry_count + stats.retry_count}"
                        )
                raw_batch = []

            for row_index, raw_line in enumerate(input_handle):
                if row_index < state.completed_rows:
                    continue
                if not raw_batch:
                    batch_start_index = row_index
                raw_batch.append(raw_line)
                if len(raw_batch) >= ROW_BATCH_SIZE:
                    flush_current_batch()

            flush_current_batch()
    except Exception:
        write_meta(job, engine, state, stats)
        write_report(job, engine, state, stats, total_input_rows=total_input_rows, status="failed")
        raise

    write_meta(job, engine, state, stats)
    write_report(job, engine, state, stats, total_input_rows=total_input_rows, status="completed")
    log(f"[job:{job.job_name}] completed rows={state.completed_rows}")


def main() -> None:
    enabled_jobs = get_enabled_jobs()
    if not enabled_jobs:
        log("no enabled translation jobs")
        return

    engine = load_translation_engine()
    for job in enabled_jobs:
        run_job(job, engine)


if __name__ == "__main__":
    main()
