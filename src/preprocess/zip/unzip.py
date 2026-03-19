from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

try:
    import zipfile_deflate64 as zipfile
except ImportError as exc:
    raise RuntimeError(
        "zipfile_deflate64 is required. Install it with: uv pip install zipfile-deflate64"
    ) from exc


# =========================
# User configuration block
# =========================
ROOT_DIR = Path("data/korean")
LOG_DIR = Path("data/korean/_unzip_logs")
MAX_PASSES = 3
WORKERS = max(1, (os.cpu_count()//2 or 2) - 1)
DELETE_AFTER_ALL_VERIFIED = True
DRY_RUN = False
VERBOSE = True

# Confirmed in this environment:
# - System `unzip` has no native multiprocessing option.
UNZIP_NATIVE_MP_SUPPORTED = False


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _scan_zip_files(root_dir: Path) -> list[Path]:
    return sorted(
        (p for p in root_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".zip"),
        key=lambda p: str(p),
    )


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _sanitize_member_path(member_name: str) -> Path | None:
    normalized = member_name.replace("\\", "/").lstrip("/")
    if not normalized:
        return None

    safe_parts: list[str] = []
    for part in normalized.split("/"):
        if not part or part == ".":
            continue
        if part == "..":
            return None
        if ":" in part:
            return None
        safe_parts.append(part)

    if not safe_parts:
        return None
    return Path(*safe_parts)


def _is_symlink_entry(info: zipfile.ZipInfo) -> bool:
    return ((info.external_attr >> 16) & 0o170000) == 0o120000


def _plan_archive(
    zf: zipfile.ZipFile,
    zip_path: Path,
) -> dict:
    extract_root = zip_path.parent.resolve()
    infos = zf.infolist()

    unsafe_examples: list[str] = []
    unsafe_count = 0
    collision_examples: list[str] = []
    collision_count = 0
    planned_kind: dict[str, bool] = {}
    ops: list[tuple[zipfile.ZipInfo, Path]] = []

    for info in infos:
        rel = _sanitize_member_path(info.filename)
        if rel is None:
            # Root-like entries ("/", "./") are ignored.
            if info.filename.replace("\\", "/").strip("/.") == "":
                continue
            unsafe_count += 1
            if len(unsafe_examples) < 10:
                unsafe_examples.append(f"unsafe-member:{info.filename}")
            continue

        if _is_symlink_entry(info):
            unsafe_count += 1
            if len(unsafe_examples) < 10:
                unsafe_examples.append(f"symlink-entry:{info.filename}")
            continue

        target = (extract_root / rel).resolve(strict=False)
        try:
            target.relative_to(extract_root)
        except ValueError:
            unsafe_count += 1
            if len(unsafe_examples) < 10:
                unsafe_examples.append(f"path-escape:{info.filename}")
            continue

        key = str(target)
        if key in planned_kind:
            unsafe_count += 1
            if len(unsafe_examples) < 10:
                unsafe_examples.append(f"duplicate-target:{target}")
            continue

        planned_kind[key] = info.is_dir()
        ops.append((info, target))

    if unsafe_count > 0:
        unsafe_examples = _dedupe_keep_order(unsafe_examples)
        return {
            "status": "failed_path",
            "reason": f"unsafe paths ({unsafe_count} entries)",
            "collision_count": 0,
            "collision_examples": unsafe_examples,
            "ops": [],
            "file_count": 0,
        }

    file_targets = {Path(k) for k, is_dir in planned_kind.items() if not is_dir}
    dir_targets = {Path(k) for k, is_dir in planned_kind.items() if is_dir}

    internal_conflict: list[str] = []
    for target in list(file_targets) + list(dir_targets):
        for parent in target.parents:
            if parent == extract_root:
                break
            if parent in file_targets:
                internal_conflict.append(f"parent-file-conflict:{target}")
                break

    if internal_conflict:
        internal_conflict = _dedupe_keep_order(internal_conflict)
        return {
            "status": "failed_path",
            "reason": "internal file/dir conflict in archive entries",
            "collision_count": 0,
            "collision_examples": internal_conflict,
            "ops": [],
            "file_count": 0,
        }

    for info, target in ops:
        if target.exists():
            collision_count += 1
            if len(collision_examples) < 20:
                collision_examples.append(str(target))

        parent = target.parent
        while True:
            if parent == extract_root:
                if parent.exists() and not parent.is_dir():
                    collision_count += 1
                    if len(collision_examples) < 20:
                        collision_examples.append(str(parent))
                break
            if parent.exists() and not parent.is_dir():
                collision_count += 1
                if len(collision_examples) < 20:
                    collision_examples.append(str(parent))
                break
            if parent.parent == parent:
                break
            parent = parent.parent

    if collision_count > 0:
        collision_examples = _dedupe_keep_order(collision_examples)
        return {
            "status": "skipped_collision",
            "reason": "existing files/directories would be overwritten",
            "collision_count": collision_count,
            "collision_examples": collision_examples,
            "ops": [],
            "file_count": 0,
        }

    file_count = sum(0 if info.is_dir() else 1 for info, _ in ops)
    return {
        "status": "ready",
        "reason": "",
        "collision_count": 0,
        "collision_examples": [],
        "ops": ops,
        "file_count": file_count,
    }


def _process_one_zip(zip_path_str: str, pass_index: int) -> dict:
    start = perf_counter()
    zip_path = Path(zip_path_str).resolve()
    result = {
        "zip_path": str(zip_path),
        "status": "failed_extract",
        "reason": "",
        "collision_count": 0,
        "collision_examples": [],
        "extracted_count": 0,
        "duration_sec": 0.0,
        "pass_index": pass_index,
    }

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            bad_member = zf.testzip()
            if bad_member is not None:
                result["status"] = "failed_test"
                result["reason"] = f"corrupt member: {bad_member}"
                return result

            plan = _plan_archive(zf, zip_path)
            if plan["status"] != "ready":
                result["status"] = plan["status"]
                result["reason"] = plan["reason"]
                result["collision_count"] = plan["collision_count"]
                result["collision_examples"] = plan["collision_examples"]
                return result

            if DRY_RUN:
                result["status"] = "extracted"
                result["reason"] = "dry-run only"
                result["extracted_count"] = plan["file_count"]
                return result

            extracted_count = 0
            for info, target in sorted(
                plan["ops"],
                key=lambda item: (0 if item[0].is_dir() else 1, len(item[1].parts), str(item[1])),
            ):
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue

                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info, "r") as src, open(target, "xb") as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
                extracted_count += 1

            result["status"] = "extracted"
            result["reason"] = ""
            result["extracted_count"] = extracted_count
            return result

    except zipfile.BadZipFile as exc:
        result["status"] = "failed_test"
        result["reason"] = f"bad zip: {exc}"
        return result
    except FileExistsError as exc:
        result["status"] = "skipped_collision"
        result["reason"] = f"race collision during extract: {exc}"
        result["collision_count"] = 1
        result["collision_examples"] = [str(exc)]
        return result
    except Exception as exc:  # noqa: BLE001
        result["status"] = "failed_extract"
        result["reason"] = f"{type(exc).__name__}: {exc}"
        return result
    finally:
        result["duration_sec"] = round(perf_counter() - start, 6)


def _run_one_pass(zip_paths: list[Path], pass_index: int) -> list[dict]:
    if not zip_paths:
        return []

    results: list[dict] = []
    if WORKERS <= 1 or len(zip_paths) == 1:
        for zip_path in zip_paths:
            results.append(_process_one_zip(str(zip_path), pass_index))
        return sorted(results, key=lambda r: r["zip_path"])

    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = {
            executor.submit(_process_one_zip, str(zip_path), pass_index): zip_path
            for zip_path in zip_paths
        }
        for future in as_completed(futures):
            zip_path = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:  # noqa: BLE001
                results.append(
                    {
                        "zip_path": str(zip_path.resolve()),
                        "status": "failed_extract",
                        "reason": f"worker exception: {type(exc).__name__}: {exc}",
                        "collision_count": 0,
                        "collision_examples": [],
                        "extracted_count": 0,
                        "duration_sec": 0.0,
                        "pass_index": pass_index,
                    }
                )

    return sorted(results, key=lambda r: r["zip_path"])


def _delete_extracted_archives(results: list[dict]) -> dict:
    extracted_paths = [Path(r["zip_path"]) for r in results if r["status"] == "extracted"]
    failure_count = sum(1 for r in results if r["status"].startswith("failed"))

    delete_summary = {
        "enabled": DELETE_AFTER_ALL_VERIFIED,
        "dry_run": DRY_RUN,
        "candidate_count": len(extracted_paths),
        "deleted_count": 0,
        "kept_count": len(extracted_paths),
        "reason": "",
        "delete_errors": [],
    }

    if not DELETE_AFTER_ALL_VERIFIED:
        delete_summary["reason"] = "disabled by configuration"
        return delete_summary

    if DRY_RUN:
        delete_summary["reason"] = "dry-run enabled"
        return delete_summary

    if failure_count > 0:
        delete_summary["reason"] = "failed_* results exist, skipped deletion"
        return delete_summary

    deleted = 0
    delete_errors: list[str] = []
    for zip_path in extracted_paths:
        try:
            if zip_path.exists():
                zip_path.unlink()
            deleted += 1
        except Exception as exc:  # noqa: BLE001
            delete_errors.append(f"{zip_path}: {type(exc).__name__}: {exc}")

    delete_summary["deleted_count"] = deleted
    delete_summary["kept_count"] = max(0, len(extracted_paths) - deleted)
    delete_summary["delete_errors"] = delete_errors[:20]
    delete_summary["reason"] = "completed" if not delete_errors else "completed with delete errors"
    return delete_summary


def _build_summary(
    started_at: str,
    finished_at: str,
    all_results: list[dict],
    delete_summary: dict,
    pass_count: int,
) -> dict:
    status_counts = Counter(r["status"] for r in all_results)
    total_duration = round(sum(r["duration_sec"] for r in all_results), 6)

    return {
        "started_at": started_at,
        "finished_at": finished_at,
        "root_dir": str(ROOT_DIR.resolve()),
        "log_dir": str(LOG_DIR.resolve()),
        "config": {
            "max_passes": MAX_PASSES,
            "workers": WORKERS,
            "delete_after_all_verified": DELETE_AFTER_ALL_VERIFIED,
            "dry_run": DRY_RUN,
            "verbose": VERBOSE,
            "unzip_native_mp_supported": UNZIP_NATIVE_MP_SUPPORTED,
            "zip_backend": "zipfile_deflate64",
        },
        "passes_executed": pass_count,
        "results_total": len(all_results),
        "status_counts": dict(sorted(status_counts.items())),
        "total_worker_duration_sec": total_duration,
        "delete_summary": delete_summary,
    }


def _write_logs(all_results: list[dict], summary: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    events_path = LOG_DIR / "events.jsonl"
    summary_path = LOG_DIR / "summary.json"

    with events_path.open("w", encoding="utf-8") as fp:
        for record in all_results:
            fp.write(json.dumps(record, ensure_ascii=False))
            fp.write("\n")

    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)
        fp.write("\n")


def main() -> None:
    started_at = _now_iso()

    root_dir = ROOT_DIR.resolve()
    if not root_dir.exists():
        raise FileNotFoundError(f"root directory not found: {root_dir}")
    if not root_dir.is_dir():
        raise NotADirectoryError(f"root path is not a directory: {root_dir}")

    if VERBOSE:
        print(f"[config] root={root_dir}")
        print(f"[config] workers={WORKERS}, max_passes={MAX_PASSES}, dry_run={DRY_RUN}")
        print(f"[config] delete_after_all_verified={DELETE_AFTER_ALL_VERIFIED}")
        print(f"[config] unzip_native_mp_supported={UNZIP_NATIVE_MP_SUPPORTED}")

    all_results: list[dict] = []
    processed_zips: set[str] = set()
    passes_executed = 0

    for pass_index in range(1, MAX_PASSES + 1):
        all_current_zips = _scan_zip_files(root_dir)
        pending = [p for p in all_current_zips if str(p.resolve()) not in processed_zips]
        if not pending:
            break

        passes_executed = pass_index
        if VERBOSE:
            print(f"[pass {pass_index}] pending zip files: {len(pending)}")

        pass_results = _run_one_pass(pending, pass_index)
        all_results.extend(pass_results)
        for zip_path in pending:
            processed_zips.add(str(zip_path.resolve()))

        extracted_count = sum(1 for r in pass_results if r["status"] == "extracted")
        if VERBOSE:
            status_counts = Counter(r["status"] for r in pass_results)
            print(f"[pass {pass_index}] status counts: {dict(sorted(status_counts.items()))}")

        # Without new extraction, no new nested zip can appear in later passes.
        if extracted_count == 0:
            break

    delete_summary = _delete_extracted_archives(all_results)
    finished_at = _now_iso()
    summary = _build_summary(started_at, finished_at, all_results, delete_summary, passes_executed)
    _write_logs(all_results, summary)

    if VERBOSE:
        print("[done] summary:")
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
