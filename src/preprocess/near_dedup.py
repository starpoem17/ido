from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

import src.preprocess.common as common
from src.preprocess.common import (
    STAGING_SCHEMA,
    ensure_parent_dir,
    log,
    now_utc_iso,
    progress_bar,
    truncate_sample,
    write_json,
)


# =========================
# User configuration block
# =========================
INPUT_ROOT = Path("data/korean_processed/exact_dedup")
OUTPUT_ROOT = Path("data/korean_processed/near_dedup")
INTERMEDIATE_ROOT = OUTPUT_ROOT / "_intermediate"
TARGET_SHARD_BYTES = 1_073_741_824
MINHASH_NGRAMS = 5
MINHASH_NUM_BUCKETS = 14
MINHASH_HASHES_PER_BUCKET = 8
MINHASH_HASH_PRECISION = 64
JACCARD_THRESHOLD = 0.8
NUM_WORKERS = 7
FETCH_RECORD_BATCH_ROWS = 256
DUCKDB_THREADS = 7
OVERWRITE_OUTPUT = False
ENABLE_TQDM = True
ENABLE_DEBUG_LOG = True
TQDM_MININTERVAL_SEC = 1.0


USAGE_PRIORITY = {"PT": 1, "SFT": 2, "REASONING": 3}


@dataclass(frozen=True)
class SignatureRow:
    record_locator: str
    source: str
    data_usage: str
    split: str
    content: str
    content_length: int
    signature: list[int]

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RowMetadata:
    record_locator: str
    source: str
    data_usage: str
    split: str
    content: str
    content_length: int


@dataclass(frozen=True)
class NearDedupShardResult:
    shard_path: str
    row_count: int
    approx_nbytes: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NearDedupSummary:
    run_id: str
    input_shard_count: int
    input_row_count: int
    signature_row_count: int
    passthrough_row_count: int
    candidate_pair_count: int
    verified_pair_count: int
    near_cluster_count: int
    retained_row_count: int
    deleted_row_count: int
    output_shard_count: int
    output_shards: list[dict[str, Any]]
    source_counts_before: dict[str, int]
    source_counts_after: dict[str, int]
    source_counts_deleted: dict[str, int]
    data_usage_counts_before: dict[str, int]
    data_usage_counts_after: dict[str, int]
    data_usage_counts_deleted: dict[str, int]
    representative_reason_counts: dict[str, int]
    jaccard_threshold: float
    started_at: str
    finished_at: str
    duration_sec: float
    output_root: str
    deleted_log_path: str
    summary_json_path: str
    manifest_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def discover_input_shards(input_root: Path) -> list[Path]:
    return sorted(input_root.glob("part-*.parquet"), key=lambda path: str(path))


def run_near_dedup(
    *,
    input_root: Path,
    output_root: Path,
    intermediate_root: Path,
    target_shard_bytes: int,
    minhash_ngrams: int,
    minhash_num_buckets: int,
    minhash_hashes_per_bucket: int,
    minhash_hash_precision: int,
    jaccard_threshold: float,
    fetch_record_batch_rows: int,
    duckdb_threads: int,
    overwrite_output: bool,
) -> NearDedupSummary:
    if minhash_ngrams < 1:
        raise ValueError("minhash_ngrams must be >= 1")
    if minhash_num_buckets < 1:
        raise ValueError("minhash_num_buckets must be >= 1")
    if minhash_hashes_per_bucket < 1:
        raise ValueError("minhash_hashes_per_bucket must be >= 1")
    if minhash_hash_precision != 64:
        raise ValueError("only 64-bit minhash precision is supported")
    if not 0.0 <= jaccard_threshold <= 1.0:
        raise ValueError("jaccard_threshold must be between 0 and 1")

    input_shards = discover_input_shards(input_root)
    if not input_shards:
        raise FileNotFoundError(f"no input parquet shards found under {input_root}")

    deleted_log_path = output_root / "_logs" / "deleted_rows_minhash.jsonl"
    summary_json_path = output_root / "_meta" / "deleted_rows_minhash_summary.json"
    manifest_path = output_root / "_meta" / "near_dedup_manifest.json"
    started_at = now_utc_iso()
    started_perf = perf_counter()
    run_id = started_at.replace("-", "").replace(":", "").replace(".", "")

    _prepare_output_root(
        output_root=output_root,
        intermediate_root=intermediate_root,
        overwrite_output=overwrite_output,
    )
    log(
        "[near_dedup] start "
        f"input_root={input_root} output_root={output_root} "
        f"input_shards={len(input_shards)} ngrams={minhash_ngrams} "
        f"num_buckets={minhash_num_buckets} hashes_per_bucket={minhash_hashes_per_bucket} "
        f"jaccard_threshold={jaccard_threshold}"
    )

    before_counts = _count_input_rows(input_shards)
    signature_summary = _build_signature_files(
        input_shards=input_shards,
        signatures_root=intermediate_root / "signatures",
        passthrough_root=intermediate_root / "passthrough",
        minhash_ngrams=minhash_ngrams,
        minhash_num_buckets=minhash_num_buckets,
        minhash_hashes_per_bucket=minhash_hashes_per_bucket,
    )
    band_files = _build_bucket_inputs(
        signature_files=signature_summary.signature_files,
        buckets_root=intermediate_root / "bucket_inputs",
        minhash_num_buckets=minhash_num_buckets,
        minhash_hashes_per_bucket=minhash_hashes_per_bucket,
    )
    candidate_pair_count = _build_candidate_pairs(
        band_files=band_files,
        candidates_root=intermediate_root / "candidates",
        duckdb_threads=duckdb_threads,
    )
    verified = _verify_candidate_pairs(
        signature_files=signature_summary.signature_files,
        candidates_root=intermediate_root / "candidates",
        jaccard_threshold=jaccard_threshold,
        fetch_record_batch_rows=fetch_record_batch_rows,
        minhash_ngrams=minhash_ngrams,
    )
    cluster_result = _build_clusters(verified=verified)
    deleted_row_count, reason_counts = _write_deleted_rows_log(
        run_id=run_id,
        deleted_log_path=deleted_log_path,
        deleted_records=cluster_result.deleted_records,
        threshold=jaccard_threshold,
    )
    output_shards = _write_retained_rows(
        input_shards=input_shards,
        passthrough_root=intermediate_root / "passthrough",
        output_root=output_root,
        target_shard_bytes=target_shard_bytes,
        removed_locators=cluster_result.removed_locators,
    )

    retained_row_count = sum(shard.row_count for shard in output_shards)
    source_counts_before = before_counts["source"]
    data_usage_counts_before = before_counts["data_usage"]
    source_counts_deleted = _count_from_deleted(cluster_result.deleted_records, key="source")
    data_usage_counts_deleted = _count_from_deleted(
        cluster_result.deleted_records, key="data_usage"
    )
    source_counts_after = _subtract_counts(source_counts_before, source_counts_deleted)
    data_usage_counts_after = _subtract_counts(
        data_usage_counts_before,
        data_usage_counts_deleted,
    )

    finished_at = now_utc_iso()
    duration_sec = round(perf_counter() - started_perf, 3)
    summary = NearDedupSummary(
        run_id=run_id,
        input_shard_count=len(input_shards),
        input_row_count=before_counts["rows"],
        signature_row_count=signature_summary.signature_row_count,
        passthrough_row_count=signature_summary.passthrough_row_count,
        candidate_pair_count=candidate_pair_count,
        verified_pair_count=verified.verified_pair_count,
        near_cluster_count=cluster_result.cluster_count,
        retained_row_count=retained_row_count,
        deleted_row_count=deleted_row_count,
        output_shard_count=len(output_shards),
        output_shards=[shard.to_dict() for shard in output_shards],
        source_counts_before=source_counts_before,
        source_counts_after=source_counts_after,
        source_counts_deleted=source_counts_deleted,
        data_usage_counts_before=data_usage_counts_before,
        data_usage_counts_after=data_usage_counts_after,
        data_usage_counts_deleted=data_usage_counts_deleted,
        representative_reason_counts=dict(sorted(reason_counts.items())),
        jaccard_threshold=jaccard_threshold,
        started_at=started_at,
        finished_at=finished_at,
        duration_sec=duration_sec,
        output_root=str(output_root),
        deleted_log_path=str(deleted_log_path),
        summary_json_path=str(summary_json_path),
        manifest_path=str(manifest_path),
    )
    _write_summary_files(
        summary=summary,
        summary_json_path=summary_json_path,
        manifest_path=manifest_path,
    )
    log(
        "[near_dedup] completed "
        f"input_rows={summary.input_row_count} candidates={summary.candidate_pair_count} "
        f"verified_pairs={summary.verified_pair_count} clusters={summary.near_cluster_count} "
        f"retained={summary.retained_row_count} deleted={summary.deleted_row_count}"
    )
    return summary


@dataclass(frozen=True)
class SignatureBuildSummary:
    signature_files: list[Path]
    signature_row_count: int
    passthrough_row_count: int


def _prepare_output_root(
    *,
    output_root: Path,
    intermediate_root: Path,
    overwrite_output: bool,
) -> None:
    if output_root.exists():
        existing_paths = list(output_root.iterdir())
        if existing_paths and not overwrite_output:
            raise FileExistsError(
                f"output_root already contains files: {output_root}. "
                "Set OVERWRITE_OUTPUT=True to rebuild."
            )
        if overwrite_output:
            shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "_logs").mkdir(parents=True, exist_ok=True)
    (output_root / "_meta").mkdir(parents=True, exist_ok=True)
    intermediate_root.mkdir(parents=True, exist_ok=True)


def _count_input_rows(input_shards: list[Path]) -> dict[str, Any]:
    source_counts: Counter[str] = Counter()
    usage_counts: Counter[str] = Counter()
    total_rows = 0
    for row in _iter_canonical_rows(input_shards):
        total_rows += 1
        source_counts[row["source"]] += 1
        usage_counts[row["data_usage"]] += 1
    return {
        "rows": total_rows,
        "source": dict(sorted(source_counts.items())),
        "data_usage": dict(sorted(usage_counts.items())),
    }


def _iter_canonical_rows(input_shards: list[Path]) -> Iterator[dict[str, Any]]:
    for shard_path in input_shards:
        parquet_file = pq.ParquetFile(shard_path)
        row_offset = 0
        for batch in parquet_file.iter_batches(batch_size=FETCH_RECORD_BATCH_ROWS):
            rows = batch.to_pylist()
            for index, row in enumerate(rows):
                row["record_locator"] = f"{shard_path}#{row_offset + index}"
                yield row
            row_offset += len(rows)


def _build_signature_files(
    *,
    input_shards: list[Path],
    signatures_root: Path,
    passthrough_root: Path,
    minhash_ngrams: int,
    minhash_num_buckets: int,
    minhash_hashes_per_bucket: int,
) -> SignatureBuildSummary:
    signatures_root.mkdir(parents=True, exist_ok=True)
    passthrough_root.mkdir(parents=True, exist_ok=True)
    signature_files: list[Path] = []
    signature_row_count = 0
    passthrough_row_count = 0
    num_hashes = minhash_num_buckets * minhash_hashes_per_bucket
    progress = progress_bar(total=len(input_shards), desc="near_dedup/signatures")
    try:
        for shard_index, shard_path in enumerate(input_shards, start=1):
            signature_rows: list[dict[str, Any]] = []
            passthrough_rows: list[dict[str, Any]] = []
            parquet_file = pq.ParquetFile(shard_path)
            row_offset = 0
            for batch in parquet_file.iter_batches(batch_size=FETCH_RECORD_BATCH_ROWS):
                rows = batch.to_pylist()
                for index, row in enumerate(rows):
                    record_locator = f"{shard_path}#{row_offset + index}"
                    content = row["content"]
                    if content is None:
                        passthrough_rows.append(
                            {
                                "source": row["source"],
                                "data_usage": row["data_usage"],
                                "split": row["split"],
                                "content": row["content"],
                                "messages": row["messages"],
                                "token_count": row["token_count"],
                                "record_locator": record_locator,
                            }
                        )
                        continue
                    signature = _compute_signature(
                        text=content,
                        minhash_ngrams=minhash_ngrams,
                        num_hashes=num_hashes,
                    )
                    signature_rows.append(
                        SignatureRow(
                            record_locator=record_locator,
                            source=row["source"],
                            data_usage=row["data_usage"],
                            split=row["split"],
                            content=content,
                            content_length=len(content),
                            signature=signature,
                        ).to_record()
                    )
                row_offset += len(rows)

            signature_path = signatures_root / f"part-{shard_index:06d}.parquet"
            signature_schema = pa.schema(
                [
                    pa.field("record_locator", pa.string(), nullable=False),
                    pa.field("source", pa.string(), nullable=False),
                    pa.field("data_usage", pa.string(), nullable=False),
                    pa.field("split", pa.string(), nullable=False),
                    pa.field("content", pa.string(), nullable=False),
                    pa.field("content_length", pa.int32(), nullable=False),
                    pa.field("signature", pa.list_(pa.uint64()), nullable=False),
                ]
            )
            pq.write_table(pa.Table.from_pylist(signature_rows, schema=signature_schema), signature_path)
            signature_files.append(signature_path)
            signature_row_count += len(signature_rows)

            if passthrough_rows:
                passthrough_path = passthrough_root / f"part-{shard_index:06d}.parquet"
                passthrough_schema = STAGING_SCHEMA.append(
                    pa.field("record_locator", pa.string(), nullable=False)
                )
                pq.write_table(
                    pa.Table.from_pylist(passthrough_rows, schema=passthrough_schema),
                    passthrough_path,
                )
            passthrough_row_count += len(passthrough_rows)
            progress.update(1)
    finally:
        progress.close()

    return SignatureBuildSummary(
        signature_files=signature_files,
        signature_row_count=signature_row_count,
        passthrough_row_count=passthrough_row_count,
    )


def _compute_signature(*, text: str, minhash_ngrams: int, num_hashes: int) -> list[int]:
    shingles = _build_shingles(text=text, n=minhash_ngrams)
    signature = [(1 << 64) - 1] * num_hashes
    for shingle in shingles:
        payload = shingle.encode("utf-8")
        for seed in range(num_hashes):
            value = _hash_payload(payload=payload, seed=seed)
            if value < signature[seed]:
                signature[seed] = value
    return signature


def _build_shingles(*, text: str, n: int) -> set[str]:
    tokens = _tokenize_korean_text(text)
    if not tokens:
        return {text}
    if len(tokens) < n:
        return {" ".join(tokens)}
    return {" ".join(tokens[index : index + n]) for index in range(len(tokens) - n + 1)}


def _tokenize_korean_text(text: str) -> list[str]:
    try:
        from datatrove.utils.typeshelper import Languages
        from datatrove.utils.word_tokenizers import load_word_tokenizer

        tokenizer = load_word_tokenizer(Languages.korean)
        return tokenizer.word_tokenize(text)
    except Exception:
        try:
            from kiwipiepy import Kiwi

            kiwi = Kiwi()
            return [token.form for token in kiwi.tokenize(text)]
        except Exception as exc:
            raise RuntimeError(
                "Korean near dedup tokenization requires datatrove or kiwipiepy. "
                "Run `uv sync` after updating dependencies."
            ) from exc


def _hash_payload(*, payload: bytes, seed: int) -> int:
    hasher = hashlib.blake2b(digest_size=8, person=seed.to_bytes(8, "little"))
    hasher.update(payload)
    return int.from_bytes(hasher.digest(), "little", signed=False)


def _build_bucket_inputs(
    *,
    signature_files: list[Path],
    buckets_root: Path,
    minhash_num_buckets: int,
    minhash_hashes_per_bucket: int,
) -> list[Path]:
    buckets_root.mkdir(parents=True, exist_ok=True)
    bucket_files: list[Path] = []
    progress = progress_bar(total=len(signature_files), desc="near_dedup/buckets")
    bucket_schema = pa.schema(
        [
            pa.field("band_idx", pa.int32(), nullable=False),
            pa.field("bucket_key", pa.string(), nullable=False),
            pa.field("record_locator", pa.string(), nullable=False),
        ]
    )
    try:
        for shard_index, signature_path in enumerate(signature_files, start=1):
            records: list[dict[str, Any]] = []
            for row in pq.read_table(signature_path).to_pylist():
                signature = row["signature"]
                for band_idx in range(minhash_num_buckets):
                    start = band_idx * minhash_hashes_per_bucket
                    end = start + minhash_hashes_per_bucket
                    band = signature[start:end]
                    records.append(
                        {
                            "band_idx": band_idx,
                            "bucket_key": hashlib.sha1(
                                repr(tuple(band)).encode("utf-8")
                            ).hexdigest(),
                            "record_locator": row["record_locator"],
                        }
                    )
            bucket_path = buckets_root / f"part-{shard_index:06d}.parquet"
            pq.write_table(pa.Table.from_pylist(records, schema=bucket_schema), bucket_path)
            bucket_files.append(bucket_path)
            progress.update(1)
    finally:
        progress.close()
    return bucket_files


def _build_candidate_pairs(
    *,
    band_files: list[Path],
    candidates_root: Path,
    duckdb_threads: int,
) -> int:
    candidates_root.mkdir(parents=True, exist_ok=True)
    db_path = candidates_root / "candidate_pairs.duckdb"
    if db_path.exists():
        db_path.unlink()
    con = duckdb.connect(str(db_path))
    try:
        con.execute(f"PRAGMA threads={duckdb_threads}")
        parquet_paths = ", ".join(f"'{_sql_quote(str(path))}'" for path in band_files)
        query = f"""
            COPY (
                SELECT DISTINCT
                    LEAST(a.record_locator, b.record_locator) AS left_locator,
                    GREATEST(a.record_locator, b.record_locator) AS right_locator
                FROM read_parquet([{parquet_paths}]) AS a
                JOIN read_parquet([{parquet_paths}]) AS b
                  ON a.band_idx = b.band_idx
                 AND a.bucket_key = b.bucket_key
                 AND a.record_locator < b.record_locator
            )
            TO '{_sql_quote(str(candidates_root / "pairs.parquet"))}'
            (FORMAT PARQUET)
        """
        log("[near_dedup/candidates] building candidate pairs")
        con.execute(query)
        count = int(
            con.execute(
                f"SELECT COUNT(*) FROM read_parquet('{_sql_quote(str(candidates_root / 'pairs.parquet'))}')"
            ).fetchone()[0]
        )
        log(f"[near_dedup/candidates] completed candidate_pair_count={count}")
        return count
    finally:
        con.close()
        if db_path.exists():
            db_path.unlink()


@dataclass(frozen=True)
class VerifiedPairsResult:
    verified_pairs: list[tuple[str, str, float]]
    metadata_by_locator: dict[str, RowMetadata]
    verified_pair_count: int


def _verify_candidate_pairs(
    *,
    signature_files: list[Path],
    candidates_root: Path,
    jaccard_threshold: float,
    fetch_record_batch_rows: int,
    minhash_ngrams: int,
) -> VerifiedPairsResult:
    signature_lookup: dict[str, RowMetadata] = {}
    for signature_path in signature_files:
        for row in pq.read_table(signature_path).to_pylist():
            signature_lookup[row["record_locator"]] = RowMetadata(
                record_locator=row["record_locator"],
                source=row["source"],
                data_usage=row["data_usage"],
                split=row["split"],
                content=row["content"],
                content_length=row["content_length"],
            )

    candidate_path = candidates_root / "pairs.parquet"
    verified_pairs: list[tuple[str, str, float]] = []
    cache: dict[str, set[str]] = {}
    parquet_file = pq.ParquetFile(candidate_path)
    total_pairs = parquet_file.metadata.num_rows
    progress = progress_bar(total=total_pairs, desc="near_dedup/verify")
    try:
        for batch in parquet_file.iter_batches(batch_size=fetch_record_batch_rows):
            for pair in batch.to_pylist():
                left = signature_lookup[pair["left_locator"]]
                right = signature_lookup[pair["right_locator"]]
                left_shingles = cache.setdefault(
                    left.record_locator,
                    _build_shingles(text=left.content, n=minhash_ngrams),
                )
                right_shingles = cache.setdefault(
                    right.record_locator,
                    _build_shingles(text=right.content, n=minhash_ngrams),
                )
                score = _compute_jaccard(left_shingles, right_shingles)
                if score >= jaccard_threshold:
                    verified_pairs.append((left.record_locator, right.record_locator, score))
            progress.update(batch.num_rows)
    finally:
        progress.close()

    metadata_by_locator = {
        locator: signature_lookup[locator]
        for pair in verified_pairs
        for locator in pair[:2]
    }
    log(
        "[near_dedup/verify] completed "
        f"candidate_pairs={total_pairs} verified_pairs={len(verified_pairs)}"
    )
    return VerifiedPairsResult(
        verified_pairs=verified_pairs,
        metadata_by_locator=metadata_by_locator,
        verified_pair_count=len(verified_pairs),
    )


def _compute_jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    intersection = len(left & right)
    union = len(left | right)
    if union == 0:
        return 0.0
    return intersection / union


@dataclass(frozen=True)
class ClusterResult:
    cluster_count: int
    removed_locators: set[str]
    deleted_records: list[dict[str, Any]]


def _build_clusters(*, verified: VerifiedPairsResult) -> ClusterResult:
    if not verified.verified_pairs:
        return ClusterResult(cluster_count=0, removed_locators=set(), deleted_records=[])

    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]

    def union(left: str, right: str) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    pair_score_map: dict[tuple[str, str], float] = {}
    for left, right, score in verified.verified_pairs:
        union(left, right)
        pair_score_map[(left, right)] = score
        pair_score_map[(right, left)] = score

    members_by_root: dict[str, list[str]] = defaultdict(list)
    for locator in verified.metadata_by_locator:
        members_by_root[find(locator)].append(locator)

    removed_locators: set[str] = set()
    deleted_records: list[dict[str, Any]] = []
    for members in members_by_root.values():
        if len(members) <= 1:
            continue
        representative = _select_representative(
            [verified.metadata_by_locator[locator] for locator in members]
        )
        for locator in members:
            if locator == representative.record_locator:
                continue
            removed_locators.add(locator)
            member = verified.metadata_by_locator[locator]
            score = pair_score_map.get((locator, representative.record_locator))
            deleted_records.append(
                {
                    "record_locator": locator,
                    "representative_locator": representative.record_locator,
                    "source": member.source,
                    "data_usage": member.data_usage,
                    "split": member.split,
                    "content": member.content,
                    "reason_label": _representative_reason(
                        member=member,
                        representative=representative,
                    ),
                    "jaccard": score,
                }
            )
    return ClusterResult(
        cluster_count=sum(1 for members in members_by_root.values() if len(members) > 1),
        removed_locators=removed_locators,
        deleted_records=deleted_records,
    )


def _select_representative(members: list[RowMetadata]) -> RowMetadata:
    return sorted(
        members,
        key=lambda row: (
            -USAGE_PRIORITY[row.data_usage],
            -row.content_length,
            row.record_locator,
        ),
    )[0]


def _representative_reason(*, member: RowMetadata, representative: RowMetadata) -> str:
    if USAGE_PRIORITY[representative.data_usage] > USAGE_PRIORITY[member.data_usage]:
        return "higher_data_usage"
    if representative.content_length > member.content_length:
        return "longer_content"
    return "stable_locator_tiebreak"


def _write_deleted_rows_log(
    *,
    run_id: str,
    deleted_log_path: Path,
    deleted_records: list[dict[str, Any]],
    threshold: float,
) -> tuple[int, Counter[str]]:
    ensure_parent_dir(deleted_log_path)
    reason_counts: Counter[str] = Counter()
    progress = progress_bar(total=len(deleted_records), desc="near_dedup/deleted_log")
    try:
        with deleted_log_path.open("w", encoding="utf-8") as f:
            for record in deleted_records:
                reason_counts[record["reason_label"]] += 1
                payload = {
                    "run_id": run_id,
                    "stage": "minhash_lsh_dedup",
                    "source": record["source"],
                    "data_usage": record["data_usage"],
                    "split": record["split"],
                    "content_hash": hashlib.sha256(
                        record["content"].encode("utf-8")
                    ).hexdigest(),
                    "record_locator": record["record_locator"],
                    "reason_code": "near_duplicate",
                    "reason_detail": (
                        f"verified near duplicate with jaccard>={threshold}"
                        if record["jaccard"] is None
                        else f"verified near duplicate with jaccard={record['jaccard']:.6f} threshold={threshold}"
                    ),
                    "representative_locator": record["representative_locator"],
                    "content_preview": truncate_sample(record["content"]),
                }
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
                progress.update(1)
    finally:
        progress.close()
    return len(deleted_records), reason_counts


def _write_retained_rows(
    *,
    input_shards: list[Path],
    passthrough_root: Path,
    output_root: Path,
    target_shard_bytes: int,
    removed_locators: set[str],
) -> list[NearDedupShardResult]:
    results: list[NearDedupShardResult] = []
    current_batches: list[pa.RecordBatch] = []
    current_rows = 0
    current_bytes = 0
    shard_index = 1
    total_rows = sum(1 for _ in _iter_canonical_rows(input_shards))
    progress = progress_bar(total=total_rows, desc="near_dedup/retained")
    try:
        for row in _iter_canonical_rows(input_shards):
            if row["record_locator"] in removed_locators:
                progress.update(1)
                continue
            retained_row = {
                "source": row["source"],
                "data_usage": row["data_usage"],
                "split": row["split"],
                "content": row["content"],
                "messages": row["messages"],
                "token_count": row["token_count"],
            }
            row_batch = pa.RecordBatch.from_pylist([retained_row], schema=STAGING_SCHEMA)
            current_batches.append(row_batch)
            current_rows += 1
            current_bytes += row_batch.nbytes
            progress.update(1)
            if current_bytes > target_shard_bytes:
                results.append(
                    _flush_output_shard(
                        output_root=output_root,
                        batches=current_batches,
                        shard_index=shard_index,
                        row_count=current_rows,
                        approx_nbytes=current_bytes,
                    )
                )
                shard_index += 1
                current_batches = []
                current_rows = 0
                current_bytes = 0
        if current_batches:
            results.append(
                _flush_output_shard(
                    output_root=output_root,
                    batches=current_batches,
                    shard_index=shard_index,
                    row_count=current_rows,
                    approx_nbytes=current_bytes,
                )
            )
    finally:
        progress.close()
    return results


def _flush_output_shard(
    *,
    output_root: Path,
    batches: list[pa.RecordBatch],
    shard_index: int,
    row_count: int,
    approx_nbytes: int,
) -> NearDedupShardResult:
    table = pa.Table.from_batches(batches).cast(STAGING_SCHEMA)
    shard_path = output_root / f"part-{shard_index:06d}.parquet"
    ensure_parent_dir(shard_path)
    pq.write_table(table, shard_path)
    log(
        f"[near_dedup/shard] wrote {shard_path} rows={row_count} approx_nbytes={approx_nbytes}"
    )
    return NearDedupShardResult(
        shard_path=str(shard_path),
        row_count=row_count,
        approx_nbytes=approx_nbytes,
    )


def _count_from_deleted(
    deleted_records: list[dict[str, Any]],
    *,
    key: str,
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in deleted_records:
        counts[str(record[key])] += 1
    return dict(sorted(counts.items()))


def _subtract_counts(before: dict[str, int], deleted: dict[str, int]) -> dict[str, int]:
    after: dict[str, int] = {}
    for key, value in before.items():
        after[key] = value - deleted.get(key, 0)
    return dict(sorted(after.items()))


def _write_summary_files(
    *,
    summary: NearDedupSummary,
    summary_json_path: Path,
    manifest_path: Path,
) -> None:
    write_json(
        summary_json_path,
        {
            "deleted_row_count": summary.deleted_row_count,
            "candidate_pair_count": summary.candidate_pair_count,
            "verified_pair_count": summary.verified_pair_count,
            "near_cluster_count": summary.near_cluster_count,
            "source_counts_deleted": summary.source_counts_deleted,
            "data_usage_counts_deleted": summary.data_usage_counts_deleted,
            "representative_reason_counts": summary.representative_reason_counts,
            "jaccard_threshold": summary.jaccard_threshold,
        },
    )
    write_json(manifest_path, summary.to_dict())


def _sql_quote(text: str) -> str:
    return text.replace("'", "''")


def main() -> None:
    common.ENABLE_TQDM = ENABLE_TQDM
    common.ENABLE_DEBUG_LOG = ENABLE_DEBUG_LOG
    common.TQDM_MININTERVAL_SEC = TQDM_MININTERVAL_SEC
    run_near_dedup(
        input_root=INPUT_ROOT,
        output_root=OUTPUT_ROOT,
        intermediate_root=INTERMEDIATE_ROOT,
        target_shard_bytes=TARGET_SHARD_BYTES,
        minhash_ngrams=MINHASH_NGRAMS,
        minhash_num_buckets=MINHASH_NUM_BUCKETS,
        minhash_hashes_per_bucket=MINHASH_HASHES_PER_BUCKET,
        minhash_hash_precision=MINHASH_HASH_PRECISION,
        jaccard_threshold=JACCARD_THRESHOLD,
        fetch_record_batch_rows=FETCH_RECORD_BATCH_ROWS,
        duckdb_threads=DUCKDB_THREADS,
        overwrite_output=OVERWRITE_OUTPUT,
    )


if __name__ == "__main__":
    main()
