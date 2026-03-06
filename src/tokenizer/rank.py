from __future__ import annotations

import csv
import datetime as dt
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


# =========================
# User Settings (edit here)
# =========================
TOKENIZER_JSON_PATH = Path("data/tokenizers/korean_bbpe_v1/tokenizer.json")
OUTPUT_DIR = Path("data/tokenizers/korean_bbpe_v1")
OUTPUT_PREFIX = "token_proxy_ranking"
TOP_K = 100
ENCODING = "utf-8"


def load_tokenizer_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding=ENCODING) as f:
        return json.load(f)


def bytes_to_unicode() -> Dict[int, str]:
    # Mirrors GPT-2 ByteLevel mapping used by ByteLevelBPE tokenizers.
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def unicode_to_bytes_map() -> Dict[str, int]:
    return {v: k for k, v in bytes_to_unicode().items()}


def build_special_token_set(added_tokens: Iterable[object]) -> Set[str]:
    out: Set[str] = set()
    for item in added_tokens:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        special = item.get("special")
        if isinstance(content, str) and bool(special):
            out.add(content)
    return out


def parse_merge_pair(entry: object) -> Optional[Tuple[str, str]]:
    if isinstance(entry, list) and len(entry) == 2:
        left, right = entry
        if isinstance(left, str) and isinstance(right, str):
            return left, right
        return None
    if isinstance(entry, str):
        parts = entry.split(" ")
        if len(parts) == 2:
            return parts[0], parts[1]
    return None


def build_merge_rank_map(merges: Iterable[object]) -> Dict[str, int]:
    rank_map: Dict[str, int] = {}
    for idx, entry in enumerate(merges):
        pair = parse_merge_pair(entry)
        if pair is None:
            continue
        merged = pair[0] + pair[1]
        if merged not in rank_map:
            rank_map[merged] = idx
    return rank_map


def to_readable_text(decoded: str) -> str:
    out: List[str] = []
    for ch in decoded:
        if ch == " ":
            out.append("␠")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\t":
            out.append("\\t")
        elif ch == "\r":
            out.append("\\r")
        else:
            code = ord(ch)
            if code < 32 or code == 127:
                out.append(f"\\x{code:02X}")
            else:
                out.append(ch)
    return "".join(out)


def decode_bytelevel_token(token: str, u2b: Dict[str, int]) -> Tuple[str, str]:
    try:
        b = bytes([u2b[ch] for ch in token])
    except KeyError:
        return token.encode("unicode_escape").decode("ascii"), "raw"

    try:
        decoded = b.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        hex_bytes = " ".join(f"{x:02X}" for x in b)
        return f"<bytes:{hex_bytes}>", "bytes"
    return to_readable_text(decoded), "utf8"


def display_raw_token(token: str, max_len: int = 48) -> str:
    s = token.encode("unicode_escape").decode("ascii")
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def display_readable_token(token: str, max_len: int = 48) -> str:
    if len(token) <= max_len:
        return token
    return token[: max_len - 3] + "..."


def make_rows(
    vocab: Dict[str, int],
    merge_rank_map: Dict[str, int],
    special_tokens: Set[str],
    merge_count: int,
    unicode_to_bytes: Dict[str, int],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for token, token_id in vocab.items():
        merge_rank = merge_rank_map.get(token)
        is_merge_token = merge_rank is not None
        sort_key = merge_rank if merge_rank is not None else math.inf
        proxy_score = (merge_count - merge_rank) if merge_rank is not None else 0
        if token in special_tokens:
            readable_token = token
            readable_kind = "special"
        else:
            readable_token, readable_kind = decode_bytelevel_token(token, unicode_to_bytes)
        rows.append(
            {
                "token": token,
                "readable_token": readable_token,
                "readable_kind": readable_kind,
                "token_id": token_id,
                "is_special": token in special_tokens,
                "is_merge_token": is_merge_token,
                "merge_rank": merge_rank,
                "proxy_score": proxy_score,
                "_sort_key": sort_key,
            }
        )

    rows.sort(key=lambda x: (x["_sort_key"], x["token_id"], x["token"]))
    for i, row in enumerate(rows, start=1):
        row["rank"] = i
        del row["_sort_key"]
    return rows


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fieldnames = [
        "rank",
        "token",
        "readable_token",
        "readable_kind",
        "token_id",
        "is_special",
        "is_merge_token",
        "merge_rank",
        "proxy_score",
    ]
    with path.open("w", encoding=ENCODING, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    with path.open("w", encoding=ENCODING) as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def print_topk(rows: List[Dict[str, object]], k: int) -> None:
    top = rows[:k]
    print()
    print(
        "Top tokens by proxy frequency "
        "(metric=bpe_merge_rank_proxy, not_actual_frequency=true, readable_token=true)"
    )
    print("-" * 138)
    print(
        f"{'rank':>6}  {'token_id':>8}  {'merge_rank':>10}  "
        f"{'score':>8}  {'special':>7}  {'kind':>7}  readable_token  raw_token"
    )
    print("-" * 138)
    for row in top:
        mr = row["merge_rank"]
        mr_s = str(mr) if mr is not None else "-"
        print(
            f"{int(row['rank']):>6}  {int(row['token_id']):>8}  {mr_s:>10}  "
            f"{int(row['proxy_score']):>8}  {str(bool(row['is_special'])):>7}  "
            f"{str(row['readable_kind']):>7}  "
            f"{display_readable_token(str(row['readable_token']), max_len=48):<48}  "
            f"{display_raw_token(str(row['token']), max_len=28)}"
        )
    print("-" * 138)
    print(f"Displayed top {len(top)} of {len(rows)} tokens.")
    print()


def main() -> None:
    tokenizer_json = load_tokenizer_json(TOKENIZER_JSON_PATH)

    model = tokenizer_json.get("model")
    if not isinstance(model, dict):
        raise RuntimeError("invalid tokenizer.json: missing model object")
    vocab = model.get("vocab")
    merges = model.get("merges")
    if not isinstance(vocab, dict):
        raise RuntimeError("invalid tokenizer.json: model.vocab is missing or invalid")
    if not isinstance(merges, list):
        raise RuntimeError("invalid tokenizer.json: model.merges is missing or invalid")

    added_tokens = tokenizer_json.get("added_tokens", [])
    if not isinstance(added_tokens, list):
        added_tokens = []

    special_tokens = build_special_token_set(added_tokens)
    merge_rank_map = build_merge_rank_map(merges)
    u2b = unicode_to_bytes_map()
    rows = make_rows(
        vocab=vocab,
        merge_rank_map=merge_rank_map,
        special_tokens=special_tokens,
        merge_count=len(merges),
        unicode_to_bytes=u2b,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}.csv"
    top_json_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_top{TOP_K}.json"
    meta_json_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_meta.json"

    write_csv(csv_path, rows)
    write_json(top_json_path, rows[:TOP_K])
    write_json(
        meta_json_path,
        {
            "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
            "tokenizer_json_path": str(TOKENIZER_JSON_PATH),
            "output_dir": str(OUTPUT_DIR),
            "output_prefix": OUTPUT_PREFIX,
            "top_k": TOP_K,
            "metric": "bpe_merge_rank_proxy",
            "not_actual_frequency": True,
            "readable_token_enabled": True,
            "readable_fallback": "hex_bytes",
            "token_count": len(rows),
            "merge_count": len(merges),
            "special_token_count": len(special_tokens),
        },
    )
    print_topk(rows, TOP_K)
    print(f"Saved CSV: {csv_path}")
    print(f"Saved Top-{TOP_K} JSON: {top_json_path}")
    print(f"Saved metadata JSON: {meta_json_path}")


if __name__ == "__main__":
    main()
