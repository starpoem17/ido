from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple


# =========================
# User Settings (edit here)
# =========================
TOKENIZERS_ROOT = Path("data/tokenizers")
TOKENIZER_JSON_NAME = "tokenizer.json"
INPUT_PROMPT = "Enter text to tokenize: "
ENCODING = "utf-8"


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


def to_readable_text(decoded: str) -> str:
    out: List[str] = []
    for ch in decoded:
        if ch == " ":
            out.append("<space>")
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


def is_roundtrip_success_prefix_space_aware(text: str, decoded: str) -> bool:
    return decoded == text or decoded == f" {text}"


def find_tokenizer_json_paths(root: Path) -> List[Path]:
    if not root.exists():
        return []
    paths = sorted(p for p in root.glob(f"*/{TOKENIZER_JSON_NAME}") if p.is_file())
    return paths


def display_tokenizer_choices(paths: Sequence[Path], root: Path) -> None:
    print("Available tokenizers:")
    for i, path in enumerate(paths, start=1):
        rel = path.parent.relative_to(root)
        print(f"  {i}. {rel} ({path})")


def select_tokenizer_path(paths: Sequence[Path], root: Path) -> Path:
    if not paths:
        raise RuntimeError(f"No tokenizer files found under: {root}")

    if len(paths) == 1:
        only = paths[0]
        rel = only.parent.relative_to(root)
        print(f"Auto-selected tokenizer: {rel} ({only})")
        return only

    display_tokenizer_choices(paths, root)
    while True:
        raw = input("Select tokenizer number: ").strip()
        if not raw:
            print("Please enter a number.")
            continue
        if not raw.isdigit():
            print("Invalid input. Enter digits only.")
            continue
        idx = int(raw)
        if 1 <= idx <= len(paths):
            chosen = paths[idx - 1]
            rel = chosen.parent.relative_to(root)
            print(f"Selected tokenizer: {rel} ({chosen})")
            return chosen
        print(f"Out of range. Choose 1..{len(paths)}.")


def load_tokenizer_metadata(tokenizer_json_path: Path) -> Tuple[Set[str], bool]:
    import json

    with tokenizer_json_path.open("r", encoding=ENCODING) as f:
        obj = json.load(f)

    special_tokens: Set[str] = set()
    added = obj.get("added_tokens", [])
    if not isinstance(added, list):
        added = []
    for item in added:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        special = item.get("special")
        if isinstance(content, str) and bool(special):
            special_tokens.add(content)

    add_prefix_space = False
    pre_tokenizer = obj.get("pre_tokenizer")
    if isinstance(pre_tokenizer, dict):
        add_prefix_space = bool(pre_tokenizer.get("add_prefix_space", False))

    return special_tokens, add_prefix_space


def classify_roundtrip(
    text: str,
    decoded: str,
    *,
    add_prefix_space: bool,
) -> Tuple[bool, bool, bool, Optional[str]]:
    strict_ok = decoded == text
    prefix_ok = is_roundtrip_success_prefix_space_aware(text, decoded)
    reason: Optional[str] = None

    if not strict_ok:
        if decoded == f" {text}" and add_prefix_space:
            reason = "leading_space_from_add_prefix_space"
        elif decoded == f" {text}":
            reason = "leading_space_unexpected"
        else:
            reason = "content_mismatch"

    strict_effective_ok = strict_ok or reason == "leading_space_from_add_prefix_space"
    return strict_ok, prefix_ok, strict_effective_ok, reason


def safe_div(numer: float, denom: float) -> float:
    if denom == 0:
        return 0.0
    return numer / denom


def format_ms(sec: float) -> str:
    return f"{sec * 1000.0:.3f} ms"


def main() -> None:
    try:
        from tokenizers import Tokenizer
    except ImportError:
        print("tokenizers is required. Run with `uv run python src/tokenizer/test.py`.")
        raise SystemExit(1)

    tokenizer_paths = find_tokenizer_json_paths(TOKENIZERS_ROOT)
    try:
        tokenizer_json_path = select_tokenizer_path(tokenizer_paths, TOKENIZERS_ROOT)
    except RuntimeError as exc:
        print(str(exc))
        raise SystemExit(1)

    special_tokens, add_prefix_space = load_tokenizer_metadata(tokenizer_json_path)
    tokenizer = Tokenizer.from_file(str(tokenizer_json_path))
    u2b = unicode_to_bytes_map()

    text = input(INPUT_PROMPT)
    if text == "":
        print("Empty input. Exit.")
        return

    t0 = time.perf_counter()
    enc = tokenizer.encode(text)
    t1 = time.perf_counter()
    decoded = tokenizer.decode(enc.ids, skip_special_tokens=False)
    t2 = time.perf_counter()

    char_count = len(text)
    token_count = len(enc.ids)
    tokens_per_char = safe_div(float(token_count), float(char_count))
    encode_sec = t1 - t0
    decode_sec = t2 - t1
    total_sec = t2 - t0

    strict_ok, prefix_ok, strict_effective_ok, strict_failure_reason = classify_roundtrip(
        text,
        decoded,
        add_prefix_space=add_prefix_space,
    )

    token_rows: List[Dict[str, object]] = []
    utf8_count = 0
    bytes_count = 0
    raw_count = 0
    special_count = 0
    for i, (tid, raw_tok) in enumerate(zip(enc.ids, enc.tokens)):
        if raw_tok in special_tokens:
            readable_tok = raw_tok
            kind = "special"
            special_count += 1
        else:
            readable_tok, kind = decode_bytelevel_token(raw_tok, u2b)
            if kind == "utf8":
                utf8_count += 1
            elif kind == "bytes":
                bytes_count += 1
            else:
                raw_count += 1

        token_rows.append(
            {
                "index": i,
                "id": tid,
                "raw": raw_tok,
                "readable": readable_tok,
                "kind": kind,
            }
        )

    byte_fragment_ratio = safe_div(float(bytes_count), float(token_count))

    print()
    print("=== Tokenizer Test Summary ===")
    print(f"Tokenizer file            : {tokenizer_json_path}")
    print(f"add_prefix_space          : {add_prefix_space}")
    print(f"Input chars (len)         : {char_count}")
    print(f"Token count               : {token_count}")
    print(f"Tokens per char           : {tokens_per_char:.6f}")
    print(f"Encode time               : {format_ms(encode_sec)}")
    print(f"Decode time               : {format_ms(decode_sec)}")
    print(f"Total time                : {format_ms(total_sec)}")
    print(f"Round-trip strict         : {strict_ok}")
    print(f"Round-trip prefix-aware   : {prefix_ok}")
    print(f"Round-trip strict-effective: {strict_effective_ok}")
    if strict_failure_reason is not None:
        print(f"Strict failure reason     : {strict_failure_reason}")
    print(f"UTF8 full-token count     : {utf8_count}")
    print(f"Byte-fragment token count : {bytes_count}")
    print(f"Special token count       : {special_count}")
    print(f"Raw-fallback token count  : {raw_count}")
    print(f"Byte-fragment ratio       : {byte_fragment_ratio:.6f}")

    print()
    print("=== Token Details ===")
    for row in token_rows:
        idx = int(row["index"])
        tid = int(row["id"])
        kind = str(row["kind"])
        readable = str(row["readable"])
        raw = str(row["raw"]).encode("unicode_escape").decode("ascii")
        print(f"[{idx:04d}] id={tid:<6d} kind={kind:<7s} readable={readable} raw={raw}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
