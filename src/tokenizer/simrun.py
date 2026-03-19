from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from src.tokenizer import common


TOKENIZER_JSON_PATH = Path("data/tokenizers/korean_bbpe_v2/tokenizer.json")
MAX_DISPLAY_TOKENS = 64
SHOW_TOKEN_BREAKDOWN = True
EXIT_COMMANDS = ("exit", "quit", ":q")


@dataclass(frozen=True)
class SimrunConfig:
    tokenizer_json_path: Path
    max_display_tokens: int
    show_token_breakdown: bool
    exit_commands: tuple[str, ...]


def build_default_config() -> SimrunConfig:
    return SimrunConfig(
        tokenizer_json_path=TOKENIZER_JSON_PATH,
        max_display_tokens=MAX_DISPLAY_TOKENS,
        show_token_breakdown=SHOW_TOKEN_BREAKDOWN,
        exit_commands=EXIT_COMMANDS,
    )


def run_simrun(
    config: SimrunConfig | None = None,
    *,
    input_fn=input,
    output_stream: TextIO = sys.stdout,
) -> None:
    config = config or build_default_config()
    if not config.tokenizer_json_path.exists():
        raise ValueError(f"tokenizer file not found: {config.tokenizer_json_path}")
    tokenizer = common.load_tokenizer(config.tokenizer_json_path)
    print(f"Tokenizer: {config.tokenizer_json_path}", file=output_stream)
    print(
        f"Enter text. Exit commands: {', '.join(config.exit_commands)}",
        file=output_stream,
    )
    while True:
        raw = input_fn("text> ")
        if raw in config.exit_commands:
            print("bye", file=output_stream)
            return
        if raw == "":
            print("empty input", file=output_stream)
            continue
        encoded = tokenizer.encode(raw)
        stats = compute_compression_stats(raw, len(encoded.ids))
        print(f"text              : {raw}", file=output_stream)
        print(f"char_count        : {stats['char_count']}", file=output_stream)
        print(f"utf8_byte_count   : {stats['utf8_byte_count']}", file=output_stream)
        print(f"token_count       : {stats['token_count']}", file=output_stream)
        print(f"tokens_per_char   : {stats['tokens_per_char']:.6f}", file=output_stream)
        print(f"chars_per_token   : {stats['chars_per_token']:.6f}", file=output_stream)
        print(f"tokens_per_byte   : {stats['tokens_per_byte']:.6f}", file=output_stream)
        if config.show_token_breakdown:
            print("token_breakdown:", file=output_stream)
            for row in build_token_breakdown(
                tokenizer, encoded.ids, config.max_display_tokens
            ):
                print(
                    (
                        f"  [{row['index']}] id={row['token_id']} "
                        f"raw={row['raw_token']} "
                        f"display={row['display_token']}"
                    ),
                    file=output_stream,
                )


def compute_compression_stats(text: str, token_count: int) -> dict[str, float | int]:
    char_count = len(text)
    utf8_byte_count = len(text.encode("utf-8"))
    return {
        "char_count": char_count,
        "utf8_byte_count": utf8_byte_count,
        "token_count": token_count,
        "tokens_per_char": (token_count / char_count) if char_count else 0.0,
        "chars_per_token": (char_count / token_count) if token_count else 0.0,
        "tokens_per_byte": (token_count / utf8_byte_count) if utf8_byte_count else 0.0,
    }


def build_token_breakdown(tokenizer, token_ids: list[int], max_display_tokens: int) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for index, token_id in enumerate(token_ids[:max_display_tokens]):
        raw_token = common.id_to_raw_token(tokenizer, token_id)
        rows.append(
            {
                "index": index,
                "token_id": token_id,
                "raw_token": common.escape_for_display(raw_token),
                "display_token": common.decode_token_for_display(tokenizer, token_id),
            }
        )
    if len(token_ids) > max_display_tokens:
        rows.append(
            {
                "index": max_display_tokens,
                "token_id": -1,
                "raw_token": "...",
                "display_token": f"... {len(token_ids) - max_display_tokens} more tokens",
            }
        )
    return rows


def main() -> None:
    run_simrun(build_default_config())


if __name__ == "__main__":
    main()
