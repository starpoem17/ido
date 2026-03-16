from __future__ import annotations

from urllib.parse import quote

from src.collect import namu


# =========================
# User configuration block
# =========================
PROMPT_TEXT = "namu> "
EXIT_COMMAND = "exit"
ALLOW_FULL_URL_INPUT = True
VERBOSE = True
LOG_PREFIX = "[run_namu]"


def log(message: str) -> None:
    if VERBOSE:
        print(f"{LOG_PREFIX} {message}")


def build_url_from_title(title: str) -> str:
    return f"{namu.NAMU_BASE_URL}/w/{quote(title, safe='')}"


def normalize_user_input(raw: str) -> str | None:
    text = raw.strip()
    if not text:
        return None
    if ALLOW_FULL_URL_INPUT and text.startswith(f"{namu.NAMU_BASE_URL}/w/"):
        return text
    return build_url_from_title(text)


def main() -> None:
    namu.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log(
        f"waiting for input. enter a namuwiki title or full url. "
        f"type '{EXIT_COMMAND}' to quit."
    )

    while True:
        try:
            raw = input(PROMPT_TEXT)
        except EOFError:
            log("received EOF, exiting.")
            break
        except KeyboardInterrupt:
            print()
            log("received keyboard interrupt, exiting.")
            break

        user_text = raw.strip()
        if not user_text:
            log("empty input, waiting for next input.")
            continue
        if user_text.lower() == EXIT_COMMAND.lower():
            log("exit requested.")
            break

        normalized_url = normalize_user_input(user_text)
        if normalized_url is None:
            log("invalid input, waiting for next input.")
            continue

        log(f"normalized_url={normalized_url}")
        try:
            output_path = namu.process_url(normalized_url)
            log(f"saved_path={output_path}")
        except Exception as exc:  # noqa: BLE001
            log(
                f"failed input={user_text} error={type(exc).__name__}: {exc}"
            )


if __name__ == "__main__":
    main()
