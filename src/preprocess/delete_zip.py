from __future__ import annotations

from pathlib import Path


# =========================
# User configuration block
# =========================
ROOT_DIR = Path("data/korean")
DRY_RUN = False
VERBOSE = True
FAILED_PRINT_LIMIT = 20


def collect_zip_files(root: Path) -> list[Path]:
    return sorted(
        (p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".zip"),
        key=lambda p: str(p),
    )


def delete_zip_files(zip_paths: list[Path], dry_run: bool) -> dict:
    deleted = 0
    failed_items: list[str] = []

    for zip_path in zip_paths:
        try:
            if not dry_run:
                zip_path.unlink()
            deleted += 1
        except Exception as exc:  # noqa: BLE001
            failed_items.append(f"{zip_path}: {type(exc).__name__}: {exc}")

    return {
        "total": len(zip_paths),
        "deleted": deleted,
        "failed": len(failed_items),
        "failed_items": failed_items,
        "remaining_after": 0,
    }


def main() -> None:
    root = ROOT_DIR.resolve()
    if not root.exists():
        raise FileNotFoundError(f"root directory not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"root path is not a directory: {root}")

    before = collect_zip_files(root)
    result = delete_zip_files(before, DRY_RUN)
    after = collect_zip_files(root)
    result["remaining_after"] = len(after)

    if VERBOSE:
        mode = "DRY_RUN" if DRY_RUN else "DELETE"
        print(f"[mode] {mode}")
        print(f"[root] {root}")
        print(f"[total_zip_before] {result['total']}")
        print(f"[deleted_or_planned] {result['deleted']}")
        print(f"[failed] {result['failed']}")
        print(f"[remaining_after] {result['remaining_after']}")

        if result["failed"] > 0:
            print("[failed_items]")
            for item in result["failed_items"][:FAILED_PRINT_LIMIT]:
                print(item)
            if result["failed"] > FAILED_PRINT_LIMIT:
                print(f"... and {result['failed'] - FAILED_PRINT_LIMIT} more")

    if result["failed"] > 0:
        raise SystemExit(1)

    raise SystemExit(0)


if __name__ == "__main__":
    main()
