from __future__ import annotations

from pathlib import Path


def main() -> None:
    corpus_dir = Path("data/corpus")
    corpus_dir.mkdir(parents=True, exist_ok=True)
    print("[build_index] Placeholder implementation.")
    print(f"[build_index] Put source documents under {corpus_dir} and replace this with a real index builder.")


if __name__ == "__main__":
    main()

