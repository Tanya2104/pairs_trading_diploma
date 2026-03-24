from __future__ import annotations

from pathlib import Path


TEXT_EXTENSIONS = {
    ".py",
    ".md",
    ".txt",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".ini",
    ".cfg",
    ".ps1",
    ".sh",
    ".gitignore",
}
SKIP_PARTS = {".git", "__pycache__", ".venv", "venv"}


def _is_text_candidate(path: Path) -> bool:
    if path.name == ".gitignore":
        return True
    return path.suffix.lower() in TEXT_EXTENSIONS


def main() -> None:
    bad_files = []

    for path in Path('.').rglob('*'):
        if not path.is_file():
            continue
        if any(part in SKIP_PARTS for part in path.parts):
            continue
        if not _is_text_candidate(path):
            continue

        data = path.read_bytes()
        if b'\x00' in data:
            bad_files.append(path.as_posix())

    if bad_files:
        print('NULL_BYTES_FOUND')
        for file in bad_files:
            print(file)
        raise SystemExit(1)

    print('NULL_BYTES_NOT_FOUND')


if __name__ == '__main__':
    main()