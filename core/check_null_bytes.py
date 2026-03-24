from __future__ import annotations

from pathlib import Path


def main() -> None:
    bad_files = []
    for path in Path('.').rglob('*.py'):
        if '.venv' in path.parts:
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

