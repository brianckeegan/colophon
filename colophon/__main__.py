"""Executable module entrypoint for ``python -m colophon``."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
