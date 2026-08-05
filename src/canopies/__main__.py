# -*- coding: utf-8 -*-
"""Allow ``python -m canopies ...`` as an alias of the ``canopies`` script."""
from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
