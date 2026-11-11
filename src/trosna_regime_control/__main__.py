"""Allow ``python -m trosna_regime_control``."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
