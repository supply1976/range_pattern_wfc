from pathlib import Path
import sys

SRC_DIR = Path(__file__).parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from wfc.cli import main

if __name__ == '__main__':
    main()
