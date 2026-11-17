import os
import sys

if sys.platform == "darwin":
    # torch and faiss-cpu each bring an OpenMP runtime on macOS; without this the test process
    # aborts as soon as both are imported (cli.py sets the same flag for the CLI).
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
