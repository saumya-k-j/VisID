import os
import sys

if sys.platform == "darwin":
    # torch and faiss-cpu each bring an OpenMP runtime on macOS; without this the test process
    # aborts as soon as both are imported (cli.py sets the same flag for the CLI).
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    # Two OpenMP runtimes in one process can still segfault once FAISS runs a parallel region
    # (seen: pytest crashed in the first FAISS Flat search after torch was imported). One OpenMP
    # thread avoids it; the benchmark already runs FAISS on 1 thread (bench.faiss_threads).
    os.environ.setdefault("OMP_NUM_THREADS", "1")
