# Minhashlib

This is a minimal implementation of MinHashing as described in Jeffrey Ullman's book *Mining Massive Datasets*. It is faster than datasketch by a significant amount in the extremely niche usecase that it was designed for, which is similarity checking strings for my ETL pipeline. As a general statement, I would say that minhashlib is good to use for similarity checking large amounts of strings efficiently but does not support the breadth of features that datasketch has (nor would I want it to at the moment). Contributions are welcome.

It can be installed via `pip install minhashlib`

If you want to run my benchmarks yourself, please check the wiki for detailed instructions.
