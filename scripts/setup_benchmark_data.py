#!/usr/bin/env python3
import argparse
import shutil
import sys
import urllib.request
from pathlib import Path


WIKI_URL = (
    "https://dumps.wikimedia.org/simplewiki/latest/"
    "simplewiki-latest-pages-articles.xml.bz2"
)


def download_file(url: str, dst: Path, force: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not force:
        print(f"skip: already exists: {dst}")
        return

    tmp = dst.with_suffix(dst.suffix + ".part")
    print(f"download: {url}")
    with urllib.request.urlopen(url) as resp, tmp.open("wb") as out:
        shutil.copyfileobj(resp, out)
    tmp.replace(dst)
    print(f"saved: {dst}")


def prefetch_20newsgroups(data_home: Path) -> None:
    try:
        from sklearn.datasets import fetch_20newsgroups
    except ImportError as exc:
        raise SystemExit(
            "scikit-learn is not installed. Install it to prefetch 20 Newsgroups."
        ) from exc

    data_home.mkdir(parents=True, exist_ok=True)
    print(f"prefetch: 20 Newsgroups into {data_home}")
    ds = fetch_20newsgroups(
        subset="all",
        remove=("headers", "footers", "quotes"),
        data_home=str(data_home),
    )
    print(f"ready: 20 Newsgroups docs={len(ds.data)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and prepare local datasets needed for benchmarks."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root (default: parent of this script)",
    )
    parser.add_argument(
        "--skip-wikipedia",
        action="store_true",
        help="Skip downloading the simplewiki dump",
    )
    parser.add_argument(
        "--skip-20newsgroups",
        action="store_true",
        help="Skip prefetching sklearn 20 Newsgroups cache",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if already present",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()

    wiki_dst = root / "data" / "simplewiki-latest-pages-articles.xml.bz2"
    sklearn_cache = root / ".cache" / "scikit_learn_data"

    if not args.skip_wikipedia:
        download_file(WIKI_URL, wiki_dst, force=args.force)
    else:
        print("skip: wikipedia")

    if not args.skip_20newsgroups:
        prefetch_20newsgroups(sklearn_cache)
    else:
        print("skip: 20newsgroups")

    print("\nsetup complete")
    print(f"- wikipedia dump: {wiki_dst}")
    print(f"- 20newsgroups cache: {sklearn_cache}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
