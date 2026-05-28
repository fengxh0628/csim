#!/usr/bin/env python3
import argparse
import hashlib
import logging
import os
import sys
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone

from dateutil.relativedelta import relativedelta
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("download_historical_data")

BASE_URL = "https://data.binance.vision/data"

DATA_TYPES_NEED_FREQUENCY = ("klines", "indexPriceKlines", "markPriceKlines", "premiumIndexKlines")

SUPPORTED_DATA_TYPES = (
    "aggTrades", "klines", "trades", "indexPriceKlines",
    "markPriceKlines", "premiumIndexKlines", "metrics",
    "fundingRate", "bookTicker",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download Binance historical data with incremental checksum verification"
    )
    parser.add_argument(
        "--asset-class", choices=("um", "cm"), default="um",
        help="Futures asset class (default: um)"
    )
    parser.add_argument(
        "--type", choices=SUPPORTED_DATA_TYPES, default="klines",
        help="Data type to download (default: klines)"
    )
    parser.add_argument(
        "--freq", default="5m",
        help="Data frequency for kline-like types (default: 5m)"
    )
    parser.add_argument(
        "--start-date",
        help="Start date. YYYY-MM-DD for daily, YYYY-MM for monthly"
    )
    parser.add_argument(
        "--end-date",
        help="End date. YYYY-MM-DD for daily, YYYY-MM for monthly. Default: yesterday"
    )
    parser.add_argument(
        "--base-dir", default="/mnt/f/binance_data",
        help="Base directory (default: /mnt/f/binance_data)"
    )
    parser.add_argument(
        "--universe", default=None,
        help="Path to universe symbol list (default: data/universe_top50_liquid.txt)"
    )
    parser.add_argument(
        "--max-workers", type=int, default=6,
        help="Max concurrent downloads (default: 6)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Skip checksum, always download"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only check and report what needs downloading, don't actually download"
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Verify checksums of existing files, no downloading"
    )
    parser.add_argument(
        "--days", type=int,
        help="Download last N days of daily data (UTC, exclusive of today)"
    )
    return parser.parse_args()


def load_symbols(universe_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if universe_path is None:
        universe_path = os.path.join(script_dir, "..", "data", "universe_top50_liquid.txt")
    with open(universe_path) as f:
        symbols = [line.strip() for line in f if line.strip()]
    logger.info("Loaded %d symbols from %s", len(symbols), universe_path)
    return symbols


def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_rel_dir(asset_class, timeperiod, data_type, symbol, freq=None):
    parts = ["futures", asset_class, timeperiod, data_type, symbol]
    if freq and data_type in DATA_TYPES_NEED_FREQUENCY:
        parts.append(freq)
    return "/".join(parts)


def build_filename(symbol, data_type, freq, date_str):
    if data_type in DATA_TYPES_NEED_FREQUENCY:
        return f"{symbol}-{freq}-{date_str}.zip"
    return f"{symbol}-{data_type}-{date_str}.zip"


def download_file(url, local_path):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    try:
        urllib.request.urlretrieve(url, local_path)
        return True
    except urllib.error.URLError:
        return False
    except Exception as e:
        logger.warning("Failed to download %s: %s", url, e)
        return False


def process_one(task):
    symbol, date_str, asset_class, timeperiod, data_type, freq, base_dir, force = task

    filename = build_filename(symbol, data_type, freq, date_str)
    rel_dir = build_rel_dir(asset_class, timeperiod, data_type, symbol, freq)
    local_dir = os.path.join(base_dir, rel_dir)
    local_zip = os.path.join(local_dir, filename)
    checksum_local = os.path.join(local_dir, f"{filename}.CHECKSUM")

    checksum_url = f"{BASE_URL}/{rel_dir}/{filename}.CHECKSUM"
    zip_url = f"{BASE_URL}/{rel_dir}/{filename}"

    need_download = True

    if not force and os.path.exists(local_zip):
        if os.path.exists(checksum_local):
            try:
                with open(checksum_local) as f:
                    expected = f.read().strip().split()[0]
                if expected and sha256_file(local_zip) == expected:
                    return None
            except Exception:
                pass

        if download_file(checksum_url, checksum_local):
            try:
                with open(checksum_local) as f:
                    expected = f.read().strip().split()[0]
                if expected and sha256_file(local_zip) == expected:
                    need_download = False
                else:
                    logger.info("Checksum mismatch for %s, re-downloading", filename)
            except Exception:
                pass

    if need_download:
        if download_file(zip_url, local_zip):
            return filename

    return None


def dry_run_check(tasks, max_workers):
    to_download = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(dry_run_one, t): t for t in tasks}
        with tqdm(total=len(futures), desc="Checking", unit="files", mininterval=1) as pbar:
            for f in as_completed(futures):
                r = f.result()
                if r:
                    to_download.append(r)
                pbar.update(1)
    logger.info("Files to download: %d", len(to_download))
    for name in sorted(to_download):
        print(f"  {name}")


def dry_run_one(task):
    symbol, date_str, asset_class, timeperiod, data_type, freq, base_dir, _ = task

    filename = build_filename(symbol, data_type, freq, date_str)
    rel_dir = build_rel_dir(asset_class, timeperiod, data_type, symbol, freq)
    local_zip = os.path.join(base_dir, rel_dir, filename)
    checksum_local = os.path.join(base_dir, rel_dir, f"{filename}.CHECKSUM")

    checksum_url = f"{BASE_URL}/{rel_dir}/{filename}.CHECKSUM"
    zip_url = f"{BASE_URL}/{rel_dir}/{filename}"

    if not os.path.exists(local_zip):
        if _url_exists(zip_url):
            return f"[MISSING]  {filename} -> {zip_url}"
        return None

    if os.path.exists(checksum_local):
        try:
            with open(checksum_local) as f:
                expected = f.read().strip().split()[0]
            if expected and sha256_file(local_zip) != expected:
                return f"[CORRUPT]  {filename}"
            return None
        except Exception:
            pass

    return f"[NOCHECKSUM]  {filename}"


def _url_exists(url):
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=10):
            return True
    except Exception:
        return False


def verify_files(tasks, max_workers):
    ok = bad = missing = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(verify_one, t): t for t in tasks}
        with tqdm(total=len(futures), desc="Verifying", unit="files", mininterval=1) as pbar:
            for f in as_completed(futures):
                r = f.result()
                if r == "ok":
                    ok += 1
                elif r == "bad":
                    bad += 1
                elif r == "missing":
                    missing += 1
                pbar.update(1)
    logger.info("Verify done: %d OK, %d CORRUPT, %d MISSING", ok, bad, missing)


def verify_one(task):
    symbol, date_str, asset_class, timeperiod, data_type, freq, base_dir, _ = task

    filename = build_filename(symbol, data_type, freq, date_str)
    rel_dir = build_rel_dir(asset_class, timeperiod, data_type, symbol, freq)
    local_zip = os.path.join(base_dir, rel_dir, filename)
    checksum_local = os.path.join(base_dir, rel_dir, f"{filename}.CHECKSUM")

    if not os.path.exists(local_zip):
        return "missing"
    if not os.path.exists(checksum_local):
        return "missing"

    try:
        with open(checksum_local) as f:
            expected = f.read().strip().split()[0]
        if not expected:
            return "missing"
        if sha256_file(local_zip) == expected:
            return "ok"
        logger.warning("CORRUPT: %s", filename)
        return "bad"
    except Exception:
        return "missing"


def gen_date_range(start, end, timeperiod):
    dates = []
    cur = start
    while cur <= end:
        dates.append(cur)
        if timeperiod == "monthly":
            cur += relativedelta(months=1)
        else:
            cur += timedelta(days=1)
    return dates


def main():
    args = parse_args()
    symbols = load_symbols(args.universe)

    today_utc = datetime.now(timezone.utc).date()

    if args.days is not None:
        timeperiod = "daily"
        end_date = date.fromisoformat(args.end_date) if args.end_date else today_utc - timedelta(days=1)
        start_date = today_utc - timedelta(days=args.days)
    elif args.start_date and "-" in args.start_date:
        parts = args.start_date.split("-")
        if len(parts) == 2:
            timeperiod = "monthly"
            start_date = datetime.strptime(args.start_date, "%Y-%m").date()
        else:
            timeperiod = "daily"
            start_date = date.fromisoformat(args.start_date)
        end_date = date.fromisoformat(args.end_date) if args.end_date else today_utc - timedelta(days=1)
    else:
        timeperiod = "monthly"
        start_date = date(2017, 1, 1)
        end_date = today_utc - timedelta(days=1)

    if start_date < date(2017, 1, 1):
        start_date = date(2017, 1, 1)

    dates = gen_date_range(start_date, end_date, timeperiod)
    fmt = "%Y-%m-%d" if timeperiod == "daily" else "%Y-%m"
    date_strs = [d.strftime(fmt) for d in dates]

    logger.info(
        "%s %s %s freq=%s from %s to %s for %d symbols",
        args.asset_class, timeperiod, args.type,
        args.freq, date_strs[0], date_strs[-1], len(symbols),
    )

    tasks = [
        (s, ds, args.asset_class, timeperiod, args.type,
         args.freq, args.base_dir, args.force)
        for s in symbols for ds in date_strs
    ]

    if args.verify:
        verify_files(tasks, args.max_workers)
        return

    if args.dry_run:
        dry_run_check(tasks, args.max_workers)
        return

    downloaded = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(process_one, t): t for t in tasks}
        with tqdm(total=len(futures), desc="Processing", unit="files", mininterval=1) as pbar:
            for f in as_completed(futures):
                r = f.result()
                if r:
                    downloaded.append(r)
                pbar.update(1)

    logger.info("Done. Downloaded %d new files.", len(downloaded))


if __name__ == "__main__":
    main()
