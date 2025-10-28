import argparse

import sys

# Ensure project root is on sys.path so we can import the top-level `data` package
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use absolute imports from the top-level `data` package (works when run as a script)
from data import POGOHDataFetcher, ClosureDataFetcher, WeatherDataFetcher


def fetch_pogoh_raw_data(start_date, end_date, chunk_size=50_000, output_file_path=None, batch_size=6, delay=1.0):
    fetcher = POGOHDataFetcher(chunk_size=chunk_size)
    fetcher.fetch_data(
        start_date=start_date,
        end_date=end_date,
        output_file_path=output_file_path,
        batch_size=batch_size,
        delay=delay,
    )

def fetch_closure_data():
    pass


def fetch_weather_data():
    pass


def main():
    parser = argparse.ArgumentParser(description="Fetch raw data.")
    parser.add_argument(
        "--d_name",
        type=str,
        choices=["pogoh", "closure", "weather"],
        required=True,
        help="Name of raw data to fetch.",
    )

    parser.add_argument("--start_month", type=str, help="Start month for pogoh (e.g. 2023-01).")
    parser.add_argument("--end_month", type=str, help="End month for pogoh (e.g. 2023-03).")
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Optional output file path for pogoh raw data.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=6,
        help="Optional batch size for pogoh fetching (default: 6).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Optional delay (seconds) between pogoh batches (default: 1.0).",
    )

    args = parser.parse_args()

    if args.d_name == "pogoh":
        if not args.start_month or not args.end_month:
            parser.error("--start_month and --end_month are required when --d_name pogoh")
        fetch_pogoh_raw_data(
            start_date=args.start_month,
            end_date=args.end_month,
            output_file_path=args.output_file,
            batch_size=args.batch_size,
            delay=args.delay,
        )

    elif args.d_name == "closure":
        fetch_closure_data()

    elif args.d_name == "weather":
        fetch_weather_data()

    else:
        raise ValueError("Invalid data type specified.")


if __name__ == "__main__":
    main()
