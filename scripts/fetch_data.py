import argparse

import sys

# Ensure project root is on sys.path so we can import the top-level `data` package
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use absolute imports from the top-level `data` package (works when run as a script)
from data import POGOHDataFetcher, ClosureDataFetcher, WeatherDataFetcher


def fetch_closure_data():
    pass


def fetch_pogoh_raw_data(start_date, end_date, chunk_size=50_000, output_file_path=None, batch_size=6, delay=1.0):
    fetcher = POGOHDataFetcher(chunk_size=chunk_size)
    fetcher.fetch_data(
        start_date=start_date,
        end_date=end_date,
        output_file_path=output_file_path,
        batch_size=batch_size,
        delay=delay,
    )


def fetch_weather_data(
    start_date,
    end_date,
    latitude=40.4406,
    longitude=-79.9959,
    timezone="America/New_York",
    hourly=False,
    output_file_path=None,
    delay=1.0,
):
    fetcher = WeatherDataFetcher(
        latitude=latitude,
        longitude=longitude,
        timezone=timezone
    )
    fetcher.fetch_data(
        start_date=start_date,
        end_date=end_date,
        output_file_path=output_file_path,
        hourly=hourly,
        delay=delay,
    )


def main():
    parser = argparse.ArgumentParser(description="Fetch raw data.")
    parser.add_argument(
        "--d_name",
        type=str,
        choices=["pogoh", "closure", "weather"],
        required=True,
        help="Name of raw data to fetch.",
    )
    # time args
    parser.add_argument("--start_date", type=str, help="Start date for pogoh (e.g. 2023-01-01).")
    parser.add_argument("--end_date", type=str, help="End date for pogoh (e.g. 2023-03-01).")
    
    # file path args
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

    # Geo args for weather
    parser.add_argument(
        "--latitude",
        type=float,
        default=40.4406,
        help="Latitude for weather data (default: 40.4406). for pittsburgh",
    )
    parser.add_argument(
        "--longitude",
        type=float,
        default=-79.9959,
        help="Longitude for weather data (default: -79.9959). for pittsburgh",
    )

    parser.add_argument(
        "--timezone",
        type=str,
        default="America/New_York",
        help="Timezone for weather data (default: America/New_York).",
    )

    parser.add_argument(
        "--hourly",
        type=bool,
        default=False,
        help="Fetch hourly weather data (default: False).",
    )


    args = parser.parse_args()

    if args.d_name == "pogoh":
        if not args.start_date or not args.end_date:
            parser.error("--start_date and --end_date are required when --d_name pogoh")
        fetch_pogoh_raw_data(
            start_date=args.start_date,
            end_date=args.end_date,
            output_file_path=args.output_file,
            batch_size=args.batch_size,
            delay=args.delay,
        )

    elif args.d_name == "closure":
        fetch_closure_data()

    elif args.d_name == "weather":
        if not args.start_date or not args.end_date:
            parser.error("--start_date and --end_date are required when --d_name weather")
        fetch_weather_data(
            start_date=args.start_date,
            end_date=args.end_date,
            latitude=args.latitude,
            longitude=args.longitude,
            timezone=args.timezone,
            hourly=args.hourly,
            output_file_path=args.output_file,
            delay=args.delay,
        )

    else:
        raise ValueError("Invalid data type specified.")


if __name__ == "__main__":
    main()
