import os
from pathlib import Path
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List
import time


class WeatherDataFetcher:
    """
    A class to fetch weather data from Open-Meteo API for use as exogenous features.

    Example Usage
    -------------
    >>> fetcher = WeatherDataFetcher(
    ...     latitude=40.4406,  # Pittsburgh coordinates
    ...     longitude=-79.9959
    ... )
    >>> df = fetcher.fetch_data(
    ...     start_date="2023-10-01",
    ...     end_date="2024-09-30",
    ...     output_file_path="weather_data.csv"
    ... )
    >>> print(f"Downloaded {len(df)} days of weather data")
    >>> print(df.head())
    """

    # Open-Meteo API endpoints
    HISTORICAL_API = "https://archive-api.open-meteo.com/v1/archive"
    FORECAST_API = "https://api.open-meteo.com/v1/forecast"

    def __init__(
        self,
        latitude: float = 40.4406,
        longitude: float = -79.9959,
        timezone: str = "America/New_York",
    ):
        """
        Initialize the weather data fetcher.

        Parameters
        ----------
        latitude : float, optional
            Latitude of the location, by default 40.4406 (Pittsburgh)
        longitude : float, optional
            Longitude of the location, by default -79.9959 (Pittsburgh)
        timezone : str, optional
            Timezone for the data, by default "America/New_York"
        """
        self.latitude = latitude
        self.longitude = longitude
        self.timezone = timezone

        # Define weather variables to fetch
        # Daily aggregates
        self.daily_variables = [
            "temperature_2m_max",
            "temperature_2m_min",
            "temperature_2m_mean",
            "precipitation_sum",
            "precipitation_hours",
            "wind_speed_10m_max",
            "wind_gusts_10m_max",
            "wind_direction_10m_dominant",
        ]

        # Hourly variables (optional, for more detailed analysis)
        self.hourly_variables = [
            "temperature_2m",
            "precipitation",
            "wind_speed_10m",
            "wind_direction_10m",
        ]

    def _fetch_weather_chunk(
        self,
        start_date: str,
        end_date: str,
        hourly: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch weather data for a specific date range.

        Parameters
        ----------
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str
            End date in 'YYYY-MM-DD' format
        hourly : bool, optional
            Whether to fetch hourly data (vs daily), by default False

        Returns
        -------
        pd.DataFrame
            DataFrame containing weather data
        """
        # Determine which API to use based on date
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        today = datetime.now().date()

        # Open-Meteo historical data goes up to ~5 days ago
        # Use forecast API for recent data
        is_historical = start_dt.date() < (today - timedelta(days=5))

        api_url = self.HISTORICAL_API if is_historical else self.FORECAST_API

        # Build parameters
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "start_date": start_date,
            "end_date": end_date,
            "timezone": self.timezone,
        }

        if hourly:
            params["hourly"] = ",".join(self.hourly_variables)
        else:
            params["daily"] = ",".join(self.daily_variables)

        try:
            print(f"  Fetching weather data from {start_date} to {end_date}...")
            response = requests.get(api_url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            # Parse the response
            if hourly:
                time_key = "hourly"
                time_col = "time"
            else:
                time_key = "daily"
                time_col = "time"

            if time_key not in data:
                raise ValueError(f"No {time_key} data in response")

            # Convert to DataFrame
            weather_data = data[time_key]
            df = pd.DataFrame(weather_data)

            # Rename time column to date
            df = df.rename(columns={time_col: "date"})

            # Convert date to datetime
            df["date"] = pd.to_datetime(df["date"])

            print(f"  ✓ Fetched {len(df)} records")

            return df

        except requests.exceptions.RequestException as e:
            raise Exception(f"Error fetching weather data: {e}")
        except (KeyError, ValueError) as e:
            raise Exception(f"Error parsing weather data: {e}")

    def _split_date_range(
        self, start_date: str, end_date: str, chunk_months: int = 12
    ) -> List[tuple[str, str]]:
        """
        Split a date range into smaller chunks for API requests.

        Open-Meteo API works best with requests <= 1 year.

        Parameters
        ----------
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str
            End date in 'YYYY-MM-DD' format
        chunk_months : int, optional
            Number of months per chunk, by default 12

        Returns
        -------
        List[tuple[str, str]]
            List of (start_date, end_date) tuples
        """
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        chunks = []
        current_start = start_dt

        while current_start <= end_dt:
            # Calculate chunk end (either chunk_months ahead or final end_date)
            chunk_end = current_start + timedelta(days=chunk_months * 30)
            if chunk_end > end_dt:
                chunk_end = end_dt

            chunks.append(
                (
                    current_start.strftime("%Y-%m-%d"),
                    chunk_end.strftime("%Y-%m-%d"),
                )
            )

            # Move to next chunk
            current_start = chunk_end + timedelta(days=1)

        return chunks

    def fetch_data(
        self,
        start_date: str,
        end_date: str,
        output_file_path: Optional[str] = None,
        hourly: bool = False,
        delay: float = 0.5,
    ) -> pd.DataFrame:
        """
        Fetch weather data for a date range.

        Parameters
        ----------
        start_date : str
            Start date in 'YYYY-MM-DD' or 'YYYY-MM' format
        end_date : str
            End date in 'YYYY-MM-DD' or 'YYYY-MM' format
        output_file_path : Optional[str], optional
            Path to save the data as CSV, by default None
        hourly : bool, optional
            Whether to fetch hourly data (vs daily), by default False
        delay : float, optional
            Delay between API requests in seconds, by default 0.5

        Returns
        -------
        pd.DataFrame
            DataFrame containing weather data
        """
        # Parse and validate dates
        try:
            # Handle YYYY-MM format by converting to first/last day of month
            if len(start_date) == 7:  # YYYY-MM
                start_dt = datetime.strptime(start_date, "%Y-%m")
                start_date = start_dt.strftime("%Y-%m-01")
            else:
                datetime.strptime(start_date, "%Y-%m-%d")

            if len(end_date) == 7:  # YYYY-MM
                end_dt = datetime.strptime(end_date, "%Y-%m")
                # Get last day of month
                next_month = end_dt + timedelta(days=32)
                last_day = next_month.replace(day=1) - timedelta(days=1)
                end_date = last_day.strftime("%Y-%m-%d")
            else:
                datetime.strptime(end_date, "%Y-%m-%d")

        except ValueError as e:
            raise ValueError(f"Invalid date format. Use 'YYYY-MM-DD' or 'YYYY-MM': {e}")

        print(f"Fetching weather data from {start_date} to {end_date}")
        print(f"Location: ({self.latitude}, {self.longitude})")
        print(f"Granularity: {'Hourly' if hourly else 'Daily'}")
        print("=" * 70)

        # Split into chunks (Open-Meteo works best with ~1 year chunks)
        chunks = self._split_date_range(start_date, end_date, chunk_months=12)
        print(f"Split into {len(chunks)} request(s)")
        print("-" * 70)

        all_data = []

        # Fetch each chunk
        for idx, (chunk_start, chunk_end) in enumerate(chunks, 1):
            print(f"\n[{idx}/{len(chunks)}] Processing {chunk_start} to {chunk_end}")

            try:
                df_chunk = self._fetch_weather_chunk(
                    start_date=chunk_start,
                    end_date=chunk_end,
                    hourly=hourly,
                )

                all_data.append(df_chunk)

                # Be respectful to the API
                if idx < len(chunks):
                    time.sleep(delay)

            except Exception as e:
                print(f"  ✗ Failed: {e}")
                continue

        # Combine all chunks
        if not all_data:
            print("\nNo data was successfully downloaded.")
            return pd.DataFrame()

        print("\n" + "-" * 70)
        print("Combining data...")
        df_combined = pd.concat(all_data, ignore_index=True)

        # Sort by date
        df_combined = df_combined.sort_values("date").reset_index(drop=True)

        # Save to CSV if path provided
        if output_file_path is None:
            output_dir = Path.cwd() / f"processed_data/raw"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file_path = (
                output_dir / f"weather_data_{start_date}_to_{end_date}.csv"
            )
        else:
            if os.path.exists(output_file_path):
                print(f"Warning: Overwriting existing file at {output_file_path}")
                os.remove(output_file_path)
            else:
                output_file_path = Path(output_file_path)
                output_file_path.parent.mkdir(parents=True, exist_ok=True)
        if output_file_path:
            df_combined.to_csv(output_file_path, index=False)
            print(f"Saved {len(df_combined)} records to {output_file_path}")

        # Print summary statistics
        print("\n" + "=" * 70)
        print("WEATHER DATA SUMMARY")
        print("=" * 70)
        print(f"Total records: {len(df_combined):,}")
        print(f"Date range: {df_combined['date'].min()} to {df_combined['date'].max()}")
        print(f"Columns: {', '.join(df_combined.columns.tolist())}")

        if not hourly:
            print("\nTemperature Statistics:")
            print(f"  Mean temp: {df_combined['temperature_2m_mean'].mean():.1f}°C")
            print(f"  Max temp: {df_combined['temperature_2m_max'].max():.1f}°C")
            print(f"  Min temp: {df_combined['temperature_2m_min'].min():.1f}°C")

            print("\nPrecipitation Statistics:")
            print(
                f"  Total precipitation: {df_combined['precipitation_sum'].sum():.1f} mm"
            )
            print(f"  Days with rain: {(df_combined['precipitation_sum'] > 0).sum()}")

            print("\nWind Statistics:")
            print(
                f"  Max wind speed: {df_combined['wind_speed_10m_max'].max():.1f} km/h"
            )
            print(
                f"  Mean wind speed: {df_combined['wind_speed_10m_max'].mean():.1f} km/h"
            )

        return df_combined
    

__all__ = ["WeatherDataFetcher"]
