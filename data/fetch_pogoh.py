import os
import requests
import pandas as pd
import io
import time
import tempfile
from pathlib import Path
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import Optional, Iterator
import gc

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class POGOHDataFetcher:
    """
    A class to fetch and process POGOH data from WPRDC API.

    Example Usage
    -------------
    >>> fetcher = POGOHDataFetcher(chunk_size=50_000)
    >>> stats = fetcher.fetch_data(
    ...    start_date="2024-07",
    ...    end_date="2024-09",
    ...    output_file_path=None,
    ...    batch_size=6,
    ...    delay=1.0,
    ...   )
    >>> print(f"Downloaded {stats['row_count']:,} rows in {stats['file_count']} files")
    """

    def __init__(self, chunk_size: int = 50_000):
        """
        Initialize the POGOH data fetcher.

        Parameters
        ----------
        chunk_size : int, optional
            Number of rows to process at a time, by default 50_000
        """
        self.base_url = "https://data.wprdc.org"
        self.api_url = f"{self.base_url}/api/3/action"
        self.package_id = "pogoh-trip-data"
        self.chunk_size = chunk_size

    def _get_memory_usage_mb(self) -> float:
        """
        Get current memory usage in MB.

        Returns
        -------
        float
            Current memory usage in megabytes, or 0.0 if psutil unavailable
        """
        if not PSUTIL_AVAILABLE:
            return 0.0
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0

    def _list_pogoh_resources(self) -> list[dict]:
        """
        Fetch all monthly files for POGOH trip data.

        Returns
        -------
        list[dict]
            A list of dictionaries containing metadata for each file
        """
        url = f"{self.api_url}/package_show"
        params = {"id": self.package_id}
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            if data.get("success"):
                return data["result"]["resources"]
            else:
                raise Exception(f"API returned error: {data.get('error')}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error fetching resources: {e}")

    def _get_resource_date(self, resource: dict) -> Optional[datetime]:
        """
        Extracts the date from the resource name.

        Parameters
        ----------
        resource : dict
            A dictionary containing metadata for a file.

        Returns
        -------
        Optional[datetime]
            The extracted date or None if not found.
        """
        name = resource.get("name", "").lower()
        description = resource.get("description", "").lower()

        months = {
            "january": 1,
            "february": 2,
            "march": 3,
            "april": 4,
            "may": 5,
            "june": 6,
            "july": 7,
            "august": 8,
            "september": 9,
            "october": 10,
            "november": 11,
            "december": 12,
        }
        for month_name, month_num in months.items():
            if month_name in name or month_name in description:
                # Try to find year
                import re

                year_match = re.search(r"20\d{2}", name + description)
                if year_match:
                    year = int(year_match.group())
                    return datetime(year, month_num, 1)

        return None

    def _filter_resources_by_date(
        self, resources: list[dict], start_date: datetime, end_date: datetime
    ) -> list[dict]:
        """
        Filters resources based on a date range.

        Parameters
        ----------
        resources : list[dict]
            A list of dictionaries containing metadata for each file.
        start_date : datetime
            The start date for filtering.
        end_date : datetime
            The end date for filtering.

        Returns
        -------
        list[dict]
            A list of dictionaries containing metadata for files within the date range.
        """
        filtered = []
        for resource in resources:
            resource_date = self._get_resource_date(resource)

            if resource_date and start_date <= resource_date <= end_date:
                filtered.append({"resource": resource, "date": resource_date})

        filtered.sort(key=lambda x: x["date"])

        return filtered

    def _stream_download_to_chunks(self, resource: dict) -> Iterator[pd.DataFrame]:
        """
        Download a resource and yield it in chunks to minimize memory usage.

        Parameters
        ----------
        resource : dict
            A dictionary containing metadata for a file.

        Yields
        ------
        pd.DataFrame
            DataFrame chunks from the downloaded resource
        """
        url = resource.get("url")

        if not url:
            raise Exception(f"No URL found for resource: {resource.get('name')}")

        print(f"  Downloading: {resource.get('name')}")

        try:
            response = requests.get(url, timeout=120, stream=True)
            response.raise_for_status()

            # Determine file type
            is_excel = (
                url.endswith(".xlsx")
                or "excel" in response.headers.get("content-type", "").lower()
            )

            if is_excel:
                # For Excel files, read in chunks if possible
                content = io.BytesIO(response.content)

                try:
                    for chunk in pd.read_excel(content, chunksize=self.chunk_size):
                        yield chunk
                except:
                    # If chunking not supported, read whole file
                    content.seek(0)
                    yield pd.read_excel(content)

            else:  # CSV
                # Stream CSV directly
                content = io.StringIO(response.text)

                try:
                    for chunk in pd.read_csv(
                        content, 
                        chunksize=self.chunk_size,
                        on_bad_lines='skip',
                        encoding='utf-8',
                        encoding_errors='replace'
                    ):
                        yield chunk
                except:
                    content.seek(0)
                    yield pd.read_csv(
                        content,
                        on_bad_lines='skip',
                        encoding='utf-8',
                        encoding_errors='replace'
                    )

        except requests.exceptions.RequestException as e:
            raise Exception(f"Error downloading resource: {e}")

    def _append_to_csv(
        self, df: pd.DataFrame, output_file_path: str, write_header: bool = False
    ) -> None:
        """
        Append a dataframe to CSV file.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to append
        output_file_path : str
            Output CSV file path
        write_header : bool, optional
            Whether to write the header row, by default False
        """
        mode = "w" if write_header else "a"
        df.to_csv(output_file_path, mode=mode, index=False, header=write_header)

    def fetch_data_streaming(
        self,
        start_date: str,
        end_date: str,
        output_file_path: str,
        delay: float = 1.0,
        monitor_memory: bool = True,
    ) -> dict:
        """
        Fetch POGOH trip data using streaming approach to minimize memory usage.
        Data is written directly to CSV as it's downloaded, one month at a time.

        Parameters
        ----------
        start_date : str
            Start date in format 'YYYY-MM-DD' or 'YYYY-MM'
        end_date : str
            End date in format 'YYYY-MM-DD' or 'YYYY-MM'
        output_file_path : str
            Output CSV file path
        delay : float, optional
            Delay in seconds between downloads, by default 1.0
        monitor_memory : bool, optional
            Whether to monitor and report memory usage, by default True

        Returns
        -------
        dict
            Dictionary with statistics (row_count, file_count)
        """
        # Parse dates
        try:
            if len(start_date) == 7:
                start_dt = datetime.strptime(start_date, "%Y-%m")
            else:
                start_dt = datetime.strptime(start_date, "%Y-%m-%d")

            if len(end_date) == 7:
                end_dt = datetime.strptime(end_date, "%Y-%m")
            else:
                end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError as e:
            raise ValueError(f"Invalid date format. Use 'YYYY-MM-DD' or 'YYYY-MM': {e}")

        if monitor_memory:
            initial_memory = self._get_memory_usage_mb()
            if initial_memory > 0:
                print(f"Initial memory usage: {initial_memory:.2f} MB")

        # Get and filter resources
        all_resources = self._list_pogoh_resources()
        filtered = self._filter_resources_by_date(all_resources, start_dt, end_dt)

        print(f"Found {len(filtered)} resources within date range")

        if not filtered:
            return {"row_count": 0, "file_count": 0}

        # Remove existing output file if it exists
        if os.path.exists(output_file_path):
            os.remove(output_file_path)

        # Statistics
        total_rows = 0
        successful_downloads = 0
        is_first_file = True

        # Process each resource
        for idx, item in enumerate(filtered, 1):
            resource = item["resource"]
            date = item["date"]

            print(f"\n[{idx}/{len(filtered)}] Processing {date.strftime('%B %Y')}...")

            try:
                chunk_count = 0
                file_rows = 0

                for chunk in self._stream_download_to_chunks(resource):
                    # Add source month column
                    chunk["source_month"] = date.strftime("%Y-%m")

                    # Write to CSV
                    write_header = is_first_file and chunk_count == 0
                    self._append_to_csv(
                        chunk, output_file_path, write_header=write_header
                    )

                    file_rows += len(chunk)
                    chunk_count += 1

                    # Clear chunk from memory
                    del chunk

                    is_first_file = False

                # Force garbage collection
                gc.collect()

                total_rows += file_rows
                successful_downloads += 1

                print(f"  ✓ Wrote {file_rows:,} rows ({chunk_count} chunks)")

                if monitor_memory:
                    current_memory = self._get_memory_usage_mb()
                    if current_memory > 0:
                        print(f"  Memory: {current_memory:.2f} MB")

                # Be respectful to the server
                if idx < len(filtered):
                    time.sleep(delay)

            except Exception as e:
                print(f"  ✗ Failed: {e}")
                continue

        print(f"\nTotal rows written: {total_rows:,}")
        print(f"Successful downloads: {successful_downloads}/{len(filtered)}")

        if monitor_memory:
            final_memory = self._get_memory_usage_mb()
            if final_memory > 0 and initial_memory > 0:
                print(f"Memory change: {final_memory - initial_memory:+.2f} MB")

        return {"row_count": total_rows, "file_count": successful_downloads}

    def fetch_data(
        self,
        start_date: str,
        end_date: str,
        output_file_path: Optional[str] = None,
        batch_size: int = 6,
        delay: float = 1.0,
    ) -> dict:
        """
        Fetch POGOH data within a date range using batched approach.

        Parameters
        ----------
        start_date : str
            The start date in 'YYYY-MM-DD' or 'YYYY-MM' format.
        end_date : str
            The end date in 'YYYY-MM-DD' or 'YYYY-MM' format.
        output_file_path : Optional[str], optional
            Path to save the combined data as CSV, by default None.
            If None, data is returned but not saved.
        batch_size : int, optional
            Number of months to download in each batch, by default 6.
        delay : float, optional
            Delay between downloads in seconds, by default 1.0.

        Returns
        -------
        dict
            A dictionary with statistics: row_count, file_count, output_file
        """
        try:
            if len(start_date) == 7:  # YYYY-MM format
                start_dt = datetime.strptime(start_date, "%Y-%m")
            else:
                start_dt = datetime.strptime(start_date, "%Y-%m-%d")

            if len(end_date) == 7:  # YYYY-MM format
                end_dt = datetime.strptime(end_date, "%Y-%m")
            else:
                end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError as e:
            raise ValueError(f"Invalid date format. Use 'YYYY-MM-DD' or 'YYYY-MM': {e}")

        print(f"Fetching POGOH data from {start_date} to {end_date}")
        print(f"Using batch size: {batch_size} months")
        print("=" * 70)

        # If no output path specified, use a default
        if output_file_path is None:
            output_file_dir = Path.cwd() / f"processed_data/raw"
            output_file_dir.mkdir(parents=True, exist_ok=True)
            output_file_path = (
                output_file_dir
                / f"pogoh_data_{start_dt.strftime('%Y%m')}_to_{end_dt.strftime('%Y%m')}.csv"
            )
        else:
            if os.path.exists(output_file_path):
                os.remove(output_file_path)
            else:
                output_file_path = Path(output_file_path)
                output_file_path.parent.mkdir(parents=True, exist_ok=True)

        total_stats = {"row_count": 0, "file_count": 0}
        batch_num = 1
        current_start = start_dt

        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Using temporary directory: {temp_dir}\n")

            while current_start <= end_dt:
                batch_end = current_start + relativedelta(months=batch_size - 1)
                if batch_end > end_dt:
                    batch_end = end_dt

                print(
                    f"\nBATCH {batch_num}: {current_start.strftime('%Y-%m')} to {batch_end.strftime('%Y-%m')}"
                )
                print("-" * 70)

                # Create temporary file path for this batch
                temp_batch_file = os.path.join(temp_dir, f"batch_{batch_num}.csv")

                # Fetch this batch
                stats = self.fetch_data_streaming(
                    start_date=current_start.strftime("%Y-%m"),
                    end_date=batch_end.strftime("%Y-%m"),
                    output_file_path=temp_batch_file,
                    delay=delay,
                    monitor_memory=True,
                )

                # Append to main file if data was downloaded
                if stats["row_count"] > 0:
                    print(f"\nAppending batch to {output_file_path}...")

                    # Read and append in chunks to maintain memory efficiency
                    is_first_batch = batch_num == 1

                    for chunk in pd.read_csv(
                        temp_batch_file, 
                        chunksize=self.chunk_size,
                        on_bad_lines='skip',
                        encoding='utf-8',
                        encoding_errors='replace'
                    ):
                        write_header = is_first_batch and total_stats["row_count"] == 0
                        self._append_to_csv(
                            chunk, output_file_path, write_header=write_header
                        )
                        total_stats["row_count"] += len(chunk)

                    total_stats["file_count"] += stats["file_count"]

                    # Temporary file will be automatically cleaned up
                    gc.collect()

                # Move to next batch
                current_start = batch_end + relativedelta(months=1)
                batch_num += 1

        # final statistics
        print("\n" + "=" * 70)
        print("ALL BATCHES COMPLETE")
        print("=" * 70)
        print(f"Total rows: {total_stats['row_count']:,}")
        print(f"Total files processed: {total_stats['file_count']}")
        print(f"Output: {output_file_path}")

        if os.path.exists(output_file_path):
            file_size = os.path.getsize(output_file_path) / (1024 * 1024)
            print(f"File size: {file_size:.2f} MB")

        total_stats["output_file"] = output_file_path
        return total_stats
