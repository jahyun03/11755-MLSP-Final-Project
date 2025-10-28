import os
import requests
import pandas as pd
import io
import time
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import Optional, Iterator, Union
import gc

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class ClosureDataFetcher:
    """
    A class to fetch and process Street Closure data from WPRDC API.

    This data contains DOMI Street Closure Permit data since May 2020.

    Each row represents roadway segment within a closure permit.

    Example Usage
    -------------
    >>> fetcher = ClosureDataFetcher(chunk_size=50_000)
    >>> stats = fetcher.fetch_data(
    ...    start_date="2024-01-01",
    ...    end_date="2024-09-30",
    ...    date_field="effective_date",
    ...    output_file_path=None,
    ...    delay=1.0,
    ... )
    >>> print(f"Downloaded {stats['row_count']:,} rows")
    """

    def __init__(self, chunk_size: int = 50_000):
        """
        Initialize data fetcher for street closure

        Parameters
        ----------
        chunk_size : int, optional
            Number of rows to process at a time, by default 50_000
        """
        self.base_url = "https://data.wprdc.org"
        self.api_url = f"{self.base_url}/api/3/action"
        self.package_id = "street-closures"
        self.chunk_size = chunk_size
        self.date_fields = [
            "effective_date",
            "expiration_date", 
            "issue_date",
            "application_date"
        ] 

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

    def _list_closure_resources(self) -> list[dict]:
        """
        Fetches all street closure data. 

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

    def _get_latest_resource(self, resources: list[dict]) -> Optional[dict]:
        """
        Brings the most recent resource (hourly update).

        Parameters
        ----------
        resources : list[dict]
            List of resource metadata dictionaries

        Returns
        -------
        Optional[dict]
            The most recent resource or None if not found
        """
        if not resources:
            return None
            

        valid_resources = []
        for resource in resources:
            if (resource.get("format", "").upper() == "CSV"
                and resource.get("url")):
                
                last_modified = resource.get("last_modified") or resource.get("created")
                if last_modified:
                    try:
                        mod_date = datetime.fromisoformat(last_modified.replace('Z', '+00:00'))
                        valid_resources.append({
                            "resource": resource,
                            "modified_date": mod_date
                        })
                    except: # if data parsing fails
                        valid_resources.append({
                            "resource": resource,
                            "modified_date": datetime.now()
                        })
        
        if not valid_resources:
            return resources[0] if resources else None
            
        latest = max(valid_resources, key=lambda x: x["modified_date"])
        return latest["resource"]

    def _stream_download_resource(self, resource: dict) -> Iterator[pd.DataFrame]:
        url = resource.get("url")
        
        if not url:
            raise Exception(f"No URL found for resource: {resource.get('name')}")

        print(f"  Downloading: {resource.get('name')}")

        try:
            response = requests.get(url, timeout=300, stream=True)
            response.raise_for_status()

            # Determine file type
            is_excel = (
                url.endswith(".xlsx")
                or "excel" in response.headers.get("content-type", "").lower()
            )

            if is_excel:
                # For Excel files
                content = io.BytesIO(response.content)
                try:
                    for chunk in pd.read_excel(content, chunksize=self.chunk_size):
                        yield chunk
                except:
                    content.seek(0)
                    yield pd.read_excel(content)

            else:  # CSV
                content = io.StringIO(response.text)
                try:
                    for chunk in pd.read_csv(
                        content,
                        chunksize=self.chunk_size,
                        on_bad_lines='skip',
                        encoding='utf-8',
                        encoding_errors='replace',
                        low_memory=False
                    ):
                        yield chunk
                except:
                    content.seek(0)
                    df = pd.read_csv(
                        content,
                        on_bad_lines='skip',
                        encoding='utf-8',
                        encoding_errors='replace',
                        low_memory=False
                    )
                    
                    for i in range(0, len(df), self.chunk_size):
                        yield df.iloc[i:i + self.chunk_size].copy()

        except requests.exceptions.RequestException as e:
            raise Exception(f"Error downloading resource: {e}")

    def _filter_data_by_date(
        self, 
        df: pd.DataFrame, 
        start_date: datetime, 
        end_date: datetime,
        date_field: str = "effective_date"
    ) -> pd.DataFrame:
        """
        Filter DataFrame by date.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame
        start_date : datetime
            Start date for filtering
        end_date : datetime
            End date for filtering
        date_field : str, optional
            Date field to filter on, by default "effective_date"

        Returns
        -------
        pd.DataFrame
            Filtered DataFrame
        """
        if date_field not in df.columns:
            print(f"  Warning: Date field '{date_field}' not found in data")
            print(f"  Available columns: {list(df.columns)[:10]}...")  # Show first 10 columns
            return df
            
        try:
            df[date_field] = pd.to_datetime(df[date_field], errors='coerce')
            
            mask = (
                (df[date_field] >= start_date) & 
                (df[date_field] <= end_date)
            )
            
            filtered_df = df[mask].copy()
            
            print(f"  Filtered from {len(df):,} to {len(filtered_df):,} rows")
            
            return filtered_df
            
        except Exception as e:
            print(f"  Warning: Error filtering by date: {e}")
            return df

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
        date_field: str = "effective_date",
        delay: float = 1.0,
        monitor_memory: bool = True,
    ) -> dict:
        """
        Fetch street closure data using streaming approach to minimize memory usage.

        Parameters
        ----------
        start_date : str
            Start date in format 'YYYY-MM-DD'
        end_date : str
            End date in format 'YYYY-MM-DD'
        output_file_path : str
            Output CSV file path
        date_field : str, optional
            Date field to filter on, by default "effective_date"
        delay : float, optional
            Delay in seconds between operations, by default 1.0
        monitor_memory : bool, optional
            Whether to monitor and report memory usage, by default True

        Returns
        -------
        dict
            Dictionary with statistics (row_count, file_count)
        """
        # Parse dates
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError as e:
            raise ValueError(f"Invalid date format. Use 'YYYY-MM-DD': {e}")

        if monitor_memory:
            initial_memory = self._get_memory_usage_mb()
            if initial_memory > 0:
                print(f"Initial memory usage: {initial_memory:.2f} MB")

        # Get resources
        all_resources = self._list_closure_resources()
        latest_resource = self._get_latest_resource(all_resources)

        if not latest_resource:
            print("No suitable resources found")
            return {"row_count": 0, "file_count": 0}

        print(f"Found latest resource: {latest_resource.get('name')}")
        print(f"Last modified: {latest_resource.get('last_modified', 'Unknown')}")

        # Remove existing output file if it exists
        if os.path.exists(output_file_path):
            os.remove(output_file_path)

        # Statistics
        total_rows = 0
        chunk_count = 0
        is_first_chunk = True

        try:
            print(f"\nProcessing closures from {start_date} to {end_date}...")
            
            for chunk in self._stream_download_resource(latest_resource):
                # Filter chunk by date
                filtered_chunk = self._filter_data_by_date(
                    chunk, start_dt, end_dt, date_field
                )
                
                if len(filtered_chunk) > 0:
                    # Add fetch metadata
                    filtered_chunk["fetch_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    filtered_chunk["date_filter_field"] = date_field
                    
                    # Write to CSV
                    write_header = is_first_chunk
                    self._append_to_csv(
                        filtered_chunk, output_file_path, write_header=write_header
                    )
                    
                    total_rows += len(filtered_chunk)
                    chunk_count += 1
                    is_first_chunk = False
                    
                    print(f"  Chunk {chunk_count}: {len(filtered_chunk):,} rows")

                # Clear chunk from memory
                del chunk, filtered_chunk
                gc.collect()

            print(f"\n Processing complete")
            print(f"  Total rows written: {total_rows:,}")
            print(f"  Chunks processed: {chunk_count}")

            if monitor_memory:
                final_memory = self._get_memory_usage_mb()
                if final_memory > 0 and initial_memory > 0:
                    print(f"  Memory change: {final_memory - initial_memory:+.2f} MB")

            return {"row_count": total_rows, "file_count": 1}

        except Exception as e:
            print(f"✗ Error processing resource: {e}")
            return {"row_count": 0, "file_count": 0}

    def fetch_data(
        self,
        start_date: str,
        end_date: str,
        date_field: str = "effective_date",
        output_file_path: Optional[str] = None,
        delay: float = 1.0,
    ) -> dict:
        """
        Fetch street closure data within a date range.

        Parameters
        ----------
        start_date : str
            The start date in 'YYYY-MM-DD' format
        end_date : str
            The end date in 'YYYY-MM-DD' format
        date_field : str, optional
            Date field to filter on, by default "effective_date"
            Options: "effective_date", "expiration_date", "issue_date", "application_date"
        output_file_path : Optional[str], optional
            Path to save the data as CSV, by default None
        delay : float, optional
            Delay between operations in seconds, by default 1.0

        Returns
        -------
        dict
            A dictionary with statistics: row_count, file_count, output_file
        """
        # Validate date field
        if date_field not in self.date_fields:
            print(f"Warning: '{date_field}' not in expected date fields: {self.date_fields}")
            print("Proceeding anyway in case field exists in data...")

        try: # parse dates
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError as e:
            raise ValueError(f"Invalid date format. Use 'YYYY-MM-DD': {e}")

        print(f"Fetching Street Closure data from {start_date} to {end_date}")
        print(f"Filtering by: {date_field}")
        print("=" * 70)

        if output_file_path is None:
            output_file_dir = Path.cwd() / "processed_data/raw"
            output_file_dir.mkdir(parents=True, exist_ok=True)
            output_file_path = (
                output_file_dir / 
                f"closure_data_{start_dt.strftime('%Y%m%d')}_to_{end_dt.strftime('%Y%m%d')}.csv"
            )
        else:
            output_file_path = Path(output_file_path)
            output_file_path.parent.mkdir(parents=True, exist_ok=True)

        # fetch the data
        stats = self.fetch_data_streaming(
            start_date=start_date,
            end_date=end_date,
            output_file_path=str(output_file_path),
            date_field=date_field,
            delay=delay,
            monitor_memory=True,
        )

        # final 
        print("\n" + "=" * 70)
        print("FETCH COMPLETE")
        print("=" * 70)
        print(f"Total rows: {stats['row_count']:,}")
        print(f"Date field used: {date_field}")
        print(f"Output: {output_file_path}")

        if os.path.exists(output_file_path):
            file_size = os.path.getsize(output_file_path) / (1024 * 1024)
            print(f"File size: {file_size:.2f} MB")

        stats["output_file"] = output_file_path
        stats["date_field"] = date_field
        return stats