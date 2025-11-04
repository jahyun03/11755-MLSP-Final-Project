import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List
import logging
import holidays


class DataProcessor:
    """
    Process and integrate multiple data sources for POGOH forecasting.
    Simplified version dealing with pogoh and weather. 
    Can be extend later with events, transit, closures
    """
    
    def __init__(self, data_dir: str = "processed_data/raw"):
        """
        Initialize the data processor
        
        Parameters
        ----------
        data_dir : str, optional
            Directory where raw data files are located
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir
        self.processed_dir = self.data_dir / "processed"
        self.output_dir = self.data_dir.parent / "ready"
        
        # Create directories
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # File pattern
        self.file_patterns = {
            'pogoh': 'pogoh_data_*.csv',
            'weather': 'weather_data_*.csv',
            # 'closures': 'closure_data_*.csv',
            # 'events': 'events_data_*.csv',
            # 'transit': 'transit_data_*.csv'
        }
        
        # Data processing configuration
        self.aggregation_config = {
            'lag_windows': [1, 7, 14]
        }
        
        # Temporal Featureese
        self.temporal_features = {
            'day_of_week': True,
            'month': True,
            'is_weekend': True,
            'is_holiday': True,
            'day_of_year': True,
            'season': True,
            'week_of_year': True,

            'month_sin': True,
            'month_cos': True,
            'dow_sin': True,
            'dow_cos': True,
        }
        
        # Weather Feature
        self.weather_features = {
            'temp_range': True,
            'is_rainy': True,
            'is_extreme_weather': True,
            'is_extreme_heat': True,
            'is_extreme_cold': True,
        }
        
        # Date Range
        self.data_range = {
            'start_date': None,
            'end_date': None,
            'common_start': None,
            'common_end': None
        }
        
        self.setup_logging()
    
    def setup_logging(self):
        # setting up logging configuration
        log_dir = self.data_dir.parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'preprocessing.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('DataProcessor')
        self.logger.info(f"DataProcessor initialized with data_dir: {self.data_dir}")
    
    def _find_latest_file(self, pattern: str) -> Optional[Path]:
        """
        Find the most recent file matching the pattern.
        
        Parameters
        ----------
        pattern : str
            Glob pattern to match files
            
        Returns
        -------
        Optional[Path]
            Path to the latest file, or None if not found
        """
        files = list(self.raw_dir.glob(pattern))
        
        if not files:
            self.logger.warning(f"No files found matching pattern: {pattern}")
            return None
        
        latest = max(files, key=lambda f: f.stat().st_mtime) # return newests
        return latest
    
    def load_pogoh_data(
        self, 
        file_path: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load POGOH trip data and perform basic cleaning.
        
        Parameters
        ----------
        file_path : Optional[str]
            Specific file path. If None, finds latest matching pattern.
        start_date : Optional[str]
            Filter start date (YYYY-MM-DD)
        end_date : Optional[str]
            Filter end date (YYYY-MM-DD)
            
        Returns
        -------
        pd.DataFrame
            Trip data with start_time and duration
        """
        self.logger.info("Loading POGOH data...")
        
        # Find file if not specified
        if file_path is None:
            file_path = self._find_latest_file(self.file_patterns['pogoh'])
            if file_path is None:
                raise FileNotFoundError("No POGOH data found.")
        else:
            file_path = Path(file_path)
        
        # Detect date column from sample
        sample = pd.read_csv(file_path, nrows=5)
        date_cols_to_try = [
            'start_time', 'Start Time', 'starttime', 
            'trip_start_time']
        
        date_col = None
        for col in date_cols_to_try:
            if col in sample.columns:
                date_col = col
                break
        
        if date_col is None:
            raise ValueError(
                f"Could not find timestamp column. "
                f"Available: {list(sample.columns)}"
            )
        
        self.logger.info(f"Using timestamp column: '{date_col}'")
        
        # Load data in chunks
        chunks = []   
        for chunk in pd.read_csv(
            file_path,
            chunksize=100_000,
            parse_dates=[date_col],
            low_memory=False
        ):
            chunk = chunk.rename(columns={date_col: 'start_time'})
            chunk = chunk.dropna(subset=['start_time'])
            
            # Filter date ranges
            if start_date:
                chunk = chunk[chunk['start_time'] >= start_date]
            if end_date:
                chunk = chunk[chunk['start_time'] <= end_date]
            
            chunks.append(chunk)
        
        df = pd.concat(chunks, ignore_index=True)
        self.logger.info(f"Loaded {len(df):,} trips")
        return df
    
    def load_weather_data(
        self,
        file_path: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load weather data.
        
        Parameters
        ----------
        file_path : Optional[str]
            Specific file path. If None, finds latest matching pattern.
        start_date : Optional[str]
            Filter start date (YYYY-MM-DD)
        end_date : Optional[str]
            Filter end date (YYYY-MM-DD)
            
        Returns
        -------
        pd.DataFrame
            Weather data with date index
        """
        self.logger.info("Loading weather data...")
        
        if file_path is None:
            file_path = self._find_latest_file(self.file_patterns['weather'])
            if file_path is None:
                raise FileNotFoundError(
                    f"No weather data found."
                )
        else:
            file_path = Path(file_path)
        
        df = pd.read_csv(file_path, parse_dates=['date'])
        
        # Filter date range
        if start_date:
            df = df[df['date'] >= start_date]
        if end_date:
            df = df[df['date'] <= end_date]
        
        self.logger.info(f"Loaded {len(df)} days of weather data")
        
        return df
    
    def aggregate_to_daily(
        self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert trip-level data to daily aggregates.
        
        Parameters
        ----------
        df : pd.DataFrame
            Trip-level data with timestamp column
        date_col : str
            Name of the timestamp column (default: 'start_time')
            
        Returns
        -------
        pd.DataFrame
            Daily aggregated data with columns: date, trip_count, duration_mean, duration_std
        """
        self.logger.info("Aggregating trips to daily level...")
        
        # Extract date
        df['date'] = pd.to_datetime(df['start_time']).dt.date
        
        # Count trips per day
        daily = df.groupby('date').size().reset_index(name = 'trip_count')
        daily['date'] = pd.to_datetime(daily['date'])
        
        self.logger.info(f"Aggregated to {len(daily)} days")
        self.logger.info(f"  Mean trips/day: {daily['trip_count'].mean():.0f}")
        
        return daily
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add temporal features based on configuration.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with 'date' column
            
        Returns
        -------
        pd.DataFrame
            DataFrame with added temporal features
        """
        self.logger.info("Creating temporal features...")
        
        if self.temporal_features.get('day_of_week'):
            df['day_of_week'] = df['date'].dt.dayofweek  # 0=Monday
            # feature_count += 1
        
        if self.temporal_features.get('month'):
            df['month'] = df['date'].dt.month
        
        if self.temporal_features.get('is_weekend'):
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        if self.temporal_features.get('is_holiday'):
            us_holidays = holidays.US()
            df['is_holiday'] = df['date'].apply(
                lambda x: x in us_holidays
            ).astype(int)
        
        if self.temporal_features.get('day_of_year'):
            df['day_of_year'] = df['date'].dt.dayofyear
        
        if self.temporal_features.get('week_of_year'):
            df['week_of_year'] = df['date'].dt.isocalendar().week
        
        # season??

        # Cyclical encoding
        if self.temporal_features.get('month_sin'):
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        
        if self.temporal_features.get('month_cos'):
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        if self.temporal_features.get('dow_sin'):
            df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        
        if self.temporal_features.get('dow_cos'):
            df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        
        return df
    
    def create_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer weather features based on configuration.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with weather columns
            
        Returns
        -------
        pd.DataFrame
            DataFrame with engineered weather features
        """
        self.logger.info("Engineering weather features...")
        
        # Standardize column name
        rename_map = {
            'temperature_2m_max': 'temp_max',
            'temperature_2m_min': 'temp_min',
            'temperature_2m_mean': 'temp_mean',
            'precipitation_sum': 'precip',
            'wind_speed_10m_max': 'wind_speed',
        }
        
        df = df.rename(columns=rename_map)

        # feature_count = 0
    
        if self.weather_features.get('temp_range'):
            if 'temp_max' in df.columns and 'temp_min' in df.columns:
                df['temp_range'] = df['temp_max'] - df['temp_min']
                # feature_count += 1
        
        if self.weather_features.get('is_rainy'):
            if 'precip' in df.columns:
                df['is_rainy'] = (df['precip'] > 1.0).astype(int)
        
        if self.weather_features.get('is_extreme_heat'):
            if 'temp_max' in df.columns:
                df['is_extreme_heat'] = (df['temp_max'] > 32).astype(int)  # >32°C = >90°F
        
        if self.weather_features.get('is_extreme_cold'):
            if 'temp_min' in df.columns:
                df['is_extreme_cold'] = (df['temp_min'] < 0).astype(int)  # Freezing
        
        if self.weather_features.get('is_extreme_weather'):
            df['is_extreme_weather'] = (
                df.get('is_extreme_heat', 0) | 
                df.get('is_extreme_cold', 0) | 
                ((df.get('precip', 0) > 20).astype(int))  # Heavy rain
            ).astype(int)
        
        self.logger.info(f"  Added weather features")
        
        return df
    
    def create_lag_features(
        self, 
        df: pd.DataFrame, 
        target_col: str = 'trip_count'
    ) -> pd.DataFrame:
        """
        Create lag and rolling window features.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with target column
        target_col : str
            Column to create lags for (default: 'trip_count')
            
        Returns
        -------
        pd.DataFrame
            DataFrame with lag features
        """
        self.logger.info(f"Creating lag features for '{target_col}'...")
        
        # lag features
        for lag in self.aggregation_config['lag_windows']:
            df[f'trip_count_lag_{lag}'] = df['trip_count'].shift(lag)
        
        # Rolling statistics (shifted to avoid data leakage)
        for window in [7, 14]:
            df[f'trip_count_rolling_mean_{window}'] = (
                df['trip_count'].shift(1).rolling(window=window).mean()
            )
            df[f'trip_count_rolling_std_{window}'] = (
                df['trip_count'].shift(1).rolling(window=window).std()
            )
        
        self.logger.info(f" Added lag and rolling features")
        
        return df
    
    def prepare_modeling_dataset(
        self,
        start_date: str,
        end_date: str,
        include_lags: bool = True,
        output_filename: str = "modeling_dataset.csv"
    ) -> pd.DataFrame:
        """
        Main pipeline: Load, process, and merge all data sources.
        
        Parameters
        ----------
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str
            End date in 'YYYY-MM-DD' format
        include_lags : bool
            Whether to include lag features (default: True)
        output_filename : str
            Name for output file (saved to output_dir)
            
        Returns
        -------
        pd.DataFrame
            Final modeling dataset ready for training
        """
        self.logger.info("STARTING DATA PREPARATION PIPELINE")
        self.logger.info(f"Date range: {start_date} to {end_date}")
        self.logger.info(f"Include lags: {include_lags}")
        self.logger.info("")
        
        # 1. Load pogoh data
        self.logger.info("loading pogoh data")
        pogoh_raw = self.load_pogoh_data(
            start_date=start_date,
            end_date=end_date
        )
        
        # 2. aggregate to daily
        self.logger.info("\n aggregating to daily")
        pogoh_daily = self.aggregate_to_daily(pogoh_raw)
        
        # 3. Load weather data
        self.logger.info("\n loading weather data")
        weather_df = self.load_weather_data(
            start_date=start_date,
            end_date=end_date
        )
        
        # 4. weather features
        self.logger.info("\n weather features")
        weather_df = self.create_weather_features(weather_df)
        
        # 5. merge data
        self.logger.info("\n merge data sources")
        date_range = pd.date_range(start_date, end_date, freq='D')
        df = pd.DataFrame({'date': date_range})
        
        df = df.merge(pogoh_daily, on='date', how='left')
        df = df.merge(weather_df, on='date', how='left')
        df = self.create_temporal_features(df) # add temporal features
        
        # 6. lag features
        if include_lags:
            self.logger.info("\n adding lag features")
            df = self.create_lag_features(df)
        else:
            self.logger.info("\n skipping lag feautres")

        # Interpolate weather columns
        weather_cols = ['temp_max', 'temp_min', 'temp_mean', 'precip', 'wind_speed']
        for col in weather_cols:
            if col in df.columns and df[col].isnull().any():
                df[col] = df[col].interpolate(method='linear', limit_direction='both')
    
        flag_cols = ['is_rainy', 'is_extreme_weather', 'is_extreme_heat', 'is_extreme_cold']
        for col in flag_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        output_path = self.output_dir / output_filename # save output directory
        df.to_csv(output_path, index=False)
    
        return df


__all__ = ["DataProcessor"]