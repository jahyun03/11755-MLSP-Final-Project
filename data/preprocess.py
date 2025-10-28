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

class DataProcessor:
    def __init__(self, data_dir = "processed_data/raw"):
        """
        Initialize the DataProcessor with paths and configuration parameters

        Parameters
        ----------
        data_dir : str, optional
            Directory where the data files are located, by default ""
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir
        self.processed_dir = self.data_dir / "processed"
        self.output_dir = self.data_dir.parent / "ready"

        # create directories if they dont exist yet
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.file_patterns = {
            'pogoh': 'pogoh_data_*.csv',
            'weather': 'weather_data_*.csv',
            'closures': 'closure_data_*.csv',
            'events': 'events_data_*.csv',
            'transit': 'transit_data_*.csv'
        }      

        # data processing param
        self.aggregation_config = {
            'freq': 'D',  # Daily aggregation
            'trip_metrics': ['count', 'duration_mean', 'duration_std'],
            'weather_metrics': ['temp_mean', 'temp_min', 'temp_max', 'precip_sum'],
            'lag_windows': [1, 7, 14, 30]  # Lag features in days
        }
        
        # temporal features
        self.temporal_features = {
            'day_of_week': True,
            'month': True,
            'quarter': True,
            'is_weekend': True,
            'is_holiday': True,  # Will need holiday calendar
            'day_of_year': True
        }

        # data range
        self.data_range = {
            'start_date': None,
            'end_date' : None,
            'common_start' : None,
            'common_end' : None
        }
        self.setup_logging()

    def setup_logging(self):
        import logging
        log_dir = self.data_dir.parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            handlers=[
            logging.FileHandler(log_dir / 'preprocessing.log'),
            logging.StreamHandler()
        ])
        self.logger = logging.getLogger('DataProcessor')
        self.logger.info(f"DataProcessor initialized with data_dir: {self.data_dir}")


    def load_pogoh_data(self, file_path):
        """
        Load POGOH trip data and perform basic cleaning operations.

        Parameters
        ----------
        """




    def aggregate_to_daily(self, df, date_col, agg_config):
        """Convert trip-level data to daily aggregates"""
        daily = df.groupby('start_date').agg({
            'trip_id': 'count',
            'duration': ['mean', 'std']
        }).round(2)
        daily.columns = ['trip_count', 'duration_mean', 'duration_std']
        return daily
    
    # def create_temporal_features(self, data):
    #     # days of week, month
    
    # def prepare_modeling(self, df):
    #     # feature engineering