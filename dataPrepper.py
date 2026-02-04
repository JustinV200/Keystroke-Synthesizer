import pandas as pd
import numpy as np
import string
from sklearn.preprocessing import StandardScaler
import joblib

class dataPrepper:
    def __init__(self, data):
        if isinstance(data, str):
            self.data = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            self.data = data.copy()
        else:
            raise ValueError("data must be a CSV path or pandas DataFrame.")
        self.scaler = None
        self.original_length = len(self.data)

    # cleaning data, remove invalid entries
    def clean_data(self):
        initial_len = len(self.data)
        #drop duplicates
        self.data.drop_duplicates(inplace=True)
        #drop rows with missing essential columns
        essential = ['DownTime', 'UpTime', 'ActionTime', 'DownEvent']
        self.data.dropna(subset=essential, inplace=True)
        #convert to numeric
        for col in ['DownTime', 'UpTime', 'ActionTime']:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
        self.data.dropna(subset=['DownTime', 'UpTime', 'ActionTime'], inplace=True)
        #sort by DownTime
        self.data.sort_values(by='DownTime', inplace=True)
        self.data.reset_index(drop=True, inplace=True)
        # Report on numer of rows removed
        removed = initial_len - len(self.data)
        if removed > 0:
            print(f"Cleaned data: removed {removed} rows ({removed/initial_len*100:.1f}%)")
    # What transformations to apply
    def transform_data(self):
        # dwell = Up - Down
        self.data['DwellTime'] = self.data['UpTime'] - self.data['DownTime']

        # flight = next Down - current Up
        self.data['NextDownTime'] = self.data['DownTime'].shift(-1)
        self.data['FlightTime']   = self.data['NextDownTime'] - self.data['UpTime']
        self.data.drop('NextDownTime', axis=1, inplace=True)

        initial_len = len(self.data)

        # Remove only clearly invalid data (negatives), keep outliers for stats
        self.data = self.data[self.data['DwellTime'] >= 0]
        self.data = self.data[(self.data['FlightTime'].isna()) | (self.data['FlightTime'] >= 0)]
        
        # Cap FlightTime at 900ms to remove outliers
        self.data.loc[self.data['FlightTime'] > 900, 'FlightTime'] = np.nan
        # Cap DwellTime at reasonable upper bound (300ms for normal typing)
        self.data.loc[self.data['DwellTime'] > 300, 'DwellTime'] = np.nan
        #cap typing speed at 490 cpm in later step

        self.data.reset_index(drop=True, inplace=True)
        removed = initial_len - len(self.data)
        if removed > 0:
            print(f"Filtered invalid timings (negatives): removed {removed} rows")
    #add in contextual flags, what type of key was pressed, pauses, cumulative counts, typing speed
    def addContextFlags(self):
        de = self.data["DownEvent"].astype(str)
        punct_set = set(string.punctuation)

        is_letter    = de.str.len().eq(1) & de.str.isalpha()
        is_digit     = de.str.len().eq(1) & de.str.isdigit()
        is_space     = (de == "Space") | (de == " ")
        is_backspace = (de == "Backspace")
        is_punct     = de.apply(lambda x: len(x) == 1 and x in punct_set)
        is_enter     = (de == "Enter") | (de == "\n")
        is_shift     = (de == "Shift")

        self.data["is_letter"]    = is_letter.astype(int)
        self.data["is_digit"]     = is_digit.astype(int)
        self.data["is_punct"]     = is_punct.astype(int)
        self.data["is_space"]     = is_space.astype(int)
        self.data["is_backspace"] = is_backspace.astype(int)
        self.data["is_enter"]     = is_enter.astype(int)
        self.data["is_shift"]     = is_shift.astype(int)

        # typing speed (chars per minute) over rolling window of DownTime
        self.data["typing_speed"] = self._calculate_typing_speed()

    def _calculate_typing_speed(self, window_size=10):
        # elapsed ms across window steps; first window_size rows become NaN
        elapsed = self.data['DownTime'].diff(window_size)
        # cpm = window_size / (elapsed_seconds / 60)
        cpm = window_size / (elapsed / 1000.0 / 60.0)
        cpm = cpm.replace([np.inf, -np.inf], np.nan)
        # Replace 0s with NaN (invalid data)
        cpm = cpm.replace(0.0, np.nan)
        cpm = cpm.clip(upper=490)  # cap at 490 cpm to avoid extreme outliers, keep NaN as NaN
        return cpm

    def add_char_encoding(self):
        # map DownEvent to char code
        #if space, backspace, enter, shift, use standard codes
        #otherwise, use char_to_code to get ASCII code
        de = self.data["DownEvent"].astype(str)
        def char_to_code(char):
            if len(char) == 1:
                return ord(char)
            if char == "Space":     return ord(' ')
            if char == "Backspace": return 8
            if char == "Enter":     return 13
            if char == "Shift":     return 16
            return 0
        self.data["char_code"] = de.apply(char_to_code)

    #use to normalize features, no longer used in main pipeline
    def save_scaler(self, path):
        if self.scaler:
            joblib.dump(self.scaler, path)
    def load_scaler(self, path):
        self.scaler = joblib.load(path)

    #  do everything here
    def get_prepared_data(self):
        print(f"Starting preprocessing: {len(self.data)} rows")

        self.clean_data()
        self.transform_data()
        self.addContextFlags()
        self.add_char_encoding()
        # keep raw values (no self.normalize())

        # ensure finite values in key columns
        self._finalize_finite()

        print(f"Preprocessing complete: {len(self.data)} rows retained "
              f"({len(self.data)/self.original_length*100:.1f}% of original)")
        
        # Debug: Check for any remaining NaNs
        nan_check_cols = ["DwellTime", "FlightTime", "typing_speed", "is_letter", "is_digit", "is_punct", "is_space", "is_backspace", "is_enter", "is_shift"]
        for col in nan_check_cols:
            if col in self.data.columns:
                nan_count = self.data[col].isna().sum()
                if nan_count > 0:
                    print(f"WARNING: {col} still has {nan_count} NaN values after preprocessing!")
        
        print(f"Stats on data after preprocessing: {self.get_statistics()}")
        return self.data

    # finalize finite values in key columns
    def _finalize_finite(self):
        cols = [
            "DwellTime", "FlightTime", "typing_speed",
            "is_letter", "is_digit", "is_punct", "is_space",
            "is_backspace", "is_enter", "is_shift"
        ]
        for c in cols:
            if c in self.data:
                # Replace inf with NaN, but keep NaN as NaN (don't fill with 0)
                self.data[c] = self.data[c].replace([np.inf, -np.inf], np.nan)
        
        # Convert NaN to 0 for binary classification flags (these should never be NaN)
        flag_cols = ["is_letter", "is_digit", "is_punct", "is_space", "is_backspace", "is_enter", "is_shift"]
        for flag_col in flag_cols:
            if flag_col in self.data.columns:
                self.data[flag_col] = self.data[flag_col].fillna(0).astype(int)
        
        # Drop rows with NaN in critical timing features, but allow 0 values for FlightTime
        initial_len = len(self.data)
        self.data = self.data[
            (self.data['DwellTime'].notna()) & (self.data['DwellTime'] != 0) &
            (self.data['FlightTime'].notna()) &  # Keep 0 FlightTime (immediate keystrokes)  
            (self.data['typing_speed'].notna()) & (self.data['typing_speed'] != 0)
        ]
        # Additional safety: drop any remaining rows with NaN in timing features only
        # (binary flags are already cleaned above)
        timing_cols = ["DwellTime", "FlightTime", "typing_speed"]
        existing_timing_cols = [c for c in timing_cols if c in self.data.columns]
        self.data = self.data.dropna(subset=existing_timing_cols)
        
        self.data.reset_index(drop=True, inplace=True)
        removed = initial_len - len(self.data)
        if removed > 0:
            print(f"Dropped rows with NaN/0 in timing features or NaN in any feature: {removed} rows ({removed/initial_len*100:.1f}%)")
    # get statistics on the data, may be useful for reporting
    def get_statistics(self):
        def safe_mean(col):
            return float(self.data[col].mean()) if col in self.data and len(self.data) else float("nan")
        def safe_mean_int(col):
            return float(self.data[col].mean()) if col in self.data and len(self.data) else float("nan")

        stats = {
            'total_keystrokes': int(len(self.data)),
            'avg_dwell_time':   safe_mean('DwellTime'),
            'avg_flight_time':  safe_mean('FlightTime'),
            'backspace_rate':   safe_mean_int('is_backspace'),
            'avg_typing_speed': safe_mean('typing_speed'),
        }
        return stats
