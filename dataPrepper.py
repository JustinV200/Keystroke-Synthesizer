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
        
        # Cap FlightTime at 900ms to remove  outliers (any extended breaks, we just wanna capture normal typing)
        self.data.loc[self.data['FlightTime'] > 900, 'FlightTime'] = np.nan

        # Cap DwellTime at reasonable upper bound (300-400ms for normal typing)
        self.data.loc[self.data['DwellTime'] > 300, 'DwellTime'] = np.nan

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
        #short pause
        self.data["is_pause_2s"] = (self.data["FlightTime"] >= 2000).fillna(False).astype(int)
        #long pause
        self.data["is_pause_5s"] = (self.data["FlightTime"] >= 5000).fillna(False).astype(int)

        # typing speed (chars per minute) over rolling window of DownTime
        self.data["typing_speed"] = self._calculate_typing_speed()

    def _calculate_typing_speed(self, window_size=10):
        # elapsed ms across window steps; first window_size rows become NaN
        elapsed = self.data['DownTime'].diff(window_size)
        # cpm = window_size / (elapsed_seconds / 60)
        cpm = window_size / (elapsed / 1000.0 / 60.0)
        cpm = cpm.replace([np.inf, -np.inf], np.nan).fillna(method="bfill")
        # Replace 0s with NaN (invalid data), then backfill
        cpm = cpm.replace(0.0, np.nan)
        # Use median as fallback instead of hardcoded 50.0 to avoid contaminating data
        median_speed = cpm.median() if not cpm.isna().all() else 150.0
        cpm = cpm.fillna(method="bfill").fillna(median_speed)
        cpm = cpm.clip(upper=500)
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
        print(f"Stats on data after preprocessing: {self.get_statistics()}")
        return self.data

    # finalize finite values in key columns
    def _finalize_finite(self):
        cols = [
            "DwellTime", "FlightTime", "typing_speed",
            "is_letter", "is_digit", "is_punct", "is_space",
            "is_backspace", "is_enter", "is_shift",
            "is_pause_2s", "is_pause_5s"
        ]
        for c in cols:
            if c in self.data:
                self.data[c] = (self.data[c]
                                .replace([np.inf, -np.inf], np.nan)
                                .fillna(0.0))

        # common NaN sources:
        if "FlightTime" in self.data:
            self.data["FlightTime"] = self.data["FlightTime"].fillna(0.0)
        if "typing_speed" in self.data:
            self.data["typing_speed"] = self.data["typing_speed"].fillna(0.0)
        if "DwellTime" in self.data:
            self.data["DwellTime"] = self.data["DwellTime"].fillna(0.0)
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
            'pause_rate_2s':    safe_mean_int('is_pause_2s'),
            'pause_rate_5s':    safe_mean_int('is_pause_5s'),
        }
        return stats
