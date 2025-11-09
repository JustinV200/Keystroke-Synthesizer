import pandas as pd
import numpy as np
import string
from sklearn.preprocessing import StandardScaler
import joblib

class dataPrepper():
    def __init__(self, data):

        if isinstance(data, str):
            self.data = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            self.data = data.copy()
        else:
         raise ValueError("data must be a CSV path or pandas DataFrame.")
        
        self.scaler = None
        self.original_length = len(self.data)

    def clean_data(self):
        initial_len = len(self.data)
        # Drop duplicates
        self.data.drop_duplicates(inplace=True)
        
        # Remove rows with missing essential columns
        essential_cols = ['DownTime', 'UpTime', 'ActionTime', 'DownEvent']
        self.data.dropna(subset=essential_cols, inplace=True)

        # Convert to numeric and sort chronologically
        for col in ['DownTime', 'UpTime', 'ActionTime']:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')

        self.data.dropna(subset=['DownTime', 'UpTime', 'ActionTime'], inplace=True)

        self.data.sort_values(by='DownTime', inplace=True)

        # Reset index after cleaning
        self.data.reset_index(drop=True, inplace=True)

        removed = initial_len - len(self.data)
        if removed > 0:
            print(f"Cleaned data: removed {removed} rows ({removed/initial_len*100:.1f}%)")


    def transform_data(self):

        # Compute dwell time
        # dwell time = time key is held down = UpTime - DownTime
        self.data['DwellTime'] = self.data['UpTime'] - self.data['DownTime']

        # Compute flight time between consecutive keys
        # flight time = time between releasing one key and pressing the next = Next DownTime - UpTime
        self.data['NextDownTime'] = self.data['DownTime'].shift(-1)
        self.data['FlightTime'] = self.data['NextDownTime'] - self.data['UpTime']

        # Drop the helper column
        self.data.drop('NextDownTime', axis=1, inplace=True)

        # Store stats before filtering
        initial_len = len(self.data)


        # Remove negative or unrealistic values
        self.data = self.data[
            (self.data['DwellTime'] >= 0) & 
            (self.data['DwellTime'] < 2000)
        ]

       # For FlightTime, allow NaN (last keystroke) but filter unrealistic values
        self.data = self.data[
            (self.data['FlightTime'].isna()) | 
            ((self.data['FlightTime'] >= 0) & (self.data['FlightTime'] < 5000))
        ]
    
        self.data.reset_index(drop=True, inplace=True)
        
        removed = initial_len - len(self.data)
        if removed > 0:
            print(f"Filtered unrealistic timings: removed {removed} rows")

    def save_scaler(self, path):
        if self.scaler:
            joblib.dump(self.scaler, path)

    def load_scaler(self, path):
        self.scaler = joblib.load(path)

    def addContextFlags(self):
        # add context flags based on ActionTime, 
        # telling apart letters,numbers, puncutation, spaces, or backspaces, or long pauses(>=2 seconds).
        de = self.data["DownEvent"].astype(str)
        # Simple category checks
        punct_set = set(string.punctuation)
        is_letter = de.str.len().eq(1) & de.str.isalpha()
        is_digit = de.str.len().eq(1) & de.str.isdigit()
        is_space = (de == "Space") | (de == " ")
        is_backspace = (de == "Backspace")
        is_punct = de.apply(lambda x: len(x) == 1 and x in punct_set)
        
        # Special keys
        is_enter = (de == "Enter") | (de == "\n")
        is_shift = (de == "Shift")
        
        # Add flags as integers
        self.data["is_letter"] = is_letter.astype(int)
        self.data["is_digit"] = is_digit.astype(int)
        self.data["is_punct"] = is_punct.astype(int)
        self.data["is_space"] = is_space.astype(int)
        self.data["is_backspace"] = is_backspace.astype(int)
        self.data["is_enter"] = is_enter.astype(int)
        self.data["is_shift"] = is_shift.astype(int)

        # Pause flags based on FlightTime
        self.data["is_pause_2s"] = (self.data["FlightTime"] >= 2000).fillna(False).astype(int)
        self.data["is_pause_5s"] = (self.data["FlightTime"] >= 5000).fillna(False).astype(int)

        # Cumulative statistics
        self.data["cum_backspace"] = self.data["is_backspace"].cumsum()
        self.data["cum_chars"] = (~self.data["is_backspace"]).cumsum()
        
        # Typing speed (characters per minute) - rolling window
        self.data["typing_speed"] = self._calculate_typing_speed()

    def _calculate_typing_speed(self, window_size=10):
        # time elapsed over window (in milliseconds)
        time_elapsed = self.data['DownTime'].diff(window_size)
        
        # Convert to characters per minute
        chars_per_min = (window_size / (time_elapsed / 1000 / 60)).fillna(0)
        
        # Cap at reasonable maximum
        chars_per_min = chars_per_min.clip(upper=500)
        
        return chars_per_min
    
    def add_char_encoding(self):
        de = self.data["DownEvent"].astype(str)
        
        def char_to_code(char):
            """Convert character/key to numeric code"""
            if len(char) == 1:
                return ord(char)
            elif char == "Space":
                return ord(' ')
            elif char == "Backspace":
                return 8
            elif char == "Enter":
                return 13
            elif char == "Shift":
                return 16
            else:
                return 0  # Unknown/special key
        
        self.data["char_code"] = de.apply(char_to_code)

    def normalize(self):
        """Normalize continuous features using StandardScaler"""
        continuous_cols = ["DwellTime", "FlightTime", "typing_speed"]
        
        # Only normalize columns that exist
        cols_present = [c for c in continuous_cols if c in self.data.columns]
        
        
        # Handle NaN in FlightTime before normalization
        self.data["is_last_key"] = self.data["FlightTime"].isna().astype(int)
        # Fill NaN with median of the column
        for col in cols_present:
            if self.data[col].isna().any():
                median_val = self.data[col].median()
                self.data[col].fillna(median_val, inplace=True)
        
        # Fit and transform
        self.scaler = StandardScaler()
        self.data[cols_present] = self.scaler.fit_transform(self.data[cols_present])


    def get_prepared_data(self):
        print(f"Starting preprocessing: {len(self.data)} rows")
        
        self.clean_data()
        self.transform_data()
        self.addContextFlags()
        self.add_char_encoding()
        self.normalize()
        
        print(f"Preprocessing complete: {len(self.data)} rows retained "
              f"({len(self.data)/self.original_length*100:.1f}% of original)")
        
        print(f'Stats on data after preprocessing: {self.get_statistics()} ')
        return self.data
    def get_statistics(self):
        """Get summary statistics of the processed data"""
        stats = {
            'total_keystrokes': len(self.data),
            'avg_dwell_time': self.data['DwellTime'].mean(),
            'avg_flight_time': self.data['FlightTime'].mean(),
            'backspace_rate': self.data['is_backspace'].mean(),
            'avg_typing_speed': self.data['typing_speed'].mean(),
            'pause_rate_2s': self.data['is_pause_2s'].mean(),
            'pause_rate_5s': self.data['is_pause_5s'].mean(),
        }
        return stats