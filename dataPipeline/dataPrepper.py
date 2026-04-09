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
    # Calculate DwellTime (per-key, independent of neighbors)
    # FlightTime is calculated later in _filter_to_surviving_keystrokes after filtering
    def transform_data(self):
        # dwell = Up - Down
        self.data['DwellTime'] = self.data['UpTime'] - self.data['DownTime']

        initial_len = len(self.data)

        # Remove clearly invalid DwellTimes (negatives)
        self.data = self.data[self.data['DwellTime'] >= 0]
        # Cap DwellTime at reasonable upper bound (300ms for normal typing)
        self.data.loc[self.data['DwellTime'] > 300, 'DwellTime'] = np.nan

        self.data.reset_index(drop=True, inplace=True)
        removed = initial_len - len(self.data)
        if removed > 0:
            print(f"Filtered invalid DwellTimes (negatives): removed {removed} rows")

    def _filter_to_surviving_keystrokes(self):
        """Replay the edit sequence to keep only keystrokes whose characters survived.
        
        - Backspace: removes the most recent surviving keystroke
        - Non-producing keys (Shift, Leftclick, etc.): removed entirely
        - Surviving character keystrokes: kept
        
        After filtering:
        - DwellTime preserved (per-key, unaffected by neighbors)
        - FlightTime recalculated; NaN where removed keys existed in between
        """
        surviving_indices = []

        for idx in range(len(self.data)):
            event = str(self.data.iloc[idx]['DownEvent'])

            if event == 'Backspace':
                if surviving_indices:
                    surviving_indices.pop()
            elif self._is_char_producing(event):
                surviving_indices.append(idx)
            # else: non-producing key (Shift, Leftclick, etc.) → skip

        if not surviving_indices:
            print("WARNING: No surviving keystrokes after replay!")
            self.data = self.data.iloc[0:0]
            return

        initial_len = len(self.data)
        orig_indices = np.array(surviving_indices)
        self.data = self.data.iloc[surviving_indices].copy()
        self.data.reset_index(drop=True, inplace=True)

        # Recalculate FlightTime on filtered data
        self.data['FlightTime'] = self.data['DownTime'].shift(-1) - self.data['UpTime']

        # NaN out FlightTime where original indices weren't consecutive
        # (removed keystrokes existed between these two survivors → timing is unreliable)
        non_consecutive = np.diff(orig_indices) != 1
        nan_mask = np.append(non_consecutive, True)  # last row → no next key
        self.data.loc[nan_mask, 'FlightTime'] = np.nan

        # Cap FlightTime at 900ms, remove negatives
        self.data.loc[self.data['FlightTime'] > 900, 'FlightTime'] = np.nan
        self.data.loc[(self.data['FlightTime'].notna()) & (self.data['FlightTime'] < 0), 'FlightTime'] = np.nan

        removed = initial_len - len(self.data)
        nan_flights = self.data['FlightTime'].isna().sum()
        print(f"Filtered to surviving keystrokes: {len(self.data)} kept, {removed} removed "
              f"({nan_flights} FlightTimes NaN'd at non-consecutive transitions)")

    @staticmethod
    def _is_char_producing(event):
        """Check if a DownEvent produces a character in the final text."""
        if len(event) == 1:
            return True
        return event in ('Space', 'Enter')
    #add in contextual flags, what type of key was pressed, pauses, cumulative counts, typing speed
    def addContextFlags(self):
    #this is for classification head, which we dont use anymore, but may be useful for analysis and future work
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

    def _add_char_columns(self):
        """Add char and prev_char columns from DownEvent.
        After filtering to surviving keystrokes, all events are character-producing.
        Ties each DwellTime to a specific character and each FlightTime to a character pair."""
        de = self.data["DownEvent"].astype(str)

        def event_to_char(event):
            if len(event) == 1:
                return event
            if event == "Space":
                return " "
            if event == "Enter":
                return "\n"
            return f"<{event}>"  # shouldn't happen after filtering

        self.data["char"] = de.apply(event_to_char)
        self.data["prev_char"] = self.data["char"].shift(1).fillna("")


    def save_scaler(self, path):
        if self.scaler:
            joblib.dump(self.scaler, path)
    def load_scaler(self, path):
        self.scaler = joblib.load(path)

    #  do everything here
    def get_prepared_data(self):
        print(f"Starting preprocessing: {len(self.data)} rows")

        self.clean_data()
        self.transform_data()              # DwellTime only (per-key)
        self._filter_to_surviving_keystrokes()  # replay edits, filter, calc FlightTime
        self.data["typing_speed"] = self._calculate_typing_speed()  # on filtered data
        self.add_char_encoding()
        self._add_char_columns()           # char/prev_char on surviving keystrokes

        # ensure finite values in key columns
        self._finalize_finite()

        print(f"Preprocessing complete: {len(self.data)} rows retained "
              f"({len(self.data)/self.original_length*100:.1f}% of original)")
        
        # Debug: Report NaN counts (NaN in FlightTime/typing_speed is intentional)
        for col in ["DwellTime", "FlightTime", "typing_speed"]:
            if col in self.data.columns:
                nan_count = self.data[col].isna().sum()
                if nan_count > 0:
                    print(f"  {col}: {nan_count} NaN values ({nan_count/len(self.data)*100:.1f}%)")
        
        print(f"Stats: {self.get_statistics()}")
        return self.data

    # finalize finite values in key columns
    def _finalize_finite(self):
        cols = ["DwellTime", "FlightTime", "typing_speed"]
        for c in cols:
            if c in self.data:
                self.data[c] = self.data[c].replace([np.inf, -np.inf], np.nan)
        
        # Only drop rows where DwellTime is invalid (NaN or 0)
        # FlightTime NaN is intentional (non-consecutive transitions after filtering)
        # typing_speed NaN is intentional (first rows in rolling window)
        # The loss function already masks NaN positions per-feature
        initial_len = len(self.data)
        self.data = self.data[
            (self.data['DwellTime'].notna()) & (self.data['DwellTime'] != 0)
        ]
        
        self.data.reset_index(drop=True, inplace=True)
        removed = initial_len - len(self.data)
        if removed > 0:
            print(f"Dropped rows with invalid DwellTime: {removed} rows ({removed/initial_len*100:.1f}%)")
    # get statistics on the data, may be useful for reporting
    def get_statistics(self):
        def safe_mean(col):
            if col in self.data and len(self.data):
                return float(self.data[col].mean(skipna=True))
            return float("nan")

        stats = {
            'total_keystrokes': int(len(self.data)),
            'avg_dwell_time':   safe_mean('DwellTime'),
            'avg_flight_time':  safe_mean('FlightTime'),
            'avg_typing_speed': safe_mean('typing_speed'),
            'nan_flight_pct':   float(self.data['FlightTime'].isna().mean() * 100) if 'FlightTime' in self.data else 0.0,
        }
        return stats
