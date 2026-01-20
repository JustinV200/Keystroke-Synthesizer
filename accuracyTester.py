#accuracy tester.py
# step 1: loop through original csv files and collect DwellTime,FlightTime,typing_speed
# dwelltime = UpTime - DownTime
# flighttime = next DownTime - current UpTime
# typing_speed = number of chars / (last ActionTime - first ActionTime) * 1000 (chars per second)
# step 2: compute mean and std for each of these 3 features across all files
# step 3: loop through original text and run synthesizeKeystrokes.py to generate predicted csv files
# collect mean and std for DwellTime,FlightTime,typing_speed from predicted csv files
# compare original vs predicted means and stds, print results
#note, this only works on the dwell time, flightime, and typing speed features, havent added the flags yet.
import os
import pandas as pd
import numpy as np
from synthesizeKeystrokes import *

# calculate dwelltime, flighttime, typing_speed for original data
# return lists of values for each feature
def computeOgStats():
    base_dir = "data"
    text_dir = os.path.join(base_dir, "texts")
    csv_dir  = os.path.join(base_dir, "csv")
    ogDwell_times = []
    ogFlight_times = []
    ogTyping_speeds = []
    for fid in os.listdir(csv_dir):
        if not fid.endswith(".csv"):
            continue
        csv_path = os.path.join(csv_dir, fid)
        df = pd.read_csv(csv_path)
        #calculate features
        dwelltime = df['UpTime'] - df['DownTime']
        flighttime = df['DownTime'].shift(-1) - df['UpTime']
        # Calculate typing speed per keystroke using rolling window (matching synthesized approach)
        window_size = 10
        elapsed = df['DownTime'].diff(window_size)
        typing_speed = window_size / (elapsed / 1000.0 / 60.0)  # chars per minute
        typing_speed = typing_speed.replace([np.inf, -np.inf], np.nan).fillna(method="bfill").fillna(0.0)
        typing_speed = typing_speed.clip(upper=500)
        #Add to lists
        ogDwell_times.extend(dwelltime.dropna().tolist())
        ogFlight_times.extend(flighttime.dropna().tolist())
        ogTyping_speeds.extend(typing_speed.dropna().tolist())
    return ogDwell_times, ogFlight_times, ogTyping_speeds

# synthesize keystrokes for each text file and get stats
def computeSynthStats(Synthesize = True):
    base_dir = "data"
    text_dir = os.path.join(base_dir, "texts")
    csv_dir = os.path.join(base_dir, "predicted_csvs")
    #if we want to resynthesize, run this, otherwise use already existing predicted csvs
    if Synthesize:
        for texts in os.listdir(text_dir):
            if not texts.endswith(".txt"):
                continue
            text_path = os.path.join(text_dir, texts)
            output_csv = os.path.join(csv_dir, texts.replace(".txt", "_predicted.csv"))
            predict_keystrokes(
                text_path=text_path,
                checkpoint_path="checkpoints/best_model.pt",
                base_model="microsoft/deberta-v3-base",
                output_csv=output_csv,
                stats_path="data/cont_stats.json"
            )
    # after generating all predicted csvs, compute stats
    synthDwell_times = []
    synthFlight_times = []
    synthTyping_speeds = []
    for fid in os.listdir(csv_dir):
        if not fid.endswith(".csv"):
            continue
        csv_path = os.path.join(csv_dir, fid)
        df = pd.read_csv(csv_path)
        #calculate features
        dwelltime = df['DwellTime']
        flighttime = df['FlightTime']
        typing_speed = df['typing_speed']
        synthDwell_times.extend(dwelltime.dropna().tolist())
        synthFlight_times.extend(flighttime.dropna().tolist())
        synthTyping_speeds.extend(typing_speed.dropna().tolist())
    return synthDwell_times, synthFlight_times, synthTyping_speeds

def compare():
    ogDwell, ogFlight, ogTyping = computeOgStats()
    synthDwell, synthFlight, synthTyping = computeSynthStats()
    print("Original Data Stats:")
    print(f"DwellTime - Mean: {np.mean(ogDwell)}, Std: {np.std(ogDwell)}")
    print(f"FlightTime - Mean: {np.mean(ogFlight)}, Std: {np.std(ogFlight)}")
    print(f"Typing Speed - Mean: {np.mean(ogTyping)}, Std: {np.std(ogTyping)}")
    print("\nSynthesized Data Stats:")
    print(f"DwellTime - Mean: {np.mean(synthDwell)}, Std: {np.std(synthDwell)}")
    print(f"FlightTime - Mean: {np.mean(synthFlight)}, Std: {np.std(synthFlight)}")
    print(f"Typing Speed - Mean: {np.mean(synthTyping)}, Std: {np.std(synthTyping)}")
    print("\nComparison:")
    print(f"DwellTime - Mean Diff: {abs(np.mean(ogDwell) - np.mean(synthDwell))}, Std Diff: {abs(np.std(ogDwell) - np.std(synthDwell))}")
    print(f"FlightTime - Mean Diff: {abs(np.mean(ogFlight) - np.mean(synthFlight))}, Std Diff: {abs(np.std(ogFlight) - np.std(synthFlight))}")
    print(f"Typing Speed - Mean Diff: {abs(np.mean(ogTyping) - np.mean(synthTyping))}, Std Diff: {abs(np.std(ogTyping) - np.std(synthTyping))}")

if __name__ == "__main__":
    compare()