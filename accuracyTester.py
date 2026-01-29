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
from scipy import stats

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
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding='latin-1')
        # set size of df to first 512 rows to match synthesized data length
        df = df.head(512)
        #calculate features
        dwelltime = df['UpTime'] - df['DownTime']
        flighttime = df['DownTime'].shift(-1) - df['UpTime']
        # Set first FlightTime to NaN to match synthesized data
        if len(flighttime) > 0:
            flighttime.iloc[0] = np.nan
        # Cap FlightTime at 900ms to match training data preprocessing (dataPrepper.py line 54)
        flighttime = flighttime.clip(upper=900)
        dwelltime = dwelltime.clip(upper=300)
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
    
    # Debug FlightTime distribution to check for outliers
    flight_array = np.array(ogFlight_times)
    print(f"FlightTime distribution analysis:")
    print(f"  Mean: {flight_array.mean():.2f}, Std: {flight_array.std():.2f}")
    print(f"  Min: {flight_array.min():.2f}, Max: {flight_array.max():.2f}")
    print(f"  Percentiles: 50%={np.percentile(flight_array, 50):.1f}, 90%={np.percentile(flight_array, 90):.1f}, 95%={np.percentile(flight_array, 95):.1f}, 99%={np.percentile(flight_array, 99):.1f}, 99.9%={np.percentile(flight_array, 99.9):.1f}")
    print(f"  Values >10sec: {(flight_array > 10000).sum()}/{len(flight_array)} ({(flight_array > 10000).mean()*100:.1f}%)")
    print(f"  Values >30sec: {(flight_array > 30000).sum()}/{len(flight_array)} ({(flight_array > 30000).mean()*100:.1f}%)")
        
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
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding='latin-1')
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
    
    print("="*80)
    print("KEYSTROKE SYNTHESIS EVALUATION")
    print("="*80)
    
    print("\n--- BASIC STATISTICS ---")
    print("\nOriginal Data:")
    print(f"  DwellTime    - Mean: {np.mean(ogDwell):8.2f}, Std: {np.std(ogDwell):8.2f}, N: {len(ogDwell)}")
    print(f"  FlightTime   - Mean: {np.mean(ogFlight):8.2f}, Std: {np.std(ogFlight):8.2f}, N: {len(ogFlight)}")
    print(f"  Typing Speed - Mean: {np.mean(ogTyping):8.2f}, Std: {np.std(ogTyping):8.2f}, N: {len(ogTyping)}")
    
    print("\nSynthesized Data:")
    print(f"  DwellTime    - Mean: {np.mean(synthDwell):8.2f}, Std: {np.std(synthDwell):8.2f}, N: {len(synthDwell)}")
    print(f"  FlightTime   - Mean: {np.mean(synthFlight):8.2f}, Std: {np.std(synthFlight):8.2f}, N: {len(synthFlight)}")
    print(f"  Typing Speed - Mean: {np.mean(synthTyping):8.2f}, Std: {np.std(synthTyping):8.2f}, N: {len(synthTyping)}")
    
    print("\n--- STATISTICAL TESTS ---\n")
    
    features = [
        ("DwellTime", ogDwell, synthDwell),
        ("FlightTime", ogFlight, synthFlight),
        ("Typing Speed", ogTyping, synthTyping)
    ]
    
    for name, orig, synth in features:
        print(f"{name}:")
        
        # 1. Two-sample t-test (tests if means differ)
        t_stat, t_pval = stats.ttest_ind(orig, synth)
        sig_t = '***' if t_pval < 0.001 else '**' if t_pval < 0.01 else '*' if t_pval < 0.05 else 'ns'
        print(f"  t-test:       t={t_stat:7.3f}, p={t_pval:.4f} {sig_t}")
        
        # 2. Kolmogorov-Smirnov test (tests if distributions differ)
        ks_stat, ks_pval = stats.ks_2samp(orig, synth)
        sig_ks = '***' if ks_pval < 0.001 else '**' if ks_pval < 0.01 else '*' if ks_pval < 0.05 else 'ns'
        print(f"  K-S test:     D={ks_stat:7.3f}, p={ks_pval:.4f} {sig_ks}")
        
        # 3. Effect size (Cohen's d)
        pooled_std = np.sqrt((np.std(orig)**2 + np.std(synth)**2) / 2)
        cohens_d = (np.mean(orig) - np.mean(synth)) / pooled_std if pooled_std > 0 else 0
        effect = "negligible" if abs(cohens_d) < 0.2 else "small" if abs(cohens_d) < 0.5 else "medium" if abs(cohens_d) < 0.8 else "large"
        print(f"  Cohen's d:    {cohens_d:7.3f} ({effect})")
        
        # 4. Mean absolute difference
        mean_diff = abs(np.mean(orig) - np.mean(synth))
        mean_pct = (mean_diff / np.mean(orig) * 100) if np.mean(orig) != 0 else 0
        print(f"  Mean diff:    {mean_diff:7.2f} ({mean_pct:.1f}%)")
        print()
    
    print("="*80)
    print("Note: larger datasets will not necessarily yield better p-values due to increased statistical power.")
    print("consider the size of the dataset, for larger ones, cohen's d and mean differences may be more informative.")

    print("="*80)
if __name__ == "__main__":
    compare()