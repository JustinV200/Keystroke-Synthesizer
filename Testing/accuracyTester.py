#accuracy tester.py
# Evaluates model accuracy using existing data pipeline components
# Compares original vs synthesized keystroke statistics

import os
import pandas as pd
import numpy as np
import torch
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Testing.synthesizeKeystrokes import predict_keystrokes
from dataPipeline.dataPrepper import dataPrepper
from dataPipeline.dataLoader import dataLoader
from scipy import stats
from torch.utils.data import DataLoader as TorchDataLoader

# Use existing data pipeline to get original statistics
def computeOgStats():
    """Extract original keystroke statistics using dataPrepper pipeline."""
    print("Loading original data using dataPrepper pipeline...")
    
    # Process all CSV files using dataPrepper (same as dataLoader does internally)
    base_dir = "../data"
    csv_dir = os.path.join(base_dir, "csv")
    
    ogDwell_times = []
    ogFlight_times = []
    ogTyping_speeds = []
    
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
    print(f"Processing {len(csv_files)} CSV files...")
    
    for i, csv_file in enumerate(csv_files):
        csv_path = os.path.join(csv_dir, csv_file)
        
        try:
            # Use dataPrepper to process each CSV file with exact same logic as training
            prepper = dataPrepper(csv_path)
            prepper.clean_data()
            prepper.transform_data()
            prepper.addContextFlags()
            prepper._calculate_typing_speed()
            prepper.add_char_encoding()
            prepper._finalize_finite()
            
            # Get the processed data (already cleaned by dataPrepper)
            processed_data = prepper.get_prepared_data()
            
            # Extract continuous features: [DwellTime, FlightTime, typing_speed] 
            cont_idx = [0, 1, 2]  # Match training indices
            
            # Extract continuous features and filter any remaining NaNs for safety
            dwell = processed_data[:, cont_idx[0]]
            flight = processed_data[:, cont_idx[1]]
            typing = processed_data[:, cont_idx[2]]
            
            # Set first FlightTime to NaN for consistency (no previous keystroke)
            if len(flight) > 0:
                flight = flight.copy()  # Avoid modifying original array
                flight[0] = np.nan
            
            # Safety filter: remove any NaNs that might have slipped through
            dwell_clean = dwell[~np.isnan(dwell)]
            flight_clean = flight[~np.isnan(flight)]  # This will skip the first FlightTime NaN
            typing_clean = typing[~np.isnan(typing)]
            
            ogDwell_times.extend(dwell_clean.tolist())
            ogFlight_times.extend(flight_clean.tolist())
            ogTyping_speeds.extend(typing_clean.tolist())
            
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            continue
            
        # Progress indicator
        if (i + 1) % 100 == 0:
            print(f"Processed {i+1}/{len(csv_files)} files...")
    
    print(f"Extracted statistics from {len(csv_files)} files")
    print(f"Total samples: DwellTime={len(ogDwell_times)}, FlightTime={len(ogFlight_times)}, TypingSpeed={len(ogTyping_speeds)}")
    
    # Debug FlightTime distribution
    if ogFlight_times:
        flight_array = np.array(ogFlight_times)
        print(f"FlightTime distribution analysis:")
        print(f"  Mean: {flight_array.mean():.2f}, Std: {flight_array.std():.2f}")
        print(f"  Min: {flight_array.min():.2f}, Max: {flight_array.max():.2f}")
        print(f"  Percentiles: 50%={np.percentile(flight_array, 50):.1f}, 90%={np.percentile(flight_array, 90):.1f}, 99%={np.percentile(flight_array, 99):.1f}")
    
    return ogDwell_times, ogFlight_times, ogTyping_speeds

# Synthesize keystrokes and compute stats using existing pipeline
def computeSynthStats(synthesize=True):
    """Generate synthetic keystrokes and extract statistics."""
    base_dir = "../data"
    text_dir = os.path.join(base_dir, "texts")
    csv_dir = os.path.join(base_dir, "predicted_csvs")
    
    # Create output directory if it doesn't exist
    os.makedirs(csv_dir, exist_ok=True)
    
    # Generate predicted CSVs if requested
    if synthesize:
        print("Generating synthetic keystrokes...")
        text_files = [f for f in os.listdir(text_dir) if f.endswith(".txt")]
        
        for i, text_file in enumerate(text_files):
            if i % 50 == 0:  # Progress indicator
                print(f"Processing {i+1}/{len(text_files)}: {text_file}")
                
            text_path = os.path.join(text_dir, text_file)
            output_csv = os.path.join(csv_dir, text_file.replace(".txt", "_predicted.csv"))
            
            try:
                predict_keystrokes(
                    text_path=text_path,
                    checkpoint_path="../checkpoints/best_model.pt",
                    base_model="microsoft/deberta-v3-base",
                    output_csv=output_csv,
                    stats_path="../data/cont_stats.json"
                )
            except Exception as e:
                print(f"Error processing {text_file}: {e}")
                continue
    
    # Collect statistics from predicted CSVs
    print("Collecting synthesized statistics...")
    synthDwell_times = []
    synthFlight_times = []
    synthTyping_speeds = []
    
    predicted_files = [f for f in os.listdir(csv_dir) if f.endswith(".csv")]
    
    for csv_file in predicted_files:
        csv_path = os.path.join(csv_dir, csv_file)
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
            
            # Extract features and remove NaN values
            dwell = df['DwellTime'].dropna()
            flight = df['FlightTime'].dropna()  # Already has NaN for first keystroke
            typing = df['typing_speed'].dropna()
            
            synthDwell_times.extend(dwell.tolist())
            synthFlight_times.extend(flight.tolist())
            synthTyping_speeds.extend(typing.tolist())
            
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            continue
    
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