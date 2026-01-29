
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
        # Cap FlightTime at 10 seconds to match training data preprocessing (dataPrepper.py line 54)
        flighttime = flighttime.clip(upper=10000)
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

def plot_distributions():
    """Plot histograms and distribution plots for the original keystroke metrics."""
    print("Computing original statistics...")
    dwell_times, flight_times, typing_speeds = computeOgStats()
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Original Keystroke Data Distributions', fontsize=16, fontweight='bold')
    
    # Calculate reasonable x-axis limits (focus on 95th percentile to avoid extreme outliers)
    dwell_95th = np.percentile(dwell_times, 95)
    flight_95th = np.percentile(flight_times, 95)
    
    # Dwell Time plots
    axes[0, 0].hist(dwell_times, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Dwell Time Distribution')
    axes[0, 0].set_xlabel('Dwell Time (ms)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_xlim(0, min(dwell_95th * 1.2, 1000))  # Cap at 1000ms max
    axes[0, 0].grid(True, alpha=0.3)
    
    # Dwell Time box plot
    axes[1, 0].boxplot(dwell_times, vert=True)
    axes[1, 0].set_title('Dwell Time Box Plot')
    axes[1, 0].set_ylabel('Dwell Time (ms)')
    axes[1, 0].set_ylim(0, min(dwell_95th * 1.2, 1000))  # Cap at 1000ms max
    axes[1, 0].grid(True, alpha=0.3)
    
    # Flight Time plots
    axes[0, 1].hist(flight_times, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[0, 1].set_title('Flight Time Distribution')
    axes[0, 1].set_xlabel('Flight Time (ms)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_xlim(0, min(flight_95th * 1.2, 2000))  # Cap at 2000ms max
    axes[0, 1].grid(True, alpha=0.3)
    
    # Flight Time box plot
    axes[1, 1].boxplot(flight_times, vert=True)
    axes[1, 1].set_title('Flight Time Box Plot')
    axes[1, 1].set_ylabel('Flight Time (ms)')
    axes[1, 1].set_ylim(0, min(flight_95th * 1.2, 2000))  # Cap at 2000ms max
    axes[1, 1].grid(True, alpha=0.3)
    
    # Typing Speed plots
    axes[0, 2].hist(typing_speeds, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 2].set_title('Typing Speed Distribution')
    axes[0, 2].set_xlabel('Typing Speed (CPM)')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Typing Speed box plot
    axes[1, 2].boxplot(typing_speeds, vert=True)
    axes[1, 2].set_title('Typing Speed Box Plot')
    axes[1, 2].set_ylabel('Typing Speed (CPM)')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('keystroke_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print(f"\n=== DISTRIBUTION SUMMARY ===")
    print(f"\nDwell Time Statistics:")
    dwell_array = np.array(dwell_times)
    print(f"  Count: {len(dwell_array):,}")
    print(f"  Mean: {dwell_array.mean():.2f} ms")
    print(f"  Std: {dwell_array.std():.2f} ms")
    print(f"  Min: {dwell_array.min():.2f} ms")
    print(f"  Max: {dwell_array.max():.2f} ms")
    print(f"  Median: {np.median(dwell_array):.2f} ms")
    
    print(f"\nFlight Time Statistics:")
    flight_array = np.array(flight_times)
    print(f"  Count: {len(flight_array):,}")
    print(f"  Mean: {flight_array.mean():.2f} ms")
    print(f"  Std: {flight_array.std():.2f} ms")
    print(f"  Min: {flight_array.min():.2f} ms")
    print(f"  Max: {flight_array.max():.2f} ms")
    print(f"  Median: {np.median(flight_array):.2f} ms")
    
    print(f"\nTyping Speed Statistics:")
    speed_array = np.array(typing_speeds)
    print(f"  Count: {len(speed_array):,}")
    print(f"  Mean: {speed_array.mean():.2f} CPM")
    print(f"  Std: {speed_array.std():.2f} CPM")
    print(f"  Min: {speed_array.min():.2f} CPM")
    print(f"  Max: {speed_array.max():.2f} CPM")
    print(f"  Median: {np.median(speed_array):.2f} CPM")

if __name__ == "__main__":
    plot_distributions()