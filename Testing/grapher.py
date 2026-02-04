
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataPipeline.dataPrepper import dataPrepper

def computeOgStats(n=None):
    """Extract original keystroke statistics using dataPrepper pipeline."""
    base_dir = "../data"
    csv_dir = os.path.join(base_dir, "csv")
    
    ogDwell_times = []
    ogFlight_times = []
    ogTyping_speeds = []
    
    # Get list of CSV files and optionally limit the number
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith(".csv")]
    if n is not None:
        csv_files = csv_files[:n]
        print(f"Analyzing {len(csv_files)} samples (limited from total available)")
    else:
        print(f"Analyzing all {len(csv_files)} samples")
    
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
            
            dwell = processed_data[:, cont_idx[0]]
            flight = processed_data[:, cont_idx[1]]
            typing = processed_data[:, cont_idx[2]]
            
            # Set first FlightTime to NaN for consistency (no previous keystroke)
            if len(flight) > 0:
                flight = flight.copy()
                flight[0] = np.nan
            
            # Filter out NaNs for graphing (no NaNs in plots)
            dwell_clean = dwell[~np.isnan(dwell)]
            flight_clean = flight[~np.isnan(flight)]  # Removes first FlightTime NaN
            typing_clean = typing[~np.isnan(typing)]
            
            ogDwell_times.extend(dwell_clean.tolist())
            ogFlight_times.extend(flight_clean.tolist())
            ogTyping_speeds.extend(typing_clean.tolist())
            
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            continue
    
    # Debug FlightTime distribution to check for outliers
    flight_array = np.array(ogFlight_times)
    print(f"FlightTime distribution analysis:")
    print(f"  Mean: {flight_array.mean():.2f}, Std: {flight_array.std():.2f}")
    print(f"  Min: {flight_array.min():.2f}, Max: {flight_array.max():.2f}")
    print(f"  Percentiles: 50%={np.percentile(flight_array, 50):.1f}, 90%={np.percentile(flight_array, 90):.1f}, 95%={np.percentile(flight_array, 95):.1f}, 99%={np.percentile(flight_array, 99):.1f}, 99.9%={np.percentile(flight_array, 99.9):.1f}")
    print(f"  Values >10sec: {(flight_array > 10000).sum()}/{len(flight_array)} ({(flight_array > 10000).mean()*100:.1f}%)")
    print(f"  Values >30sec: {(flight_array > 30000).sum()}/{len(flight_array)} ({(flight_array > 30000).mean()*100:.1f}%)")
        
    return ogDwell_times, ogFlight_times, ogTyping_speeds

def plot_distributions(n=None):
    """Plot histograms and distribution plots for the original keystroke metrics."""
    print("Computing original statistics...")
    dwell_times, flight_times, typing_speeds = computeOgStats(n)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Add sample size to title
    sample_text = f" ({n} samples)" if n is not None else " (all samples)"
    fig.suptitle(f'Original Keystroke Data Distributions{sample_text}', fontsize=16, fontweight='bold')
    
    # Calculate reasonable x-axis limits (focus on 95th percentile to avoid extreme outliers)
    dwell_95th = np.percentile(dwell_times, 95)
    flight_95th = np.percentile(flight_times, 95)
    
    # Dwell Time plots
    axes[0, 0].hist(dwell_times, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Dwell Time Distribution')
    axes[0, 0].set_xlabel('Dwell Time (ms)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_xlim(0, dwell_95th)  # Larger range to show full distribution
    axes[0, 0].grid(True, alpha=0.3)
    
    # Dwell Time box plot
    axes[1, 0].boxplot(dwell_times, vert=True)
    axes[1, 0].set_title('Dwell Time Box Plot')
    axes[1, 0].set_ylabel('Dwell Time (ms)')
    axes[1, 0].set_ylim(0, dwell_95th)  # Larger range to show full distribution
    axes[1, 0].grid(True, alpha=0.3)
    
    # Flight Time plots
    axes[0, 1].hist(flight_times, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[0, 1].set_title('Flight Time Distribution')
    axes[0, 1].set_xlabel('Flight Time (ms)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_xlim(0, flight_95th)  # Larger range to show full distribution
    axes[0, 1].grid(True, alpha=0.3)
    
    # Flight Time box plot
    axes[1, 1].boxplot(flight_times, vert=True)
    axes[1, 1].set_title('Flight Time Box Plot')
    axes[1, 1].set_ylabel('Flight Time (ms)')
    axes[1, 1].set_ylim(0, flight_95th)  # Larger range to show full distribution
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

def plot_interactive_distributions():
    """Create an interactive plot with slider to control sample size."""
    from matplotlib.widgets import Slider
    
    # Get total number of samples
    csv_dir = os.path.join("data", "csv")
    total_samples = len([f for f in os.listdir(csv_dir) if f.endswith(".csv")])
    
    # Create figure and subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    plt.subplots_adjust(bottom=0.25)
    
    # Add slider
    ax_slider = plt.axes([0.2, 0.02, 0.5, 0.03])
    slider = Slider(ax_slider, 'Samples', 1, total_samples, valinit=10, valfmt='%d')
    
    def update_plots(val):
        n_samples = int(slider.val)
        
        # Clear previous plots
        for ax in axes:
            ax.clear()
        
        # Compute stats for current sample size
        dwell_times, flight_times, typing_speeds = computeOgStats(n_samples)
        
        # Calculate 95th percentiles for adaptive axis limits
        dwell_95th = np.percentile(dwell_times, 95)
        flight_95th = np.percentile(flight_times, 95)
        
        # Plot distributions
        axes[0].hist(dwell_times, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0].set_title(f'Dwell Time ({n_samples} samples)')
        axes[0].set_xlabel('Dwell Time (ms)')
        axes[0].set_xlim(0, dwell_95th)
        axes[0].grid(True, alpha=0.3)
        
        axes[1].hist(flight_times, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[1].set_title(f'Flight Time ({n_samples} samples)')
        axes[1].set_xlabel('Flight Time (ms)')
        axes[1].set_xlim(0, flight_95th)
        axes[1].grid(True, alpha=0.3)
        
        axes[2].hist(typing_speeds, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[2].set_title(f'Typing Speed ({n_samples} samples)')
        axes[2].set_xlabel('Typing Speed (CPM)')
        axes[2].grid(True, alpha=0.3)
        
        plt.draw()
    
    # Initial plot
    update_plots(10)
    
    # Connect slider to update function
    slider.on_changed(update_plots)
    
    plt.show()

if __name__ == "__main__":
    # Regular distribution plots
    plot_distributions()
    
    # Interactive plot with slider
    print("\\nStarting interactive plot...")
    plot_interactive_distributions()