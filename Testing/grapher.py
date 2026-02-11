
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from accuracyTester import computeOgStats
# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class grapher:
    def __init__(self):
        pass

    def comparisonPlots(self, synthDwell_times, synthFlight_times, synthTyping_speeds, ogDwell_times, ogFlight_times, ogTyping_speeds):
        """Create side-by-side comparison plots for original vs synthesized data."""
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Add sample size to title
        fig.suptitle(f'Original vs Synthesized Keystroke Data Distributions', fontsize=16, fontweight='bold')
        
        # Calculate reasonable x-axis limits (focus on 95th percentile to avoid extreme outliers)
        dwell_95th = max(np.percentile(self.ogDwell_times, 95), np.percentile(synthDwell_times, 95))
        flight_95th = max(np.percentile(self.ogFlight_times, 95), np.percentile(synthFlight_times, 95))
        
        # Dwell Time plots
        axes[0, 0].hist(self.ogDwell_times, bins=50, alpha=0.7, color='skyblue', edgecolor='black', label='Original')
        axes[0, 0].hist(synthDwell_times, bins=50, alpha=0.7, color='salmon', edgecolor='black', label='Synthesized')
        axes[0, 0].set_title('Dwell Time Distribution')
        axes[0, 0].set_xlabel('Dwell Time (ms)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_xlim(0, dwell_95th)  # Larger range to show full distribution
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Dwell Time box plot
        axes[1, 0].boxplot([self.ogDwell_times, synthDwell_times], vert=True)
        axes[1, 0].set_title('Dwell Time Box Plot')
        axes[1, 0].set_ylabel('Dwell Time (ms)')
        axes[1, 0].set_xticklabels(['Original', 'Synthesized'])
        axes[1, 0].set_ylim(0, dwell_95th)  # Larger range to show full distribution
        axes[1, 0].grid(True, alpha=0.3)
        # Flight Time plots
        axes[0, 1].hist(self.ogFlight_times, bins=50, alpha=0.7, color='lightcoral', edgecolor='black', label='Original')
        axes[0, 1].hist(synthFlight_times, bins=50, alpha=0.7, color='lightblue', edgecolor='black', label='Synthesized')
        axes[0, 1].set_title('Flight Time Distribution')
        axes[0, 1].set_xlabel('Flight Time (ms)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_xlim(0, flight_95th)  # Larger range to show full distribution
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        # Flight Time box plot
        axes[1, 1].boxplot([self.ogFlight_times, synthFlight_times], vert=True)
        axes[1, 1].set_title('Flight Time Box Plot')
        axes[1, 1].set_ylabel('Flight Time (ms)')
        axes[1, 1].set_xticklabels(['Original', 'Synthesized'])
        axes[1, 1].set_ylim(0, flight_95th)  # Larger range to show full distribution
        axes[1, 1].grid(True, alpha=0.3)
        # Typing Speed plots
        axes[0, 2].hist(self.ogTyping_speeds, bins=50, alpha=0.7, color='lightgreen', edgecolor='black', label='Original')
        axes[0, 2].hist(synthTyping_speeds, bins=50, alpha=0.7, color='lightyellow', edgecolor='black', label='Synthesized')
        axes[0, 2].set_title('Typing Speed Distribution')
        axes[0, 2].set_xlabel('Typing Speed (CPM)')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].legend()
        # Typing Speed box plot
        axes[1, 2].boxplot([self.ogTyping_speeds, synthTyping_speeds], vert=True)
        axes[1, 2].set_title('Typing Speed Box Plot')
        axes[1, 2].set_ylabel('Typing Speed (CPM)')
        axes[1, 2].set_xticklabels(['Original', 'Synthesized'])
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('comparison_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
    def ogDataplots(self, n=None):
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
