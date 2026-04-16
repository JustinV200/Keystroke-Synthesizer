"""Visualization utilities for keystroke data analysis.

Provides histogram and box-plot comparisons of original vs. synthesized
keystroke distributions, as well as standalone original-data plots and an
interactive sample-size slider.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class grapher:
    """Plotting helper for keystroke distribution analysis."""
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
        dwell_95th = max(np.percentile(ogDwell_times, 95), np.percentile(synthDwell_times, 95))
        flight_95th = max(np.percentile(ogFlight_times, 95), np.percentile(synthFlight_times, 95))
        
        # Dwell Time plots
        axes[0, 0].hist(ogDwell_times, bins=50, alpha=0.7, color='skyblue', edgecolor='black', label='Original')
        axes[0, 0].hist(synthDwell_times, bins=50, alpha=0.7, color='salmon', edgecolor='black', label='Synthesized')
        axes[0, 0].set_title('Dwell Time Distribution')
        axes[0, 0].set_xlabel('Dwell Time (ms)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_xlim(0, dwell_95th)  # Larger range to show full distribution
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Dwell Time box plot
        axes[1, 0].boxplot([ogDwell_times, synthDwell_times], vert=True)
        axes[1, 0].set_title('Dwell Time Box Plot')
        axes[1, 0].set_ylabel('Dwell Time (ms)')
        axes[1, 0].set_xticklabels(['Original', 'Synthesized'])
        axes[1, 0].set_ylim(0, dwell_95th)  # Larger range to show full distribution
        axes[1, 0].grid(True, alpha=0.3)
        # Flight Time plots
        axes[0, 1].hist(ogFlight_times, bins=50, alpha=0.7, color='lightcoral', edgecolor='black', label='Original')
        axes[0, 1].hist(synthFlight_times, bins=50, alpha=0.7, color='lightblue', edgecolor='black', label='Synthesized')
        axes[0, 1].set_title('Flight Time Distribution')
        axes[0, 1].set_xlabel('Flight Time (ms)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_xlim(0, flight_95th)  # Larger range to show full distribution
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        # Flight Time box plot
        axes[1, 1].boxplot([ogFlight_times, synthFlight_times], vert=True)
        axes[1, 1].set_title('Flight Time Box Plot')
        axes[1, 1].set_ylabel('Flight Time (ms)')
        axes[1, 1].set_xticklabels(['Original', 'Synthesized'])
        axes[1, 1].set_ylim(0, flight_95th)  # Larger range to show full distribution
        axes[1, 1].grid(True, alpha=0.3)
        # Typing Speed plots
        axes[0, 2].hist(ogTyping_speeds, bins=50, alpha=0.7, color='lightgreen', edgecolor='black', label='Original')
        axes[0, 2].hist(synthTyping_speeds, bins=50, alpha=0.7, color='lightyellow', edgecolor='black', label='Synthesized')
        axes[0, 2].set_title('Typing Speed Distribution')
        axes[0, 2].set_xlabel('Typing Speed (CPM)')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].legend()
        # Typing Speed box plot
        axes[1, 2].boxplot([ogTyping_speeds, synthTyping_speeds], vert=True)
        axes[1, 2].set_title('Typing Speed Box Plot')
        axes[1, 2].set_ylabel('Typing Speed (CPM)')
        axes[1, 2].set_xticklabels(['Original', 'Synthesized'])
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('comparison_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
    def ogDataplots(self, n=None):
        """Plot histograms and box plots for original keystroke metrics.

        Args:
            n (int | None): Number of CSV samples to process.  If None, all
                available samples are used.
        """
        print("Computing original statistics...")
        from accuracyTester import computeOgStats
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

    def perCharPlots(self, og_df, synth_df, top_n=40):
        """Side-by-side bar charts comparing per-character mean DwellTime and typing_speed.

        Args:
            og_df (pd.DataFrame): Original data with char, prev_char, DwellTime, FlightTime, typing_speed.
            synth_df (pd.DataFrame): Synthesized data with same columns.
            top_n (int): Number of most frequent characters to plot.
        """
        def format_char(c):
            if c == ' ':
                return '⎵'
            elif c == '\n':
                return '↵'
            elif c == '\t':
                return '⇥'
            return c

        # Get top_n most frequent characters from original data
        og_char_counts = og_df['char'].value_counts()
        top_chars = og_char_counts.head(top_n).index.tolist()

        og_filtered = og_df[og_df['char'].isin(top_chars)]
        synth_filtered = synth_df[synth_df['char'].isin(top_chars)]

        chars_sorted = sorted(top_chars, key=lambda c: og_char_counts.get(c, 0), reverse=True)
        labels = [format_char(c) for c in chars_sorted]

        fig, axes = plt.subplots(2, 1, figsize=(20, 12))
        fig.suptitle(f'Per-Character Comparison: Original vs Synthesized (Top {top_n} chars by frequency)',
                     fontsize=16, fontweight='bold')

        x = np.arange(len(chars_sorted))
        width = 0.35

        for ax, metric, ylabel in zip(axes, ['DwellTime', 'typing_speed'],
                                       ['Dwell Time (ms)', 'Typing Speed (CPM)']):
            og_means = og_filtered.groupby('char')[metric].mean()
            synth_means = synth_filtered.groupby('char')[metric].mean()

            og_vals = [og_means.get(c, 0) for c in chars_sorted]
            synth_vals = [synth_means.get(c, 0) for c in chars_sorted]

            ax.bar(x - width / 2, og_vals, width, label='Original', color='skyblue', edgecolor='black', linewidth=0.5)
            ax.bar(x + width / 2, synth_vals, width, label='Synthesized', color='salmon', edgecolor='black', linewidth=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=10)
            ax.set_ylabel(ylabel)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig('per_char_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Saved per_char_comparison.png")

    def charPairPlots(self, og_df, synth_df, top_n=100):
        """Horizontal bar chart comparing mean FlightTime for the top N character pairs.

        Args:
            og_df (pd.DataFrame): Original data with char, prev_char, FlightTime columns.
            synth_df (pd.DataFrame): Synthesized data with same columns.
            top_n (int): Number of most frequent character pairs to plot.
        """
        def format_pair(p):
            return p.replace(' ', '⎵').replace('\n', '↵').replace('\t', '⇥')

        # Drop rows without valid FlightTime or prev_char
        og_pairs = og_df.dropna(subset=['FlightTime']).copy()
        og_pairs = og_pairs[og_pairs['prev_char'] != '']
        synth_pairs = synth_df.dropna(subset=['FlightTime']).copy()
        synth_pairs = synth_pairs[synth_pairs['prev_char'] != '']

        # Build pair labels
        og_pairs['pair'] = og_pairs['prev_char'] + '→' + og_pairs['char']
        synth_pairs['pair'] = synth_pairs['prev_char'] + '→' + synth_pairs['char']

        # Top N most frequent pairs from original data
        pair_counts = og_pairs['pair'].value_counts()
        top_pairs = pair_counts.head(top_n).index.tolist()

        og_means = og_pairs[og_pairs['pair'].isin(top_pairs)].groupby('pair')['FlightTime'].mean()
        synth_means = synth_pairs[synth_pairs['pair'].isin(top_pairs)].groupby('pair')['FlightTime'].mean()

        # Sort by original frequency, keep only pairs present in both
        pairs_sorted = [p for p in top_pairs if p in og_means.index]
        og_vals = [og_means.get(p, 0) for p in pairs_sorted]
        synth_vals = [synth_means.get(p, 0) for p in pairs_sorted]
        labels = [format_pair(p) for p in pairs_sorted]

        y = np.arange(len(pairs_sorted))
        height = 0.35

        fig, ax = plt.subplots(figsize=(14, max(20, len(pairs_sorted) * 0.3)))
        fig.suptitle(f'Per-Pair Mean FlightTime: Original vs Synthesized (Top {top_n} pairs)',
                     fontsize=14, fontweight='bold')

        ax.barh(y + height / 2, og_vals, height, label='Original', color='skyblue', edgecolor='black', linewidth=0.3)
        ax.barh(y - height / 2, synth_vals, height, label='Synthesized', color='salmon', edgecolor='black', linewidth=0.3)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8, fontfamily='monospace')
        ax.set_xlabel('Mean FlightTime (ms)')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3, axis='x')
        ax.invert_yaxis()  # Most frequent pair at top

        plt.tight_layout()
        plt.savefig('per_pair_flighttime.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Saved per_pair_flighttime.png")

def plot_interactive_distributions():
    """Launch an interactive matplotlib plot with a slider to control sample size."""
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
