"""
Simplified Plot Generator for DQN Training Analysis
Generates clean plots with only moving averages for all difficulty levels
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import glob

# Set clean style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class CleanPlotGenerator:
    def __init__(self, logs_dir="logs", plots_dir="plots"):
        """
        Initialize clean plot generator
        
        Args:
            logs_dir: Directory containing CSV log files
            plots_dir: Directory to save generated plots
        """
        self.logs_dir = Path(logs_dir)
        self.plots_dir = Path(plots_dir)
        self.plots_dir.mkdir(exist_ok=True)
        
        # Load all available level data
        self.data = {}
        self.levels = []
        
        # Define clean color scheme for levels
        self.level_colors = {
            'easy': '#2E86AB',    # Blue
            'medium': '#A23B72',   # Purple  
            'hard': '#F18F01'      # Orange
        }
        
        self.load_all_data()
    
    def load_all_data(self):
        """Load all CSV files from logs directory"""
        csv_files = glob.glob(str(self.logs_dir / "dqn_training_*.csv"))
        
        if not csv_files:
            print(f"No CSV files found in {self.logs_dir}")
            print("Expected files: dqn_training_easy.csv, dqn_training_medium.csv, dqn_training_hard.csv")
            return
        
        for csv_file in csv_files:
            level = Path(csv_file).stem.replace('dqn_training_', '')
            try:
                df = pd.read_csv(csv_file)
                self.data[level] = df
                self.levels.append(level)
                print(f"✓ Loaded {len(df)} episodes for '{level}' level")
            except Exception as e:
                print(f"✗ Error loading {csv_file}: {e}")
        
        print(f"\nTotal levels loaded: {len(self.levels)}")
    
    def plot_rewards_vs_episodes(self):
        """
        Plot 1: Clean rewards vs episodes with moving averages only
        """
        plt.figure(figsize=(12, 6))
        
        for level in self.levels:
            df = self.data[level]
            color = self.level_colors.get(level, 'gray')
            
            # Use 100-episode moving average
            if 'avg_reward_100' in df.columns:
                reward_avg = df['avg_reward_100']
            else:
                reward_avg = df['total_reward'].rolling(window=100, min_periods=1).mean()
            
            # Plot smooth moving average line
            plt.plot(df['episode'], reward_avg, 
                    linewidth=3, color=color, 
                    label=f'{level.capitalize()} Level', 
                    marker='', alpha=0.9)
        
        plt.title('Rewards vs Episodes (100-Episode Moving Average)', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Total Reward', fontsize=12)
        plt.legend(loc='upper left', fontsize=11, frameon=True, fancybox=True, shadow=True)
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # Add horizontal line at reward=0
        plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3, linewidth=1)
        
        # Add subtle background
        plt.gca().set_facecolor('#f8f9fa')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        
        plt.tight_layout()
        plot_path = self.plots_dir / "rewards_vs_episodes.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {plot_path}")
    
    def plot_steps_vs_episodes(self):
        """
        Plot 2: Clean steps vs episodes with moving averages only
        """
        plt.figure(figsize=(12, 6))
        
        for level in self.levels:
            df = self.data[level]
            color = self.level_colors.get(level, 'gray')
            
            # Calculate 50-episode moving average of steps
            steps_avg = df['steps_taken'].rolling(window=50, min_periods=1).mean()
            
            # Plot smooth line
            plt.plot(df['episode'], steps_avg, 
                    linewidth=3, color=color,
                    label=f'{level.capitalize()} Level',
                    marker='', alpha=0.9)
        
        plt.title('Steps Taken vs Episodes (50-Episode Moving Average)', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Steps Taken', fontsize=12)
        plt.legend(loc='upper right', fontsize=11, frameon=True, fancybox=True, shadow=True)
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # Add subtle background
        plt.gca().set_facecolor('#f8f9fa')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        
        plt.tight_layout()
        plot_path = self.plots_dir / "steps_vs_episodes.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {plot_path}")
    
    def plot_traps_vs_treasures(self):
        """
        Plot 3: Clean traps vs treasures with moving averages only
        """
        plt.figure(figsize=(12, 6))
        
        for level in self.levels:
            df = self.data[level]
            color = self.level_colors.get(level, 'gray')
            
            # Calculate efficiency: treasures per wall hit
            # Avoid division by zero
            df['efficiency'] = df['treasures_collected'] / (df['walls_hit'] + 1)
            
            # Calculate 50-episode moving average of efficiency
            efficiency_avg = df['efficiency'].rolling(window=50, min_periods=1).mean()
            
            # Plot smooth line
            plt.plot(df['episode'], efficiency_avg, 
                    linewidth=3, color=color,
                    label=f'{level.capitalize()} Level',
                    marker='', alpha=0.9)
        
        plt.title('Treasures per Wall Hit (50-Episode Moving Average)', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Treasures per Wall Hit', fontsize=12)
        plt.legend(loc='upper left', fontsize=11, frameon=True, fancybox=True, shadow=True)
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # Add subtle background
        plt.gca().set_facecolor('#f8f9fa')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        
        plt.tight_layout()
        plot_path = self.plots_dir / "traps_vs_treasures.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {plot_path}")
    
    def plot_combined_performance(self):
        """
        Additional Plot: Combined performance comparison
        Shows key metrics in one clean plot
        """
        plt.figure(figsize=(14, 8))
        
        metrics = [
            ('total_reward', 'Reward', 'avg_reward_100'),
            ('treasures_collected', 'Treasures', 'avg_treasures_100'),
            ('final_score', 'Score', None),
            ('hint_accuracy', 'Accuracy (%)', None)
        ]
        
        # Create subplots for each metric
        n_metrics = len(metrics)
        for idx, (col, name, avg_col) in enumerate(metrics):
            plt.subplot(2, 2, idx + 1)
            
            for level in self.levels:
                df = self.data[level]
                color = self.level_colors.get(level, 'gray')
                
                # Use average column if available, otherwise calculate
                if avg_col and avg_col in df.columns:
                    metric_avg = df[avg_col]
                else:
                    metric_avg = df[col].rolling(window=50, min_periods=1).mean()
                
                # Plot clean line
                plt.plot(df['episode'], metric_avg, 
                        linewidth=2, color=color,
                        label=f'{level.capitalize()}')
            
            plt.title(f'{name} (Moving Average)', fontsize=12, fontweight='bold')
            plt.xlabel('Episode')
            plt.ylabel(name)
            
            if idx == 0:  # Only show legend in first subplot
                plt.legend(loc='upper left', fontsize=9, frameon=True)
            
            plt.grid(True, alpha=0.2, linestyle='--')
            plt.gca().set_facecolor('#f8f9fa')
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
        
        plt.suptitle('Performance Metrics Across All Levels', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        plot_path = self.plots_dir / "combined_performance.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {plot_path}")
    
    def plot_final_scores_comparison(self):
        """
        Additional Plot: Bar chart comparing final scores across levels
        """
        plt.figure(figsize=(10, 6))
        
        level_names = []
        avg_scores = []
        score_stds = []
        
        for level in self.levels:
            df = self.data[level]
            level_names.append(level.capitalize())
            avg_scores.append(df['final_score'].mean())
            score_stds.append(df['final_score'].std())
        
        colors = [self.level_colors[l.lower()] for l in level_names]
        x_pos = np.arange(len(level_names))
        
        # Create bar chart
        bars = plt.bar(x_pos, avg_scores, yerr=score_stds,
                     capsize=10, alpha=0.8, color=colors,
                     edgecolor='black', linewidth=1.5)
        
        plt.title('Average Final Score by Difficulty Level', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Difficulty Level', fontsize=12)
        plt.ylabel('Average Final Score', fontsize=12)
        plt.xticks(x_pos, level_names)
        plt.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # Add value labels on bars
        for bar, value, std_val in zip(bars, avg_scores, score_stds):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{value:.1f} ± {std_val:.1f}', 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Add subtle background
        plt.gca().set_facecolor('#f8f9fa')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        
        plt.tight_layout()
        plot_path = self.plots_dir / "final_scores_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {plot_path}")
    
    def plot_epsilon_decay(self):
        """
        Additional Plot: Epsilon decay comparison
        """
        plt.figure(figsize=(12, 6))
        
        for level in self.levels:
            df = self.data[level]
            color = self.level_colors.get(level, 'gray')
            
            if 'epsilon' in df.columns:
                # Plot epsilon decay
                plt.plot(df['episode'], df['epsilon'], 
                        linewidth=2.5, color=color,
                        label=f'{level.capitalize()} Level',
                        alpha=0.8)
        
        plt.title('Epsilon Decay (Exploration Rate) Over Episodes', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Epsilon (Exploration Rate)', fontsize=12)
        plt.legend(loc='upper right', fontsize=11, frameon=True, fancybox=True, shadow=True)
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # Use logarithmic scale for better visualization
        plt.yscale('log')
        
        # Add subtle background
        plt.gca().set_facecolor('#f8f9fa')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        
        plt.tight_layout()
        plot_path = self.plots_dir / "epsilon_decay.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {plot_path}")
    
    def create_all_plots(self):
        """Generate all clean plots"""
        print("\n" + "="*60)
        print("   CLEAN PLOT GENERATION - ALL LEVELS")
        print("="*60 + "\n")
        
        if not self.data:
            print("❌ No data loaded. Check your CSV files in the 'logs' directory.")
            return
        
        # Generate three main required plots
        self.plot_rewards_vs_episodes()
        self.plot_steps_vs_episodes()
        self.plot_traps_vs_treasures()
        
        # Generate additional useful plots
        self.plot_combined_performance()
        self.plot_final_scores_comparison()
        self.plot_epsilon_decay()
        
        print("\n" + "="*60)
        print("SUMMARY:")
        print(f"✓ Main plots saved to: {self.plots_dir.absolute()}")
        print("\nGenerated Plots:")
        print("  1. rewards_vs_episodes.png")
        print("  2. steps_vs_episodes.png")
        print("  3. traps_vs_treasures.png")
        print("  4. combined_performance.png")
        print("  5. final_scores_comparison.png")
        print("  6. epsilon_decay.png")
        print("="*60)

def main():
    """Main function to run the clean plot generator"""
    # Set paths
    logs_dir = "logs"
    plots_dir = "plots"
    
    # Check if logs directory exists
    if not Path(logs_dir).exists():
        print(f"⚠️  Directory '{logs_dir}' not found.")
        print("\nCreating directory structure...")
        Path(logs_dir).mkdir(exist_ok=True)
        Path(plots_dir).mkdir(exist_ok=True)
        
        print(f"\nPlease place your CSV files in '{logs_dir}/':")
        print(f"  - {logs_dir}/dqn_training_easy.csv")
        print(f"  - {logs_dir}/dqn_training_medium.csv")
        print(f"  - {logs_dir}/dqn_training_hard.csv")
        print("\nThen run this script again.")
        return
    
    # Check for CSV files
    csv_files = list(Path(logs_dir).glob("dqn_training_*.csv"))
    if not csv_files:
        print(f"❌ No CSV files found in '{logs_dir}'")
        print("\nExpected files:")
        print("  - dqn_training_easy.csv")
        print("  - dqn_training_medium.csv")
        print("  - dqn_training_hard.csv")
        print("\nMake sure your files are named correctly.")
        return
    
    # Initialize and run plot generator
    try:
        plotter = CleanPlotGenerator(logs_dir=logs_dir, plots_dir=plots_dir)
        plotter.create_all_plots()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nCheck that your CSV files have the required columns:")
        print("  - episode, level, total_reward, steps_taken, treasures_collected")
        print("  - final_score, lives_remaining, epsilon, walls_hit")

if __name__ == "__main__":
    main()