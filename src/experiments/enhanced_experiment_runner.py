"""
Enhanced Experiment Runner Module with Statistical Analysis and Schema Entropy

This module provides experiment runner with statistical validation,
error bars, confidence intervals, and enhanced schema entropy analysis.
"""

import os
import json
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set plotting style
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    try:
        plt.style.use('seaborn-whitegrid')
    except:
        plt.style.use('default')
        plt.grid(True)

sns.set_context("paper", font_scale=1.5)
sns.set_palette("deep")


class EnhancedExperimentRunner:
    """Enhanced experiment runner with statistical analysis including schema entropy."""

    def __init__(self, output_dir: str = ".", n_runs: int = 10):
        """
        Initialize the enhanced experiment runner.

        Args:
            output_dir: Directory to save results and figures
            n_runs: Number of runs for statistical analysis
        """
        self.output_dir = output_dir
        self.n_runs = n_runs
        self.results_dir = os.path.join(output_dir, "results")
        self.figures_dir = os.path.join(output_dir, "figures")
        self.tables_dir = os.path.join(output_dir, "tables")

        # Create directories
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)
        os.makedirs(self.tables_dir, exist_ok=True)

        # Initialize results storage
        self.results = {
            "channel_capacity": [],
            "bottleneck_analysis": [],
            "error_propagation": [],
            "schema_entropy": []
        }

        logger.info(f"Enhanced experiment runner initialized with {n_runs} runs per experiment")

    def run_experiment(self, experiment_name: str):
        """
        Run a specific experiment with statistical analysis.

        Args:
            experiment_name: Name of the experiment to run
        """
        if experiment_name == "channel_capacity":
            from experiments.enhanced_channel_capacity import run_channel_capacity_experiment_with_statistics
            results = run_channel_capacity_experiment_with_statistics(
                self.results_dir, n_runs=self.n_runs
            )
            self.results["channel_capacity"] = results

        elif experiment_name == "bottleneck":
            from experiments.enhanced_bottleneck_analysis import run_bottleneck_analysis_experiment_with_statistics
            results = run_bottleneck_analysis_experiment_with_statistics(
                self.results_dir, n_runs=self.n_runs
            )
            self.results["bottleneck_analysis"] = results

        elif experiment_name == "error_propagation":
            from experiments.enhanced_error_propagation import run_error_propagation_experiment_with_statistics
            results = run_error_propagation_experiment_with_statistics(
                self.results_dir, n_runs=self.n_runs
            )
            self.results["error_propagation"] = results

        elif experiment_name == "schema_entropy":
            # Use the enhanced schema entropy experiment with statistical validation
            from experiments.enhanced_schema_entropy import run_schema_entropy_experiment_with_statistics
            results = run_schema_entropy_experiment_with_statistics(
                self.results_dir, n_runs=self.n_runs
            )
            self.results["schema_entropy"] = results

        elif experiment_name == "all":
            logger.info("Running all experiments with statistical analysis")
            self.run_experiment("channel_capacity")
            self.run_experiment("bottleneck")
            self.run_experiment("error_propagation")
            self.run_experiment("schema_entropy")

        else:
            logger.error(f"Unknown experiment: {experiment_name}")

        # Generate enhanced figures and tables
        self.generate_figures()
        self.generate_tables()

    def generate_figures(self):
        """Generate figures with error bars and confidence intervals."""
        logger.info("Generating enhanced figures with statistical information")

        # Enhanced Figure 1: Channel Capacity with Error Bars (4 separate figures)
        self._generate_enhanced_channel_capacity_figures()

        # Enhanced Figure 2: Bottleneck Analysis with Significance (1 figure)
        self._generate_enhanced_bottleneck_figure()

        # Enhanced Figure 3: Error Propagation with Statistical Tests (2 separate figures)
        self._generate_enhanced_error_propagation_figures()

        # Enhanced Figure 4: Schema Entropy with Statistical Validation (2 separate figures)
        self._generate_enhanced_schema_entropy_figures()

        logger.info("Enhanced figure generation completed")

    def _generate_enhanced_channel_capacity_figures(self):
        """Generate separate channel capacity figures with error bars."""
        if not self.results["channel_capacity"]:
            logger.warning("No channel capacity results to plot")
            return

        df = pd.DataFrame(self.results["channel_capacity"])

        # Figure 1a: C1 Query Encoding Channel
        c1_data = df[df["channel"] == "C1"]
        if not c1_data.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            x_vals = c1_data["embedding_dim"]
            y_vals = c1_data["mutual_information"]
            y_errs = c1_data.get("mutual_information_std", [0] * len(y_vals))

            ax.errorbar(x_vals, y_vals, yerr=y_errs, marker='o', capsize=5, capthick=2)
            ax.set_title("C1: Query Encoding Channel")
            ax.set_xlabel("Embedding Dimension")
            ax.set_ylabel("Mutual Information (bits)")
            ax.set_xscale("log")
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(self.figures_dir, "enhanced_channel_capacity_a.png"), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(self.figures_dir, "enhanced_channel_capacity_a.pdf"), bbox_inches='tight')
            plt.close()

        # Figure 1b: C2 Retrieval Channel
        c2_data = df[df["channel"] == "C2"]
        if not c2_data.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            # Remove duplicates and sort by context_size
            c2_clean = c2_data.drop_duplicates(subset=['context_size']).sort_values('context_size')
            x_vals = c2_clean["context_size"]
            y_vals = c2_clean["mutual_information"]
            y_errs = c2_clean.get("mutual_information_std", [0] * len(y_vals))

            ax.errorbar(x_vals, y_vals, yerr=y_errs, marker='o', capsize=5, capthick=2)
            ax.set_title("C2: Retrieval Channel")
            ax.set_xlabel("Context Size")
            ax.set_ylabel("Mutual Information (bits)")
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(self.figures_dir, "enhanced_channel_capacity_b.png"), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(self.figures_dir, "enhanced_channel_capacity_b.pdf"), bbox_inches='tight')
            plt.close()

        # Figure 1c: C3 Context Integration Channel
        c3_data = df[df["channel"] == "C3"]
        if not c3_data.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            # Remove duplicates and sort by context_size
            c3_clean = c3_data.drop_duplicates(subset=['context_size']).sort_values('context_size')
            x_vals = c3_clean["context_size"]
            y_vals = c3_clean["mutual_information"]
            y_errs = c3_clean.get("mutual_information_std", [0] * len(y_vals))

            ax.errorbar(x_vals, y_vals, yerr=y_errs, marker='o', capsize=5, capthick=2)
            ax.set_title("C3: Context Integration Channel")
            ax.set_xlabel("Context Size")
            ax.set_ylabel("Mutual Information (bits)")
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(self.figures_dir, "enhanced_channel_capacity_c.png"), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(self.figures_dir, "enhanced_channel_capacity_c.pdf"), bbox_inches='tight')
            plt.close()

        # Figure 1d: C4 Generation Channel
        c4_data = df[df["channel"] == "C4"]
        if not c4_data.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            # Remove duplicates and sort by temperature
            c4_clean = c4_data.drop_duplicates(subset=['temperature']).sort_values('temperature')
            x_vals = c4_clean["temperature"]
            y_vals = c4_clean["mutual_information"]
            y_errs = c4_clean.get("mutual_information_std", [0] * len(y_vals))

            ax.errorbar(x_vals, y_vals, yerr=y_errs, marker='o', capsize=5, capthick=2)
            ax.set_title("C4: Generation Channel")
            ax.set_xlabel("Temperature")
            ax.set_ylabel("Mutual Information (bits)")
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(self.figures_dir, "enhanced_channel_capacity_d.png"), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(self.figures_dir, "enhanced_channel_capacity_d.pdf"), bbox_inches='tight')
            plt.close()

    def _generate_enhanced_bottleneck_figure(self):
        """Generate enhanced bottleneck figure with significance indicators."""
        if not self.results["bottleneck_analysis"]:
            logger.warning("No bottleneck analysis results to plot")
            return

        df = pd.DataFrame(self.results["bottleneck_analysis"])

        # Create figure with significance indicators
        fig, ax = plt.subplots(figsize=(12, 8))

        # Create grouped bar plot
        channels = df['channel'].unique()
        improvement_factors = sorted(df['improvement_factor'].unique())

        x = np.arange(len(channels))
        width = 0.35

        # Define colors based on improvement factor only (lighter for lower, darker for higher)
        base_color = 'steelblue'
        factor_colors = {}

        if len(improvement_factors) == 2:
            # Assume first is 50% (1.5x) and second is 100% (2.0x) based on caption
            factor_colors[improvement_factors[0]] = 'lightsteelblue'  # lighter for 50%
            factor_colors[improvement_factors[1]] = 'steelblue'  # darker for 100%
        else:
            # Generate colors with increasing darkness
            colors = ['lightsteelblue', 'steelblue', 'darkblue', 'navy']
            for i, factor in enumerate(improvement_factors):
                factor_colors[factor] = colors[min(i, len(colors) - 1)]

        for i, factor in enumerate(improvement_factors):
            factor_data = df[df['improvement_factor'] == factor]

            means = []
            errors = []
            significances = []
            p_values = []

            for channel in channels:
                channel_data = factor_data[factor_data['channel'] == channel]
                if not channel_data.empty:
                    mean_val = channel_data['improvement_percent'].iloc[0]
                    std_val = channel_data.get('improvement_percent_std', pd.Series([0])).iloc[0]
                    is_sig = channel_data.get('significant', pd.Series([False])).iloc[0]
                    p_val = channel_data.get('p_value', pd.Series([1.0])).iloc[0]

                    means.append(mean_val)
                    errors.append(std_val)
                    significances.append(is_sig)
                    p_values.append(p_val)
                else:
                    means.append(0)
                    errors.append(0)
                    significances.append(False)
                    p_values.append(1.0)

            # Create bars with consistent color per improvement factor
            bars = ax.bar(x + i * width, means, width, yerr=errors,
                          label=f'{int((factor - 1) * 100)}% improvement',
                          capsize=5, color=factor_colors[factor], alpha=0.8)

            # Add significance stars
            for j, (bar, is_sig, p_val) in enumerate(zip(bars, significances, p_values)):
                if is_sig:
                    if p_val < 0.001:
                        star = '**'
                    elif p_val < 0.01:
                        star = '**'
                    else:
                        star = '*'

                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + errors[j] + max(means) * 0.02,
                            star, ha='center', va='bottom', fontweight='bold', fontsize=10)

        # Customize plot
        ax.set_xlabel('Channel', fontsize=12)
        ax.set_ylabel('Performance Improvement (%)', fontsize=12)
        ax.set_xticks(x + width * (len(improvement_factors) - 1) / 2)
        ax.set_xticklabels(channels, rotation=0, ha='center')

        # Simple legend showing improvement levels
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        # Add significance note with italicized p
        ax.text(0.02, 0.85, r'$*$ $\mathit{p}<0.05$, $**$ $\mathit{p}<0.001$',
                transform=ax.transAxes, va='top', fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, "enhanced_bottleneck_analysis.png"),
                    dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(self.figures_dir, "enhanced_bottleneck_analysis.pdf"),
                    bbox_inches='tight')
        plt.close()

        logger.info(
            f"Generated enhanced bottleneck analysis figure: {os.path.join(self.figures_dir, 'enhanced_bottleneck_analysis.png')}")

    def _generate_enhanced_bottleneck_figure_old2(self):
        """Generate enhanced bottleneck figure with significance indicators."""
        if not self.results["bottleneck_analysis"]:
            logger.warning("No bottleneck analysis results to plot")
            return

        df = pd.DataFrame(self.results["bottleneck_analysis"])

        # Create figure with significance indicators
        fig, ax = plt.subplots(figsize=(12, 8))

        # Create grouped bar plot
        channels = df['channel'].unique()
        improvement_factors = sorted(df['improvement_factor'].unique())

        x = np.arange(len(channels))
        width = 0.35

        for i, factor in enumerate(improvement_factors):
            factor_data = df[df['improvement_factor'] == factor]

            means = []
            errors = []
            colors = []

            for channel in channels:
                channel_data = factor_data[factor_data['channel'] == channel]
                if not channel_data.empty:
                    mean_val = channel_data['improvement_percent'].iloc[0]
                    std_val = channel_data.get('improvement_percent_std', pd.Series([0])).iloc[0]
                    is_sig = channel_data.get('significant', pd.Series([False])).iloc[0]

                    means.append(mean_val)
                    errors.append(std_val)
                    colors.append('darkblue' if is_sig else 'lightblue')
                else:
                    means.append(0)
                    errors.append(0)
                    colors.append('lightgray')

            bars = ax.bar(x + i * width, means, width, yerr=errors,
                          label=f'{factor}x improvement',
                          capsize=5, color=colors, alpha=0.8)

            # Add significance stars
            for j, (bar, channel) in enumerate(zip(bars, channels)):
                channel_data = factor_data[factor_data['channel'] == channel]
                if not channel_data.empty:
                    is_sig = channel_data.get('significant', pd.Series([False])).iloc[0]
                    p_val = channel_data.get('p_value', pd.Series([1.0])).iloc[0]

                    if is_sig:
                        star = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*'
                        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + errors[j] + 5,
                                star, ha='center', va='bottom', fontweight='bold')

        ax.set_xlabel('Channel')
        ax.set_ylabel('Performance Improvement (%)')
        #ax.set_title('Impact of Channel Improvements on End-to-End Performance\n(* p<0.05, ** p<0.01, *** p<0.001)')
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(channels)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, "enhanced_bottleneck_analysis.png"), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(self.figures_dir, "enhanced_bottleneck_analysis.pdf"), bbox_inches='tight')
        plt.close()

    def _generate_enhanced_error_propagation_figures(self):
        """Generate separate error propagation figures with statistical significance."""
        if not self.results["error_propagation"]:
            logger.warning("No error propagation results to plot")
            return

        df = pd.DataFrame(self.results["error_propagation"])

        # Figure 3a: Single Channel Errors with Error Bars
        single_df = df[df["error_type"] == "single"]
        if not single_df.empty:
            fig, ax = plt.subplots(figsize=(10, 6))

            for channel in single_df['channel'].unique():
                channel_data = single_df[single_df['channel'] == channel]

                x_vals = channel_data['error_level']
                y_vals = channel_data['accuracy_drop']
                y_errs = channel_data.get('accuracy_drop_std', [0] * len(y_vals))

                ax.errorbar(x_vals, y_vals, yerr=y_errs, marker='o',
                            label=channel, capsize=5, capthick=2)

            ax.set_title("Single Channel Errors")
            ax.set_xlabel("Error Level")
            ax.set_ylabel("Accuracy Drop")
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(self.figures_dir, "enhanced_error_propagation_a.png"), dpi=300,
                        bbox_inches='tight')
            plt.savefig(os.path.join(self.figures_dir, "enhanced_error_propagation_a.pdf"), bbox_inches='tight')
            plt.close()

        # Figure 3b: Multi-Channel Interaction Effects with Significance
        multi_df = df[df["error_type"] == "multi"]
        if not multi_df.empty:
            fig, ax = plt.subplots(figsize=(12, 6))

            channels = multi_df['channels']
            effects = multi_df['interaction_effect']
            errors = multi_df.get('interaction_effect_std', [0] * len(effects))
            significant = multi_df.get('significant', [False] * len(effects))

            # Color bars based on significance and effect direction
            colors = []
            for sig, effect in zip(significant, effects):
                if sig:
                    colors.append('red' if effect > 0 else 'blue')
                else:
                    colors.append('lightgray')

            bars = ax.bar(range(len(channels)), effects, yerr=errors,
                          capsize=5, color=colors, alpha=0.8)

            # Add significance indicators
            for i, (bar, sig, p_val) in enumerate(zip(bars, significant,
                                                      multi_df.get('p_value', [1.0] * len(bars)))):
                if sig:
                    star = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*'
                    height = bar.get_height()
                    try:
                        error_val = errors.iloc[i] if i < len(errors) else 0
                    except (IndexError, KeyError):
                        error_val = 0
                    y_pos = height + error_val + 0.002 if height >= 0 else height - error_val - 0.002
                    ax.text(bar.get_x() + bar.get_width() / 2, y_pos, star,
                            ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')

            #ax.set_title("Interaction Effects Between Channels\n(* p<0.05, ** p<0.01, *** p<0.001)")
            ax.set_xlabel("Channel Pair")
            ax.set_ylabel("Interaction Effect (Δ Accuracy)")
            ax.set_xticks(range(len(channels)))
            ax.set_xticklabels(channels, rotation=45)
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(self.figures_dir, "enhanced_error_propagation_b.png"), dpi=300,
                        bbox_inches='tight')
            plt.savefig(os.path.join(self.figures_dir, "enhanced_error_propagation_b.pdf"), bbox_inches='tight')
            plt.close()

    def _generate_enhanced_schema_entropy_figures(self):
        """Generate separate schema entropy figures with statistical validation and error bars."""
        if not self.results["schema_entropy"]:
            logger.warning("No schema entropy results to plot")
            return

        df = pd.DataFrame(self.results["schema_entropy"])

        # Figure 4a: Retrieval performance vs schema entropy with error bars
        fig, ax = plt.subplots(figsize=(10, 6))

        embedding_dims = sorted(df['embedding_dim'].unique())

        for dim in embedding_dims:
            dim_data = df[df['embedding_dim'] == dim]
            x_vals = dim_data['schema_entropy']
            y_vals = dim_data['recall']
            y_errs = dim_data.get('recall_std', [0] * len(y_vals))

            ax.errorbar(x_vals, y_vals, yerr=y_errs, marker='o', capsize=3,
                        label=f'dim={dim}', alpha=0.8)

        ax.set_title("Retrieval Recall vs Schema Entropy\n(with statistical error bars)")
        ax.set_xlabel("Schema Entropy (bits)")
        ax.set_ylabel("Retrieval Recall")
        ax.legend(title="Embedding Dim")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, "enhanced_schema_entropy_a.png"), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(self.figures_dir, "enhanced_schema_entropy_a.pdf"), bbox_inches='tight')
        plt.close()

        # Figure 4b: Theoretical vs actual performance with confidence regions
        fig, ax = plt.subplots(figsize=(10, 6))

        # Calculate efficiency ratios
        efficiency_ratios = df['efficiency_ratio']
        theoretical_bounds = df['theoretical_bound']

        # Color points by bound respect
        colors = ['green' if respected else 'red' for respected in df['bound_respected']]

        scatter = ax.scatter(theoretical_bounds, efficiency_ratios,
                             c=colors, alpha=0.7, s=50)

        # Add theoretical efficiency line (perfect efficiency = 1.0)
        ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, label='Perfect Efficiency')

        ax.set_title("Efficiency Ratio vs Theoretical Bound\n(Green: Bound Respected, Red: Violated)")
        ax.set_xlabel("Theoretical Bound: min(log$_2$(d), H(S))")
        ax.set_ylabel("Efficiency Ratio (Actual/Theoretical)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, "enhanced_schema_entropy_b.png"), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(self.figures_dir, "enhanced_schema_entropy_b.pdf"), bbox_inches='tight')
        plt.close()
    def generate_figures_old(self):
        """Generate figures with error bars and confidence intervals."""
        logger.info("Generating enhanced figures with statistical information")

        # Enhanced Figure 1: Channel Capacity with Error Bars
        self._generate_enhanced_channel_capacity_figure()

        # Enhanced Figure 2: Bottleneck Analysis with Significance
        self._generate_enhanced_bottleneck_figure()

        # Enhanced Figure 3: Error Propagation with Statistical Tests
        self._generate_enhanced_error_propagation_figure()

        # Enhanced Figure 4: Schema Entropy with Statistical Validation
        self._generate_enhanced_schema_entropy_figure()

        logger.info("Enhanced figure generation completed")

    def _generate_enhanced_channel_capacity_figure_old(self):
        """Generate enhanced channel capacity figure with error bars."""
        if not self.results["channel_capacity"]:
            logger.warning("No channel capacity results to plot")
            return

        df = pd.DataFrame(self.results["channel_capacity"])

        # Create figure with error bars
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # C1: Query Encoding Channel
        c1_data = df[df["channel"] == "C1"]
        if not c1_data.empty:
            ax = axes[0, 0]
            x_vals = c1_data["embedding_dim"]
            y_vals = c1_data["mutual_information"]
            y_errs = c1_data.get("mutual_information_std", [0] * len(y_vals))

            ax.errorbar(x_vals, y_vals, yerr=y_errs, marker='o', capsize=5, capthick=2)
            ax.set_title("C1: Query Encoding Channel")
            ax.set_xlabel("Embedding Dimension")
            ax.set_ylabel("Mutual Information (bits)")
            ax.set_xscale("log")
            ax.grid(True, alpha=0.3)

        # C2: Retrieval Channel
        c2_data = df[df["channel"] == "C2"]
        if not c2_data.empty:
            ax = axes[0, 1]
            # Remove duplicates and sort by context_size
            c2_clean = c2_data.drop_duplicates(subset=['context_size']).sort_values('context_size')
            x_vals = c2_clean["context_size"]
            y_vals = c2_clean["mutual_information"]
            y_errs = c2_clean.get("mutual_information_std", [0] * len(y_vals))

            ax.errorbar(x_vals, y_vals, yerr=y_errs, marker='o', capsize=5, capthick=2)
            ax.set_title("C2: Retrieval Channel")
            ax.set_xlabel("Context Size")
            ax.set_ylabel("Mutual Information (bits)")
            ax.grid(True, alpha=0.3)

        # C3: Context Integration Channel
        c3_data = df[df["channel"] == "C3"]
        if not c3_data.empty:
            ax = axes[1, 0]
            # Remove duplicates and sort by context_size
            c3_clean = c3_data.drop_duplicates(subset=['context_size']).sort_values('context_size')
            x_vals = c3_clean["context_size"]
            y_vals = c3_clean["mutual_information"]
            y_errs = c3_clean.get("mutual_information_std", [0] * len(y_vals))

            ax.errorbar(x_vals, y_vals, yerr=y_errs, marker='o', capsize=5, capthick=2)
            ax.set_title("C3: Context Integration Channel")
            ax.set_xlabel("Context Size")
            ax.set_ylabel("Mutual Information (bits)")
            ax.grid(True, alpha=0.3)

        # C4: Generation Channel - FIXED
        c4_data = df[df["channel"] == "C4"]
        if not c4_data.empty:
            ax = axes[1, 1]
            # Remove duplicates and sort by temperature - this was missing!
            c4_clean = c4_data.drop_duplicates(subset=['temperature']).sort_values('temperature')
            x_vals = c4_clean["temperature"]
            y_vals = c4_clean["mutual_information"]
            y_errs = c4_clean.get("mutual_information_std", [0] * len(y_vals))

            ax.errorbar(x_vals, y_vals, yerr=y_errs, marker='o', capsize=5, capthick=2)
            ax.set_title("C4: Generation Channel")
            ax.set_xlabel("Temperature")
            ax.set_ylabel("Mutual Information (bits)")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, "enhanced_channel_capacity.png"), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(self.figures_dir, "enhanced_channel_capacity.pdf"), bbox_inches='tight')
        plt.close()

    def _generate_enhanced_bottleneck_figure_old(self):
        """Generate enhanced bottleneck figure with significance indicators."""
        if not self.results["bottleneck_analysis"]:
            logger.warning("No bottleneck analysis results to plot")
            return

        df = pd.DataFrame(self.results["bottleneck_analysis"])

        # Create figure with significance indicators
        fig, ax = plt.subplots(figsize=(12, 8))

        # Create grouped bar plot
        channels = df['channel'].unique()
        improvement_factors = sorted(df['improvement_factor'].unique())

        x = np.arange(len(channels))
        width = 0.35

        for i, factor in enumerate(improvement_factors):
            factor_data = df[df['improvement_factor'] == factor]

            means = []
            errors = []
            colors = []

            for channel in channels:
                channel_data = factor_data[factor_data['channel'] == channel]
                if not channel_data.empty:
                    mean_val = channel_data['improvement_percent'].iloc[0]
                    std_val = channel_data.get('improvement_percent_std', pd.Series([0])).iloc[0]
                    is_sig = channel_data.get('significant', pd.Series([False])).iloc[0]

                    means.append(mean_val)
                    errors.append(std_val)
                    colors.append('darkblue' if is_sig else 'lightblue')
                else:
                    means.append(0)
                    errors.append(0)
                    colors.append('lightgray')

            bars = ax.bar(x + i * width, means, width, yerr=errors,
                          label=f'{factor}x improvement',
                          capsize=5, color=colors, alpha=0.8)

            # Add significance stars
            for j, (bar, channel) in enumerate(zip(bars, channels)):
                channel_data = factor_data[factor_data['channel'] == channel]
                if not channel_data.empty:
                    is_sig = channel_data.get('significant', pd.Series([False])).iloc[0]
                    p_val = channel_data.get('p_value', pd.Series([1.0])).iloc[0]

                    if is_sig:
                        star = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*'
                        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + errors[j] + 5,
                                star, ha='center', va='bottom', fontweight='bold')

        ax.set_xlabel('Channel')
        ax.set_ylabel('Performance Improvement (%)')
        ax.set_title('Impact of Channel Improvements on End-to-End Performance\n(* p<0.05, ** p<0.01, *** p<0.001)')
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(channels)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, "enhanced_bottleneck_analysis.png"), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(self.figures_dir, "enhanced_bottleneck_analysis.pdf"), bbox_inches='tight')
        plt.close()

    def _generate_enhanced_error_propagation_figure_old(self):
        """Generate enhanced error propagation figure with statistical significance."""
        if not self.results["error_propagation"]:
            logger.warning("No error propagation results to plot")
            return

        df = pd.DataFrame(self.results["error_propagation"])

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: Single Channel Errors with Error Bars
        single_df = df[df["error_type"] == "single"]
        if not single_df.empty:
            for channel in single_df['channel'].unique():
                channel_data = single_df[single_df['channel'] == channel]

                x_vals = channel_data['error_level']
                y_vals = channel_data['accuracy_drop']
                y_errs = channel_data.get('accuracy_drop_std', [0] * len(y_vals))

                # Color based on significance
                colors = ['red' if sig else 'gray' for sig in channel_data.get('significant', [False] * len(y_vals))]

                ax1.errorbar(x_vals, y_vals, yerr=y_errs, marker='o',
                             label=channel, capsize=5, capthick=2)

            ax1.set_title("Single Channel Errors")
            ax1.set_xlabel("Error Level")
            ax1.set_ylabel("Accuracy Drop")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        # Plot 2: Multi-Channel Interaction Effects with Significance
        multi_df = df[df["error_type"] == "multi"]
        if not multi_df.empty:
            channels = multi_df['channels']
            effects = multi_df['interaction_effect']
            errors = multi_df.get('interaction_effect_std', [0] * len(effects))
            significant = multi_df.get('significant', [False] * len(effects))

            # Color bars based on significance and effect direction
            colors = []
            for sig, effect in zip(significant, effects):
                if sig:
                    colors.append('red' if effect > 0 else 'blue')
                else:
                    colors.append('lightgray')

            bars = ax2.bar(range(len(channels)), effects, yerr=errors,
                           capsize=5, color=colors, alpha=0.8)

            # Add significance indicators
            for i, (bar, sig, p_val) in enumerate(zip(bars, significant,
                                                      multi_df.get('p_value', [1.0] * len(bars)))):
                if sig:
                    star = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*'
                    height = bar.get_height()
                    try:
                        error_val = errors.iloc[i] if i < len(errors) else 0
                    except (IndexError, KeyError):
                        error_val = 0
                    y_pos = height + error_val + 0.002 if height >= 0 else height - error_val - 0.002
                    ax2.text(bar.get_x() + bar.get_width() / 2, y_pos, star,
                             ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')

            ax2.set_title("Interaction Effects Between Channels\n(* p<0.05, ** p<0.01, *** p<0.001)")
            ax2.set_xlabel("Channel Pair")
            ax2.set_ylabel("Interaction Effect (Δ Accuracy)")
            ax2.set_xticks(range(len(channels)))
            ax2.set_xticklabels(channels, rotation=45)
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, "enhanced_error_propagation.png"), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(self.figures_dir, "enhanced_error_propagation.pdf"), bbox_inches='tight')
        plt.close()

    def _generate_enhanced_schema_entropy_figure_old(self):
        """Generate enhanced schema entropy figure with statistical validation and error bars."""
        if not self.results["schema_entropy"]:
            logger.warning("No schema entropy results to plot")
            return

        df = pd.DataFrame(self.results["schema_entropy"])

        # Create figure with the two best visualizations
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot 1: Retrieval performance vs schema entropy with error bars
        embedding_dims = sorted(df['embedding_dim'].unique())

        for dim in embedding_dims:
            dim_data = df[df['embedding_dim'] == dim]
            x_vals = dim_data['schema_entropy']
            y_vals = dim_data['recall']
            y_errs = dim_data.get('recall_std', [0] * len(y_vals))

            ax1.errorbar(x_vals, y_vals, yerr=y_errs, marker='o', capsize=3,
                         label=f'dim={dim}', alpha=0.8)

        ax1.set_title("Retrieval Recall vs Schema Entropy\n(with statistical error bars)")
        ax1.set_xlabel("Schema Entropy (bits)")
        ax1.set_ylabel("Retrieval Recall")
        ax1.legend(title="Embedding Dim")
        ax1.grid(True, alpha=0.3)

        # Plot 2: Theoretical vs actual performance with confidence regions
        # Calculate efficiency ratios
        efficiency_ratios = df['efficiency_ratio']
        theoretical_bounds = df['theoretical_bound']

        # Color points by bound respect
        colors = ['green' if respected else 'red' for respected in df['bound_respected']]

        scatter = ax2.scatter(theoretical_bounds, efficiency_ratios,
                              c=colors, alpha=0.7, s=50)

        # Add theoretical efficiency line (perfect efficiency = 1.0)
        ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, label='Perfect Efficiency')

        ax2.set_title("Efficiency Ratio vs Theoretical Bound\n(Green: Bound Respected, Red: Violated)")
        ax2.set_xlabel("Theoretical Bound: min(log2(d), H(S))")
        ax2.set_ylabel("Efficiency Ratio (Actual/Theoretical)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, "enhanced_schema_entropy.png"), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(self.figures_dir, "enhanced_schema_entropy.pdf"), bbox_inches='tight')
        plt.close()
    def _generate_enhanced_schema_entropy_figure_4(self):
        """Generate enhanced schema entropy figure with statistical validation and error bars."""
        if not self.results["schema_entropy"]:
            logger.warning("No schema entropy results to plot")
            return

        df = pd.DataFrame(self.results["schema_entropy"])

        # Create figure with enhanced statistical visualizations
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Plot 1: Retrieval performance vs schema entropy with error bars
        ax1 = axes[0, 0]
        embedding_dims = sorted(df['embedding_dim'].unique())

        for dim in embedding_dims:
            dim_data = df[df['embedding_dim'] == dim]
            x_vals = dim_data['schema_entropy']
            y_vals = dim_data['recall']
            y_errs = dim_data.get('recall_std', [0] * len(y_vals))

            ax1.errorbar(x_vals, y_vals, yerr=y_errs, marker='o', capsize=3,
                         label=f'dim={dim}', alpha=0.8)

        ax1.set_title("Retrieval Recall vs Schema Entropy\n(with statistical error bars)")
        ax1.set_xlabel("Schema Entropy (bits)")
        ax1.set_ylabel("Retrieval Recall")
        ax1.legend(title="Embedding Dim", bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Theoretical vs actual performance with confidence regions
        ax2 = axes[0, 1]

        # Calculate efficiency ratios
        efficiency_ratios = df['efficiency_ratio']
        theoretical_bounds = df['theoretical_bound']

        # Color points by bound respect
        colors = ['green' if respected else 'red' for respected in df['bound_respected']]

        scatter = ax2.scatter(theoretical_bounds, efficiency_ratios,
                              c=colors, alpha=0.7, s=50)

        # Add theoretical efficiency line (perfect efficiency = 1.0)
        ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, label='Perfect Efficiency')

        ax2.set_title("Efficiency Ratio vs Theoretical Bound\n(Green: Bound Respected, Red: Violated)")
        ax2.set_xlabel("Theoretical Bound: min(log$_2$(d), H(S))")
        ax2.set_ylabel("Efficiency Ratio (Actual/Theoretical)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Bound violation analysis
        ax3 = axes[1, 0]

        # Group by limiting factor
        df['limiting_factor'] = df.apply(
            lambda row: 'Dimension' if np.log2(row['embedding_dim']) < row['schema_entropy'] else 'Entropy',
            axis=1
        )

        # Box plot of efficiency ratios by limiting factor
        limiting_factors = ['Dimension', 'Entropy']
        efficiency_data = []
        labels = []

        for factor in limiting_factors:
            factor_data = df[df['limiting_factor'] == factor]
            if not factor_data.empty:
                efficiency_data.append(factor_data['efficiency_ratio'])
                labels.append(f'{factor}\n(n={len(factor_data)})')

        if efficiency_data:
            bp = ax3.boxplot(efficiency_data, labels=labels, patch_artist=True)

            # Color boxes
            colors = ['lightblue', 'lightcoral']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)

        ax3.set_title("Efficiency by Limiting Factor\n(Statistical Distribution)")
        ax3.set_ylabel("Efficiency Ratio")
        ax3.grid(True, alpha=0.3)

        # Plot 4: Statistical significance of bound violations
        ax4 = axes[1, 1]

        # Count significant violations
        violation_summary = df.groupby(['schema_entropy', 'embedding_dim']).agg({
            'bound_violation_rate': 'mean',
            'significantly_below_bound': 'any'
        }).reset_index()

        # Create heatmap of violation rates
        pivot_violations = violation_summary.pivot(index='schema_entropy',
                                                   columns='embedding_dim',
                                                   values='bound_violation_rate')

        im = ax4.imshow(pivot_violations.values, cmap='Reds', aspect='auto',
                        vmin=0, vmax=1)

        # Set ticks and labels
        ax4.set_xticks(range(len(pivot_violations.columns)))
        ax4.set_xticklabels(pivot_violations.columns)
        ax4.set_yticks(range(len(pivot_violations.index)))
        ax4.set_yticklabels([f'{val:.1f}' for val in pivot_violations.index])

        ax4.set_title("Bound Violation Rate Heatmap")
        ax4.set_xlabel("Embedding Dimension")
        ax4.set_ylabel("Schema Entropy (bits)")

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax4)
        cbar.set_label('Violation Rate')

        # Add text annotations for significant violations
        for i, entropy in enumerate(pivot_violations.index):
            for j, dim in enumerate(pivot_violations.columns):
                violation_rate = pivot_violations.iloc[i, j]
                if not np.isnan(violation_rate):
                    # Check if this condition has significant violations
                    condition_data = violation_summary[
                        (violation_summary['schema_entropy'] == entropy) &
                        (violation_summary['embedding_dim'] == dim)
                        ]

                    if not condition_data.empty and condition_data['significantly_below_bound'].iloc[0]:
                        ax4.text(j, i, '!', ha='center', va='center',
                                 color='white', fontweight='bold', fontsize=12)

        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, "enhanced_schema_entropy.png"), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(self.figures_dir, "enhanced_schema_entropy.pdf"), bbox_inches='tight')
        plt.close()

    def generate_tables(self):
        """Generate enhanced tables with statistical information."""
        logger.info("Generating enhanced tables with statistical data")

        # Enhanced statistical summary table
        self._generate_statistical_summary_table()

        # Enhanced channel capacity table
        self._generate_enhanced_channel_capacity_table()

        # Enhanced bottleneck analysis table
        self._generate_enhanced_bottleneck_table()

        # Enhanced schema entropy table
        self._generate_enhanced_schema_entropy_table()

        logger.info("Enhanced table generation completed")

    def _generate_statistical_summary_table(self):
        """Generate comprehensive statistical summary table."""
        summary_data = []

        # Add statistical information for each experiment
        for exp_name, results in self.results.items():
            if results and exp_name in ['channel_capacity', 'bottleneck_analysis', 'error_propagation',
                                        'schema_entropy']:
                df = pd.DataFrame(results)

                if exp_name == 'channel_capacity':
                    for _, row in df.iterrows():
                        summary_data.append({
                            'Experiment': 'Channel Capacity',
                            'Condition': f"{row['channel']} ({row.get('embedding_dim', row.get('context_size', row.get('temperature', 'N/A')))})",
                            'Mean': row['mutual_information'],
                            'Std': row.get('mutual_information_std', 'N/A'),
                            'CI_Lower': row.get('mutual_information_ci_lower', 'N/A'),
                            'CI_Upper': row.get('mutual_information_ci_upper', 'N/A'),
                            'N_Runs': row.get('n_runs', 'N/A')
                        })

                elif exp_name == 'bottleneck_analysis':
                    for _, row in df.iterrows():
                        summary_data.append({
                            'Experiment': 'Bottleneck Analysis',
                            'Condition': f"{row['channel']} ({row['improvement_factor']}x)",
                            'Mean': row['improvement_percent'],
                            'Std': row.get('improvement_percent_std', 'N/A'),
                            'P_Value': row.get('p_value', 'N/A'),
                            'Significant': row.get('significant', 'N/A'),
                            'Effect_Size': row.get('effect_size', 'N/A'),
                            'N_Runs': row.get('n_runs', 'N/A')
                        })

                elif exp_name == 'schema_entropy':
                    for _, row in df.iterrows():
                        summary_data.append({
                            'Experiment': 'Schema Entropy',
                            'Condition': f"H={row['schema_entropy']:.1f}, d={row['embedding_dim']}",
                            'Mean': row['recall'],
                            'Std': row.get('recall_std', 'N/A'),
                            'CI_Lower': row.get('recall_ci_lower', 'N/A'),
                            'CI_Upper': row.get('recall_ci_upper', 'N/A'),
                            'Bound_Respected': row.get('bound_respected', 'N/A'),
                            'Efficiency_Ratio': row.get('efficiency_ratio', 'N/A'),
                            'N_Runs': row.get('n_runs', 'N/A')
                        })

        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(os.path.join(self.tables_dir, "statistical_summary.csv"), index=False)

    def _generate_enhanced_channel_capacity_table(self):
        """Generate enhanced channel capacity table with statistics."""
        if not self.results["channel_capacity"]:
            return

        df = pd.DataFrame(self.results["channel_capacity"])

        # Format table with statistics
        enhanced_table = []
        for _, row in df.iterrows():
            enhanced_table.append({
                'Channel': row['channel'],
                'Parameter': row.get('embedding_dim', row.get('context_size', row.get('temperature', 'N/A'))),
                'Capacity_Mean': f"{row['mutual_information']:.3f}",
                'Capacity_Std': f"{row.get('mutual_information_std', 0):.3f}",
                '95%_CI': f"[{row.get('mutual_information_ci_lower', 0):.3f}, {row.get('mutual_information_ci_upper', 0):.3f}]",
                'N_Runs': row.get('n_runs', 'N/A')
            })

        enhanced_df = pd.DataFrame(enhanced_table)
        enhanced_df.to_csv(os.path.join(self.tables_dir, "enhanced_channel_capacity.csv"), index=False)

    def _generate_enhanced_bottleneck_table(self):
        """Generate enhanced bottleneck table with significance tests."""
        if not self.results["bottleneck_analysis"]:
            return

        df = pd.DataFrame(self.results["bottleneck_analysis"])

        # Format table with statistical tests
        enhanced_table = []
        for _, row in df.iterrows():
            enhanced_table.append({
                'Channel': row['channel'],
                'Improvement_Factor': f"{row['improvement_factor']:.1f}x",
                'Improvement_Mean': f"{row['improvement_percent']:.2f}%",
                'Improvement_Std': f"{row.get('improvement_percent_std', 0):.2f}%",
                'P_Value': f"{row.get('p_value', 'N/A'):.4f}" if isinstance(row.get('p_value'), float) else 'N/A',
                'Significant': 'Yes' if row.get('significant', False) else 'No',
                'Effect_Size': f"{row.get('effect_size', 0):.3f}",
                'N_Runs': row.get('n_runs', 'N/A')
            })

        enhanced_df = pd.DataFrame(enhanced_table)
        enhanced_df.to_csv(os.path.join(self.tables_dir, "enhanced_bottleneck_analysis.csv"), index=False)

    def _generate_enhanced_schema_entropy_table(self):
        """Generate enhanced schema entropy table with statistical validation."""
        if not self.results["schema_entropy"]:
            return

        df = pd.DataFrame(self.results["schema_entropy"])

        # Format table with statistical validation information
        enhanced_table = []
        for _, row in df.iterrows():
            enhanced_table.append({
                'Schema_Entropy': f"{row['schema_entropy']:.1f}",
                'Embedding_Dim': row['embedding_dim'],
                'Theoretical_Bound': f"{row['theoretical_bound']:.3f}",
                'Recall_Mean': f"{row['recall']:.4f}",
                'Recall_Std': f"{row.get('recall_std', 0):.4f}",
                'Recall_CI': f"[{row.get('recall_ci_lower', 0):.4f}, {row.get('recall_ci_upper', 0):.4f}]",
                'Efficiency_Ratio': f"{row.get('efficiency_ratio', 0):.4f}",
                'Efficiency_Std': f"{row.get('efficiency_ratio_std', 0):.4f}",
                'Bound_Respected': 'Yes' if row.get('bound_respected', False) else 'No',
                'Violation_Rate': f"{row.get('bound_violation_rate', 0) * 100:.1f}%",
                'Significantly_Below_Bound': 'Yes' if row.get('significantly_below_bound', False) else 'No',
                'N_Runs': row.get('n_runs', 'N/A')
            })