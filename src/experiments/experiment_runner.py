"""
Experiment Runner Module

This module provides the main experiment runner class that manages all
experiments and handles result collection, figure generation, and table creation.
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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set the aesthetic style of the plots

# With this more compatible version:
try:
    # For newer versions of seaborn
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    try:
        # For older versions of seaborn
        plt.style.use('seaborn-whitegrid')
    except:
        # Fallback if neither works
        plt.style.use('default')
        plt.grid(True)

sns.set_context("paper", font_scale=1.5)
sns.set_palette("deep")


class ExperimentRunner:
    """Run the RAG simulation experiments and collect results."""

    def __init__(self, output_dir: str = "."):
        """
        Initialize the experiment runner.

        Args:
            output_dir: Directory to save results and figures
        """
        # Set output directories
        self.output_dir = output_dir
        self.results_dir = os.path.join(output_dir, "results")
        self.figures_dir = os.path.join(output_dir, "figures")
        self.tables_dir = os.path.join(output_dir, "tables")

        # Create directories if they don't exist
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

        logger.info(f"Experiment runner initialized with output directory: {output_dir}")

    def run_experiment(self, experiment_name: str):
        """
        Run a specific experiment.

        Args:
            experiment_name: Name of the experiment to run
        """
        if experiment_name == "channel_capacity":
            from experiments.channel_capacity import run_channel_capacity_experiment
            results = run_channel_capacity_experiment(self.results_dir)
            self.results["channel_capacity"] = results

        elif experiment_name == "bottleneck":
            from experiments.bottleneck_analysis import run_bottleneck_analysis_experiment
            results = run_bottleneck_analysis_experiment(self.results_dir)
            self.results["bottleneck_analysis"] = results

        elif experiment_name == "error_propagation":
            from experiments.error_propagation import run_error_propagation_experiment
            results = run_error_propagation_experiment(self.results_dir)
            self.results["error_propagation"] = results

        elif experiment_name == "schema_entropy":
            from experiments.schema_entropy import run_schema_entropy_experiment
            results = run_schema_entropy_experiment(self.results_dir)
            self.results["schema_entropy"] = results

        elif experiment_name == "all":
            logger.info("Running all experiments")
            self.run_experiment("channel_capacity")
            self.run_experiment("bottleneck")
            self.run_experiment("error_propagation")
            self.run_experiment("schema_entropy")

        else:
            logger.error(f"Unknown experiment: {experiment_name}")

        # Generate figures and tables after running experiments
        self.generate_figures()
        self.generate_tables()

    def load_results(self, experiment_name: Optional[str] = None):
        """
        Load existing results from files.

        Args:
            experiment_name: Specific experiment to load, or None to load all
        """
        if experiment_name is None or experiment_name == "channel_capacity":
            json_path = os.path.join(self.results_dir, "channel_capacity.json")
            if os.path.exists(json_path):
                with open(json_path, "r") as f:
                    self.results["channel_capacity"] = json.load(f)
                logger.info(f"Loaded channel capacity results from {json_path}")

        if experiment_name is None or experiment_name == "bottleneck":
            json_path = os.path.join(self.results_dir, "bottleneck_analysis.json")
            if os.path.exists(json_path):
                with open(json_path, "r") as f:
                    self.results["bottleneck_analysis"] = json.load(f)
                logger.info(f"Loaded bottleneck analysis results from {json_path}")

        if experiment_name is None or experiment_name == "error_propagation":
            json_path = os.path.join(self.results_dir, "error_propagation.json")
            if os.path.exists(json_path):
                with open(json_path, "r") as f:
                    self.results["error_propagation"] = json.load(f)
                logger.info(f"Loaded error propagation results from {json_path}")

        if experiment_name is None or experiment_name == "schema_entropy":
            json_path = os.path.join(self.results_dir, "schema_entropy.json")
            if os.path.exists(json_path):
                with open(json_path, "r") as f:
                    self.results["schema_entropy"] = json.load(f)
                logger.info(f"Loaded schema entropy results from {json_path}")

    def generate_figures(self):
        """Generate figures from experimental results."""
        logger.info("Generating figures")

        # Create figures directory
        os.makedirs(self.figures_dir, exist_ok=True)

        # Figure 1: Channel Capacity Measurements
        self._generate_channel_capacity_figure()

        # Figure 2: Bottleneck Analysis
        self._generate_bottleneck_figure()

        # Figure 3: Error Propagation Effects
        self._generate_error_propagation_figure()

        # Figure 4: Schema Entropy Effects
        self._generate_schema_entropy_figure()

        logger.info("Figure generation completed")

    def _generate_channel_capacity_figure(self):
        """Generate figure for channel capacity results."""
        if not self.results["channel_capacity"]:
            logger.warning("No channel capacity results to plot")
            return

        # Convert to DataFrame for easier plotting
        df = pd.DataFrame(self.results["channel_capacity"])

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        #fig.suptitle("Figure 1: Channel Capacity Measurements", fontsize=16)

        # Plot C1: Query Encoding Channel (embedding dimension)
        c1_data = df[df["channel"] == "C1"]
        if not c1_data.empty:
            ax = axes[0, 0]
            sns.lineplot(data=c1_data, x="embedding_dim", y="mutual_information",
                         marker="o", ax=ax)
            ax.set_title("C1: Query Encoding Channel")
            ax.set_xlabel("Embedding Dimension")
            ax.set_ylabel("Mutual Information (bits)")
            ax.set_xscale("log")

        # Plot C2: Retrieval Channel (context size)
        c2_data = df[df["channel"] == "C2"]
        if not c2_data.empty:
            ax = axes[0, 1]
            sns.lineplot(data=c2_data, x="context_size", y="mutual_information",
                         marker="o", ax=ax)
            ax.set_title("C2: Retrieval Channel")
            ax.set_xlabel("Context Size")
            ax.set_ylabel("Mutual Information (bits)")

        # Plot C3: Context Integration Channel (context size)
        c3_data = df[df["channel"] == "C3"]
        if not c3_data.empty:
            ax = axes[1, 0]
            sns.lineplot(data=c3_data, x="context_size", y="mutual_information",
                         marker="o", ax=ax)
            ax.set_title("C3: Context Integration Channel")
            ax.set_xlabel("Context Size")
            ax.set_ylabel("Mutual Information (bits)")

        # Plot C4: Generation Channel (temperature)
        c4_data = df[df["channel"] == "C4"]
        if not c4_data.empty:
            ax = axes[1, 1]
            sns.lineplot(data=c4_data, x="temperature", y="mutual_information",
                         marker="o", ax=ax)
            ax.set_title("C4: Generation Channel")
            ax.set_xlabel("Temperature")
            ax.set_ylabel("Mutual Information (bits)")

        # Adjust layout and save
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(self.figures_dir, "channel_capacity.png"), dpi=300)
        plt.savefig(os.path.join(self.figures_dir, "channel_capacity.pdf"))
        plt.close()

        logger.info(f"Generated channel capacity figure: {os.path.join(self.figures_dir, 'channel_capacity.png')}")

    def _generate_bottleneck_figure(self):
        """Generate figure for bottleneck analysis results."""
        if not self.results["bottleneck_analysis"]:
            logger.warning("No bottleneck analysis results to plot")
            return

        # Convert to DataFrame for easier plotting
        df = pd.DataFrame(self.results["bottleneck_analysis"])

        # Create figure
        plt.figure(figsize=(10, 6))
        #plt.title("Figure 2: Impact of Channel Improvements on End-to-End Performance", fontsize=14)

        # Plot improvement percentage by channel and improvement factor
        sns.barplot(data=df, x="channel", y="improvement_percent", hue="improvement_factor")

        plt.xlabel("Channel")
        plt.ylabel("Performance Improvement (%)")
        plt.legend(title="Improvement Factor")

        # Add text labels above bars
        for i, row in enumerate(df.itertuples()):
            plt.text(i % 4 - 0.2 + (i // 4) * 0.2,
                     row.improvement_percent + 1,
                     f"{row.improvement_percent:.1f}%",
                     ha="center", fontsize=9)

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, "bottleneck_analysis.png"), dpi=300)
        plt.savefig(os.path.join(self.figures_dir, "bottleneck_analysis.pdf"))
        plt.close()

        logger.info(
            f"Generated bottleneck analysis figure: {os.path.join(self.figures_dir, 'bottleneck_analysis.png')}")

    def _generate_error_propagation_figure(self):
        """Generate figure for error propagation results."""
        if not self.results["error_propagation"]:
            logger.warning("No error propagation results to plot")
            return

        # Convert to DataFrame for easier plotting
        df = pd.DataFrame(self.results["error_propagation"])

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        #fig.suptitle("Figure 3: Error Propagation Effects", fontsize=16)

        # Plot 1: Single Channel Errors
        single_channel_df = df[df["error_type"] == "single"]
        if not single_channel_df.empty:
            sns.lineplot(data=single_channel_df, x="error_level", y="accuracy_drop",
                         hue="channel", marker="o", ax=ax1)
            ax1.set_title("Single Channel Errors")
            ax1.set_xlabel("Error Level")
            ax1.set_ylabel("Accuracy Drop")

        # Plot 2: Multi-Channel Interaction Effects
        multi_channel_df = df[df["error_type"] == "multi"]
        if not multi_channel_df.empty:
            sns.barplot(data=multi_channel_df, x="channels", y="interaction_effect", ax=ax2)
            ax2.set_title("Interaction Effects Between Channels")
            ax2.set_xlabel("Channel Pair")
            ax2.set_ylabel("Interaction Effect (Δ Accuracy)")
            ax2.tick_params(axis='x', rotation=45)

            # Add text labels above bars
            for i, row in enumerate(multi_channel_df.itertuples()):
                ax2.text(i, row.interaction_effect + 0.01,
                         f"{row.interaction_effect:.3f}",
                         ha="center", fontsize=9)

        # Adjust layout and save
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(self.figures_dir, "error_propagation.png"), dpi=300)
        plt.savefig(os.path.join(self.figures_dir, "error_propagation.pdf"))
        plt.close()

        logger.info(f"Generated error propagation figure: {os.path.join(self.figures_dir, 'error_propagation.png')}")

    def _generate_schema_entropy_figure(self):
        """Generate figure for schema entropy results."""
        if not self.results["schema_entropy"]:
            logger.warning("No schema entropy results to plot")
            return

        # Convert to DataFrame for easier plotting
        df = pd.DataFrame(self.results["schema_entropy"])

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        #fig.suptitle("Figure 4: Schema Entropy Effects on RAG Performance", fontsize=16)

        # Plot 1: Retrieval performance vs schema entropy for different embedding dimensions
        sns.lineplot(data=df, x="schema_entropy", y="recall", hue="embedding_dim",
                     marker="o", ax=ax1)
        ax1.set_title("Retrieval Recall vs Schema Entropy")
        ax1.set_xlabel("Schema Entropy (bits)")
        ax1.set_ylabel("Retrieval Recall")
        ax1.legend(title="Embedding Dim")

        # Plot 2: Measured performance vs theoretical bound
        # Create a new DataFrame with theoretical bound and actual performance
        bound_df = df.copy()
        bound_df["information_rate"] = bound_df["recall"] * np.log2(bound_df["embedding_dim"])

        sns.scatterplot(data=bound_df, x="theoretical_bound", y="information_rate",
                        hue="embedding_dim", style="schema_entropy", ax=ax2)

        # Add theoretical maximum line (y=x)
        max_val = bound_df[["theoretical_bound", "information_rate"]].max().max()
        ax2.plot([0, max_val], [0, max_val], 'k--', alpha=0.7)

        ax2.set_title("Actual vs Theoretical Information Rate")
        ax2.set_xlabel("Theoretical Bound: min(log₂(d), H(S))")
        ax2.set_ylabel("Measured Information Rate (bits)")
        ax2.legend(title="Embedding Dim")

        # Adjust layout and save
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(self.figures_dir, "schema_entropy.png"), dpi=300)
        plt.savefig(os.path.join(self.figures_dir, "schema_entropy.pdf"))
        plt.close()

        logger.info(f"Generated schema entropy figure: {os.path.join(self.figures_dir, 'schema_entropy.png')}")

    def generate_tables(self):
        """Generate tables from experimental results."""
        logger.info("Generating tables")

        # Create tables directory
        os.makedirs(self.tables_dir, exist_ok=True)

        # Table 1: Channel Capacity Summary
        self._generate_channel_capacity_table()

        # Table 2: Bottleneck Analysis Summary
        self._generate_bottleneck_table()

        # Table 3: Error Propagation Summary
        self._generate_error_propagation_table()

        # Table 4: Schema Entropy Effects
        self._generate_schema_entropy_table()

        logger.info("Table generation completed")

    def _generate_channel_capacity_table(self):
        """Generate table for channel capacity results."""
        if not self.results["channel_capacity"]:
            logger.warning("No channel capacity results to tabulate")
            return

        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(self.results["channel_capacity"])

        # Create summary table
        c1_data = df[df["channel"] == "C1"].sort_values("embedding_dim")
        c2_data = df[df["channel"] == "C2"].sort_values("context_size")
        c3_data = df[df["channel"] == "C3"].sort_values("context_size")
        c4_data = df[df["channel"] == "C4"].sort_values("temperature")

        # Create summary tables
        c1_summary = c1_data[["embedding_dim", "mutual_information"]].rename(
            columns={"embedding_dim": "Parameter Value", "mutual_information": "Capacity (bits)"})
        c1_summary["Channel"] = "C1: Query Encoding"
        c1_summary["Parameter"] = "Embedding Dimension"

        c2_summary = c2_data[["context_size", "mutual_information"]].rename(
            columns={"context_size": "Parameter Value", "mutual_information": "Capacity (bits)"})
        c2_summary["Channel"] = "C2: Retrieval"
        c2_summary["Parameter"] = "Context Size"

        c3_summary = c3_data[["context_size", "mutual_information"]].rename(
            columns={"context_size": "Parameter Value", "mutual_information": "Capacity (bits)"})
        c3_summary["Channel"] = "C3: Context Integration"
        c3_summary["Parameter"] = "Context Size"

        c4_summary = c4_data[["temperature", "mutual_information"]].rename(
            columns={"temperature": "Parameter Value", "mutual_information": "Capacity (bits)"})
        c4_summary["Channel"] = "C4: Generation"
        c4_summary["Parameter"] = "Temperature"

        # Combine tables
        summary_table = pd.concat([c1_summary, c2_summary, c3_summary, c4_summary])
        summary_table = summary_table[["Channel", "Parameter", "Parameter Value", "Capacity (bits)"]]

        # Format capacity to 2 decimal places
        summary_table["Capacity (bits)"] = summary_table["Capacity (bits)"].map(lambda x: f"{x:.2f}")

        # Save to CSV
        summary_table.to_csv(os.path.join(self.tables_dir, "channel_capacity.csv"), index=False)

        # Also save as LaTeX format for the paper
        with open(os.path.join(self.tables_dir, "channel_capacity.tex"), "w") as f:
            f.write("\\begin{table}[t]\n")
            f.write("\\centering\n")
            f.write("\\caption{Channel Capacity Measurements Under Varying Conditions}\n")
            f.write("\\label{tab:channel_capacity}\n")
            f.write("\\begin{tabular}{llcc}\n")
            f.write("\\toprule\n")
            f.write("Channel & Parameter & Value & Capacity (bits) \\\\\n")
            f.write("\\midrule\n")

            current_channel = ""
            for _, row in summary_table.iterrows():
                channel = row["Channel"]
                parameter = row["Parameter"]
                value = row["Parameter Value"]
                capacity = row["Capacity (bits)"]

                # Only print channel once for each group
                if channel != current_channel:
                    channel_print = channel
                    current_channel = channel
                else:
                    channel_print = ""

                f.write(f"{channel_print} & {parameter} & {value} & {capacity} \\\\\n")

            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")

        logger.info(f"Generated channel capacity table: {os.path.join(self.tables_dir, 'channel_capacity.csv')}")

    def _generate_bottleneck_table(self):
        """Generate table for bottleneck analysis results."""
        if not self.results["bottleneck_analysis"]:
            logger.warning("No bottleneck analysis results to tabulate")
            return

        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(self.results["bottleneck_analysis"])

        # Create summary table
        summary = df.pivot(index="channel", columns="improvement_factor", values="improvement_percent")
        summary = summary.reset_index()
        summary.columns = ["Channel", "50% Improvement", "100% Improvement"]

        # Add channel description
        channel_desc = {
            "C1": "Query Encoding",
            "C2": "Retrieval",
            "C3": "Context Integration",
            "C4": "Generation"
        }

        summary["Description"] = summary["Channel"].map(channel_desc)
        summary["Channel"] = summary["Channel"].map(lambda x: f"C₁" if x == "C1" else
        f"C₂" if x == "C2" else
        f"C₃" if x == "C3" else
        f"C₄" if x == "C4" else x)

        # Reorder columns
        summary = summary[["Channel", "Description", "50% Improvement", "100% Improvement"]]

        # Format percentages
        summary["50% Improvement"] = summary["50% Improvement"].map(lambda x: f"{x:.2f}%")
        summary["100% Improvement"] = summary["100% Improvement"].map(lambda x: f"{x:.2f}%")

        # Save to CSV
        summary.to_csv(os.path.join(self.tables_dir, "bottleneck_analysis.csv"), index=False)

        # Also save as LaTeX format for the paper
        with open(os.path.join(self.tables_dir, "bottleneck_analysis.tex"), "w") as f:
            f.write("\\begin{table}[t]\n")
            f.write("\\centering\n")
            f.write("\\caption{Performance Improvement from Enhancing Individual Channels}\n")
            f.write("\\label{tab:bottleneck_analysis}\n")
            f.write("\\begin{tabular}{llcc}\n")
            f.write("\\toprule\n")
            f.write("Channel & Description & 50\\% Improvement & 100\\% Improvement \\\\\n")
            f.write("\\midrule\n")

            for _, row in summary.iterrows():
                channel = row["Channel"]
                desc = row["Description"]
                imp50 = row["50% Improvement"]
                imp100 = row["100% Improvement"]

                f.write(f"{channel} & {desc} & {imp50} & {imp100} \\\\\n")

            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")

        logger.info(f"Generated bottleneck analysis table: {os.path.join(self.tables_dir, 'bottleneck_analysis.csv')}")

    def _generate_error_propagation_table(self):
        """Generate table for error propagation results."""
        if not self.results["error_propagation"]:
            logger.warning("No error propagation results to tabulate")
            return

        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(self.results["error_propagation"])

        # 1. Single Channel Errors Summary
        single_df = df[df["error_type"] == "single"]
        single_summary = single_df.pivot(index="channel", columns="error_level", values="accuracy_drop")
        single_summary = single_summary.reset_index()
        single_summary.columns = ["Channel", "Error 0.1", "Error 0.2", "Error 0.3"]

        # Rename channels
        single_summary["Channel"] = single_summary["Channel"].map(lambda x: f"C₁" if x == "C1" else
        f"C₂" if x == "C2" else
        f"C₃" if x == "C3" else
        f"C₄" if x == "C4" else x)

        # Format numbers
        for col in ["Error 0.1", "Error 0.2", "Error 0.3"]:
            single_summary[col] = single_summary[col].map(lambda x: f"{x:.4f}")

        # 2. Multi-Channel Interactions Summary
        multi_df = df[df["error_type"] == "multi"]
        multi_summary = multi_df[["channels", "accuracy_drop", "expected_drop", "interaction_effect"]]

        # Calculate relative interaction strength
        multi_summary["interaction_ratio"] = multi_summary["interaction_effect"] / multi_summary["expected_drop"]

        # Format columns
        multi_summary["accuracy_drop"] = multi_summary["accuracy_drop"].map(lambda x: f"{x:.4f}")
        multi_summary["expected_drop"] = multi_summary["expected_drop"].map(lambda x: f"{x:.4f}")
        multi_summary["interaction_effect"] = multi_summary["interaction_effect"].map(lambda x: f"{x:.4f}")
        multi_summary["interaction_ratio"] = multi_summary["interaction_ratio"].map(lambda x: f"{x:.2f}")

        # Rename columns
        multi_summary = multi_summary.rename(columns={
            "channels": "Channel Pair",
            "accuracy_drop": "Actual Drop",
            "expected_drop": "Expected Drop",
            "interaction_effect": "Interaction Effect",
            "interaction_ratio": "Interaction Ratio"
        })

        # Save to CSV
        single_summary.to_csv(os.path.join(self.tables_dir, "error_single_channel.csv"), index=False)
        multi_summary.to_csv(os.path.join(self.tables_dir, "error_multi_channel.csv"), index=False)

        # Also save as LaTeX format for the paper
        with open(os.path.join(self.tables_dir, "error_propagation.tex"), "w") as f:
            # Single channel errors
            f.write("\\begin{table}[t]\n")
            f.write("\\centering\n")
            f.write("\\caption{Accuracy Drop from Single-Channel Errors}\n")
            f.write("\\label{tab:error_single}\n")
            f.write("\\begin{tabular}{lccc}\n")
            f.write("\\toprule\n")
            f.write("Channel & Error 0.1 & Error 0.2 & Error 0.3 \\\\\n")
            f.write("\\midrule\n")

            for _, row in single_summary.iterrows():
                channel = row["Channel"]
                e1 = row["Error 0.1"]
                e2 = row["Error 0.2"]
                e3 = row["Error 0.3"]

                f.write(f"{channel} & {e1} & {e2} & {e3} \\\\\n")

            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n\n")

            # Multi-channel errors
            f.write("\\begin{table}[t]\n")
            f.write("\\centering\n")
            f.write("\\caption{Interaction Effects Between Channels with Error Level 0.2}\n")
            f.write("\\label{tab:error_multi}\n")
            f.write("\\begin{tabular}{lcccc}\n")
            f.write("\\toprule\n")
            f.write("Channel Pair & Actual Drop & Expected Drop & Interaction Effect & Ratio \\\\\n")
            f.write("\\midrule\n")

            for _, row in multi_summary.iterrows():
                pair = row["Channel Pair"]
                actual = row["Actual Drop"]
                expected = row["Expected Drop"]
                effect = row["Interaction Effect"]
                ratio = row["Interaction Ratio"]

                f.write(f"{pair} & {actual} & {expected} & {effect} & {ratio} \\\\\n")

            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")

        logger.info(f"Generated error propagation tables: {os.path.join(self.tables_dir, 'error_single_channel.csv')}")

    def _generate_schema_entropy_table(self):
        """Generate table for schema entropy results."""
        if not self.results["schema_entropy"]:
            logger.warning("No schema entropy results to tabulate")
            return

        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(self.results["schema_entropy"])

        # Create summary table grouped by schema entropy and embedding dimension
        summary = df.pivot_table(
            index="schema_entropy",
            columns="embedding_dim",
            values=["accuracy", "precision", "recall", "theoretical_bound"],
            aggfunc="mean"
        )

        # Reset index for easier manipulation
        summary = summary.reset_index()

        # Format for CSV output - flatten the multi-index columns
        csv_summary = pd.DataFrame()
        csv_summary["Schema Entropy (bits)"] = summary["schema_entropy"]

        for dim in sorted(df["embedding_dim"].unique()):
            csv_summary[f"Accuracy (dim={dim})"] = summary[("accuracy", dim)].map(lambda x: f"{x:.4f}")
            csv_summary[f"Recall (dim={dim})"] = summary[("recall", dim)].map(lambda x: f"{x:.4f}")
            csv_summary[f"Bound (dim={dim})"] = summary[("theoretical_bound", dim)].map(lambda x: f"{x:.4f}")

        # Save to CSV
        csv_summary.to_csv(os.path.join(self.tables_dir, "schema_entropy.csv"), index=False)

        # Also save as LaTeX format for the paper
        with open(os.path.join(self.tables_dir, "schema_entropy.tex"), "w") as f:
            f.write("\\begin{table}[t]\n")
            f.write("\\centering\n")
            f.write("\\caption{Effect of Schema Entropy on RAG Performance}\n")
            f.write("\\label{tab:schema_entropy}\n")
            f.write("\\begin{tabular}{cccccc}\n")
            f.write("\\toprule\n")
            f.write("Schema & \\multicolumn{2}{c}{Dim = 128} & \\multicolumn{2}{c}{Dim = 512} \\\\\n")
            f.write("Entropy (bits) & Recall & Bound & Recall & Bound \\\\\n")
            f.write("\\midrule\n")

            # Select a subset of dimensions for clarity
            selected_dims = [128, 512]

            for _, row in summary.iterrows():
                entropy = row["schema_entropy"]

                parts = [f"{entropy[0]:.1f}"]
                for dim in selected_dims:
                    recall = row[("recall", dim)]
                    bound = row[("theoretical_bound", dim)]
                    parts.append(f"{recall:.3f}")
                    parts.append(f"{bound:.3f}")

                f.write(" & ".join(parts) + " \\\\\n")

            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")

        logger.info(f"Generated schema entropy table: {os.path.join(self.tables_dir, 'schema_entropy.csv')}")


if __name__ == "__main__":
    # Example usage
    runner = ExperimentRunner(output_dir=".")

    # Load existing results
    runner.load_results()

    # Generate figures and tables
    runner.generate_figures()
    runner.generate_tables()