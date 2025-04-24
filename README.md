# Information-Theoretic RAG Simulation Framework

This repository contains a Python framework for simulating and analyzing Retrieval-Augmented Generation (RAG) systems in database contexts, using an information-theoretic approach. The code supports the experiments described in the research paper "Information-Theoretic Analysis of Retrieval-Augmented Generation in Database Systems".

## Overview

The framework provides a comprehensive simulation environment for:

1. Generating synthetic database schemas with controlled entropy levels
2. Creating synthetic database content and natural language queries
3. Simulating the entire RAG pipeline as a series of information channels
4. Measuring information-theoretic properties of each channel
5. Analyzing bottlenecks, error propagation, and schema entropy effects

## Requirements

- Python 3.8+
- NumPy
- Pandas
- Matplotlib
- Seaborn
- SciPy
- scikit-learn
- tqdm

Install dependencies with:

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn tqdm
```

## Project Structure

- `information_theoretic_rag.py`: The main simulation framework
- `results/`: Raw experimental results in JSON format
- `figures/`: Generated figures in PNG and PDF formats
- `tables/`: Generated tables in CSV and LaTeX formats

## Running Experiments

### Running All Experiments

To run all experiments:

```bash
python information_theoretic_rag.py --experiment all
```

This will:
1. Generate synthetic database schemas and content
2. Run all experiments (channel capacity, bottleneck analysis, error propagation, schema entropy)
3. Generate all figures and tables

### Running Individual Experiments

To run specific experiments:

```bash
python information_theoretic_rag.py --experiment channel_capacity
python information_theoretic_rag.py --experiment bottleneck
python information_theoretic_rag.py --experiment error_propagation
python information_theoretic_rag.py --experiment schema_entropy
```

### Generating Figures from Existing Results

If you've already run the experiments and just want to regenerate the figures:

```bash
python information_theoretic_rag.py --figures_only
```

## Experiments

The framework includes four main experiments:

### 1. Channel Capacity Experiment

Measures the information capacity of each channel in the RAG pipeline under varying conditions:
- Query Encoding (C₁): Tests varying embedding dimensions
- Retrieval (C₂): Tests varying context sizes
- Context Integration (C₃): Tests varying context sizes
- Generation (C₄): Tests varying temperature settings

### 2. Bottleneck Analysis Experiment

Identifies the primary bottlenecks in the RAG pipeline by:
- Establishing baseline end-to-end performance
- Improving each channel independently (by 50% and 100%)
- Measuring the improvement in end-to-end performance

### 3. Error Propagation Experiment

Analyzes how errors propagate through the RAG pipeline:
- Injects controlled errors into individual channels
- Injects errors into multiple channels simultaneously
- Measures interaction effects between channels

### 4. Schema Entropy Experiment

Investigates the relationship between database schema complexity and RAG performance:
- Tests schemas with varying entropy levels (2-10 bits)
- Tests with varying embedding dimensions (64-1024)
- Compares empirical performance against theoretical bounds

## Outputs

The framework generates:

1. **Results**: Raw experimental data in JSON format
2. **Figures**: Four main figures corresponding to each experiment:
   - Figure 1: Channel capacity measurements under varying conditions
   - Figure 2: Impact of channel improvements on end-to-end performance
   - Figure 3: Error propagation effects in multi-channel experiments
   - Figure 4: Schema entropy effects on RAG performance
3. **Tables**: Summary tables in CSV and LaTeX formats:
   - Channel capacity measurements
   - Bottleneck analysis summary
   - Error propagation effects
   - Schema entropy effects

## Customization

You can customize the experimental parameters by modifying the constants in each experiment method. Key parameters include:

- Embedding dimensions
- Context window sizes
- Temperature settings
- Error levels
- Schema entropy levels
- Number of synthetic queries and database records

## Citation

If you use this framework in your research, please cite our paper:

```bibtex
@article{author2025information,
  title={Information-Theoretic Analysis of Retrieval-Augmented Generation in Database Systems},
  author={Author, A.},
  journal={Journal of Database Research},
  year={2025},
  volume={X},
  number={X},
  pages={XXX--XXX}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
