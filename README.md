# fairness-recommender-systems

# Fairness in Recommender Systems
Bachelor Thesis: University of Zurich, 2026  
Author: Mohamed Nacer Chabbi  
Supervisor: Salima Jaoua

## Dataset
This project uses the MovieLens 1M dataset, which must be downloaded 
separately from:  
https://grouplens.org/datasets/movielens/1m/

After downloading, place the extracted folder at:  
ml-1m/ml-1m/

The folder should contain: users.dat, movies.dat, ratings.dat

## Requirements
Python 3.8 or higher. Install dependencies with:

pip install pandas numpy matplotlib scikit-learn scipy

## Execution Order
The scripts must be run in the following order:

1. Recommender_Systems.py
   Defines the five recommendation strategies used in the study.

2. simulations.py
   Runs the simulation across ten cumulative time steps and saves 
   results to results/simulation_results.pkl

3. fairness_metrics_rs.py
   Loads the simulation results and computes all fairness metrics. 
   Generates plots and CSV files in the plots/ and results/ folders.

## Output
- results/rs_extended/fairness_rs_extended_all.csv  
  Full metric results across all five recommender systems and ten 
  time steps (also included in this repository)
- plots/fairness_rs_extended/final/  
  All figures used in the empirical analysis

## Note on AI Assistance
Parts of this code were developed with the assistance of generative 
AI tools, as declared in the thesis. The scientific content, metric 
implementations, and analytical decisions are the author's own.
