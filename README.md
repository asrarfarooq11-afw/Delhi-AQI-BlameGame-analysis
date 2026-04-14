 ## Delhi AQI Regional Impact Analysis

## Overview

This project performs statistical and machine learning analysis on Delhi Air Quality Index (AQI) data to identify regional contributions from neighboring states such as Haryana and Punjab. The analysis uses station-aware filtering, lag-based modeling, and feature ablation techniques to understand cross-regional pollution dynamics.

## Objective

* Analyze AQI trends across Delhi, Haryana, and Punjab
* Identify regional contributors to Delhi pollution
* Perform lag-based analysis to detect delayed pollution effects
* Evaluate feature importance using statistical and ML techniques

## Tech Stack

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* Jupyter Notebook

## Methodology

### Data Processing

* Station-aware filtering based on station start year
* Data cleaning and preprocessing
* State-wise AQI aggregation

### Statistical Analysis

* Correlation analysis
* Lag-based regional impact analysis
* P-value significance testing

### Machine Learning Models

* Linear Regression
* Random Forest Regressor
* XGBoost Regressor

### Feature Analysis

* Feature ablation experiments
* Regional contribution evaluation
* Seasonal pattern detection (stubble burning analysis)

## Key Findings

* Haryana identified as a major contributor to Delhi AQI levels
* Punjab shows lag-based influence on Delhi pollution
* Seasonal pollution spikes linked to stubble burning periods
* Regional AQI variables significantly improve prediction performance

## My Contribution

* Data preprocessing and station-aware filtering
* Statistical correlation analysis
* Feature ablation experiments
* Lag-based regional modeling
* Machine learning model implementation
* Visualization and interpretation of results

## Project Type

Academic Group Project

## How to Run

1. Clone repository

```bash
git clone https://github.com/yourusername/repository-name.git
```

2. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

3. Open notebook

```bash
jupyter notebook Aqi_data_prediction.ipynb
```

4. Run all cells

## Dataset

Delhi, Haryana, and Punjab AQI station data including:

* Station-wise AQI readings
* State-level pollution data
* Seasonal pollution trends

## Future Improvements

* Deploy real-time AQI prediction system
* Build interactive Streamlit dashboard
* Integrate live AQI API
* Improve model accuracy using deep learning

## Author

Asrar Farooq

---

This project demonstrates statistical modeling, machine learning, and environmental data analysis applied to real-world air pollution problems.
