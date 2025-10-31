# Dryer_Unit_operations_1
This is a repository for the 1st lab in Unit Operations 1 at the school of Chemical Engineering (NTUA) about the  drying process. The objective in the  removal of volatile liquid (usually water) from solids. We are trying to find the equilibrium humidity in our samples and the drying constant that we can use to scale up our  process.


# Drying Process Analysis Tool

A Python tool for analyzing drying process data using exponential decay modeling, curve fitting, and numerical differentiation.

## Overview

This project provides a comprehensive analysis of drying kinetics data by fitting experimental data to an exponential decay model of the form:

x(t) = x_inf + (x₀ - x_inf) * exp(-k * t)

Where:
- x(t) is the measured value at time t
- x_inf is the equilibrium value
- x₀ is the initial value
- k is the drying rate constant

## Features

- **Curve Fitting**: Non-linear regression to determine model parameters
- **Log-Linear Analysis**: Linear regression on transformed data for parameter verification
- **Numerical Differentiation**: Calculation of derivatives to validate the model
- **Comprehensive Visualization**: Multi-panel plots showing data, fits, and derivatives
- **Excel Integration**: Read data from Excel files and export results

## Requirements

- Python 3.7+
- Required packages:
  - numpy
  - pandas
  - scipy
  - matplotlib
  - scikit-learn

## Installation

1. Clone or download this repository
2. Install required packages:
pip install numpy pandas scipy matplotlib scikit-learn openpyxl

## Usage

### Basic Usage

1. Prepare your Excel data file with the following structure:
   - Column A: Time values (labeled 't')
   - Subsequent columns: Measurement data (x1, x2, x3, etc.)

2. Run the analysis:
analyze_excel('your_data_file.xlsx')

### Output

The tool generates:
- **Visualizations**: Multi-panel plots for each data column saved in a `plots/` folder
- **Derivative Data**: Excel file (`derivatives_output.xlsx`) containing calculated derivatives

### Plot Components

Each analysis generates a 2×2 plot grid showing:
1. **Data and Fit**: Original data with fitted curve and parameters
2. **Extended Prediction**: Model prediction over extended time range
3. **Log Transformation**: Linear fit of log-transformed data
4. **Numerical Derivative**: dx/dt vs. x relationship

## Functions

### `plot_x_analysis(t, x, title=None)`
Performs comprehensive analysis on a single dataset:
- Fits exponential decay model
- Performs log-linear regression
- Calculates numerical derivatives
- Generates visualization

### `analyze_excel(filename)`
Processes an Excel file with multiple data columns:
- Reads time and measurement data
- Calls `plot_x_analysis` for each data column
- Exports derivatives to Excel

## Example

# Analyze your drying data
analyze_excel('drying_data_file_name.xlsx')

## File Structure

project/
├── drying_analysis.py    # Main analysis script
├── your_data_file.xlsx   # Input data (example)
├── plots/                # Generated plots folder
│   ├── x1.png
│   ├── x2.png
│   └── ...
└── derivatives_output.xlsx  # Output derivatives

## Model Details

The tool assumes first-order kinetics for the drying process:
- **Primary Model**: x(t) = x_inf + (x₀ - x_inf) * exp(-k * t)
- **Differential Form**: dx/dt = -k * (x - x_inf)

## Parameters Calculated

- x_inf: Equilibrium value
- k: Drying rate constant
- R²: Goodness of fit for both curve fit and log-linear regression

  
https://github.com/user-attachments/assets/d6f42111-9512-419e-b1b8-5ff46066ab92



## Notes

- Ensure time units are consistent (minutes in the current implementation)
- The tool handles multiple datasets in a single Excel file
- Generated plots are automatically saved in PNG format
- The analysis assumes the data follows exponential decay behavior

## License

This project is provided for academic and research use.
