# Temperature Prediction Model

## Overview
This project contains a machine learning pipeline to predict indoor temperatures (`Temperature_C`) based on various room characteristics, environmental factors, and temporal data. Two machine learning algorithms are utilized and compared: **Support Vector Regression (SVR)** and **Random Forest**.

## Project Structure
- `temperature_dataset_full.csv`: The dataset containing the features and target variable.
- `temperature_modeling.ipynb`: A Jupyter Notebook containing the end-to-end machine learning pipeline, from data loading to model evaluation.

## Dataset
The dataset includes both numerical and categorical features related to:
- Temporal data (Date, Day, TimeSlot)
- Physical space characteristics (Room dimensions, GridSize, WindowCount, Floor/Wall materials)
- Environmental conditions (OutdoorTemp_C, Humidity_pct, Rainfall, CloudCover)
- Occupancy and activity (People, ActiveElectronics, WindowsOpen, DoorStatus)

## Workflow

The `temperature_modeling.ipynb` notebook follows these key steps:

1. **Data loading:** Reading the raw data into a Pandas DataFrame.
2. **Data preparation:** Data cleaning, including the removal of missing values.
3. **Data preprocessing:** 
   - Separating features and the target variable (`Temperature_C`).
   - One-hot encoding for categorical variables.
   - Splitting the dataset into training and testing sets (80/20 split).
   - Feature scaling using `StandardScaler` (which is critical for the SVR model).
4. **Data Visualization:** Exploratory Data Analysis (EDA) using `matplotlib` and `seaborn`.
5. **Model preparation/initialization:** Initializing the Support Vector Regression and Random Forest models.
6. **Model training:** Fitting the models to the scaled training data.
7. **Model testing/evaluation:** Evaluating the models' performance on the testing set to compare accuracy and error metrics.

## Requirements
To run the notebook, you need the following Python libraries installed:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `jupyter`

## Usage
1. Ensure that the `temperature_dataset_full.csv` file is in the same directory as the notebook.
2. Open the `temperature_modeling.ipynb` notebook.
3. Run all cells sequentially to observe the data pipeline, train the models, and view the evaluation results.
