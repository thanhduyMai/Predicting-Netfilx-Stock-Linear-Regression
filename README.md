# Predicting-Netfilx-Stock-Linear-Regression

Netflix Stock Price Prediction

1. Introduction

This project focuses on predicting Netflix stock prices using linear regression. The dataset originates from the Netflix Stock Price Prediction dataset and has been processed for analysis.

2. Data Source

Original Source: Kaggle

Original Dataset Name: Netflix Stock Price Prediction

Original Dataset Link: https://www.kaggle.com/datasets/jainilcoder/netflix-stock-price-prediction

3. Modifications and Processing

The dataset has been preprocessed by removing unnecessary columns and applying linear regression modeling techniques. The key modifications include:

Data cleaning and handling missing values

Feature selection for predictive modeling

Implementing linear regression for stock price prediction

4. Project Objective

Analyze historical stock data of Netflix

Apply linear regression to predict future stock prices

Evaluate model performance using statistical metrics

5. Usage Instructions

To run this project, use Jupyter Notebook and install the required libraries:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

Load the dataset:

df = pd.read_csv("Netflix-Stock-Data.csv")

Execute the Jupyter Notebook file Netflix-Stock-Linear-Regression.ipynb to perform the analysis.

6. Notes

This dataset has been processed and may not contain all columns from the original dataset.

Ensure all dependencies are installed before running the notebook.

For further analysis, refer to the original dataset.

For any questions or modifications, please reach out to the project maintainer.
