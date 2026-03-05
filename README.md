
# Solar Radiation Forecasting using Machine Learning

## Overview

This project develops machine learning models to **forecast daily solar radiation using meteorological time-series data**. Accurate solar radiation forecasting is important for **renewable energy planning, solar power generation management, and smart grid optimization**.

The project investigates the performance of **ensemble machine learning models and deep learning architectures** under limited data conditions.

This work was completed as part of the MSc dissertation in **Artificial Intelligence & Data Science at the University of Hull**.

---

## Objectives

The main objectives of this project are:

* Forecast daily solar radiation using meteorological variables
* Compare traditional machine learning models with deep learning approaches
* Evaluate forecasting performance under **limited data conditions**
* Analyze model robustness and bias–variance trade-offs

---

## Dataset

The dataset consists of **meteorological time-series observations**, including variables such as:

* Temperature
* Humidity
* Wind speed
* Atmospheric pressure
* Solar radiation measurements

Feature engineering was applied to generate **lag-based and seasonal features** to improve predictive performance.

---

## Methodology

### Data Preprocessing

The following preprocessing steps were applied:

* Data cleaning and handling missing values
* Feature scaling and normalization
* Creation of lag features for time-series forecasting
* Chronological train–test split to avoid data leakage

---

### Machine Learning Models

The following models were implemented and evaluated:

**Ensemble Models**

* Random Forest
* XGBoost

**Deep Learning Models**

* LSTM (Long Short-Term Memory)
* CNN–LSTM hybrid model

These models were selected to compare **tree-based ensemble learning with deep neural time-series architectures**.

---

## Model Evaluation

Model performance was evaluated using standard regression metrics:

* **R² (Coefficient of Determination)**
* **RMSE (Root Mean Square Error)**
* **MAE (Mean Absolute Error)**
* **MAPE (Mean Absolute Percentage Error)**

The best-performing models achieved approximately:

**R² ≈ 0.98**

indicating strong predictive capability on the test dataset.

---


## Key Contributions

This project demonstrates:

* A complete **machine learning pipeline for renewable energy forecasting**
* Comparative evaluation of **ensemble vs deep learning approaches**
* Robust methodology using **chronological validation**
* Practical insights into **ML performance under limited data conditions**

---

## Applications

The methodology developed in this project can support:

* Solar power generation forecasting
* Renewable energy grid integration
* Smart grid planning and operation
* Energy demand–supply optimization

---

## Future Work

Possible extensions include:

* Probabilistic forecasting for uncertainty quantification
* Hybrid physical–machine learning models
* Integration with real-time energy management systems
* Application to larger multi-site renewable datasets

---

## Author
Majid Rasheed
MSc Artificial Intelligence & Data Science
University of Hull

Research interests include:

* Machine learning for renewable energy systems
* Time-series forecasting under uncertainty
* Smart grid optimization and infrastructure AI

---
