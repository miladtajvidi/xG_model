# Expected Goals (xG) Model with StatsBomb Data

This project implements an Expected Goals (xG) model using StatsBomb's open data. The model uses logistic regression with various shot features and includes Venn-ABERS calibration for improved probability estimates.

## Project Structure

- `shots.py`: Extracts shot events from StatsBomb's open data
- `xG_Data_prep.py`: Data cleaning and feature engineering
- `xG_model.py`: Model training and evaluation
- `xG_calibration.py`: Probability calibration using Venn-ABERS method
- `venn_ABERS.py`: Template implementation of Venn-ABERS calibration

## Features

The model includes features such as :
- Location coordinates (start_x, start_y, end_x, end_y)
- Distance and angle to goal
- Shot technique and body part used
- First-time shot indicator
- Number of defenders/teammates in shot path
- Goalkeeper positioning
- Under pressure indicator
- Play pattern

## Model Pipeline

1. **DataExtraction** (`shots.py`):
   - Extracts shot events

2. **Data Preparation** (`xG_Data_prep.py`):
   - Handles missing values
   - Calculates geometric features (distance, angle)
   - Engineers defensive coverage and the rest of the features
   

3. **Model Training** (`xG_model.py`):
   - Applying Standard Scaler for numeric columns
   - One-hot encoding of categorical variables
   - Train/test split
   - Logistic regression with grid search
   - Model evaluation using:
     - Log Loss
     - ROC AUC
     - Brier Score

4. **Probability Calibration** (`xG_calibration.py`):
   - Implements Venn-ABERS calibration
   - Compares different calibration methods:
     - Inductive Venn-ABERS (IVAP)
     - Cross Venn-ABERS (CVAP)
     - Pre-fitted Venn-ABERS
   - Generates calibration curves

## Results

The model achieves:
- ROC AUC Score: ~0.92
- Log Loss: ~0.20
- Brier Score: ~0.06

Venn-ABERS calibration helps improve the reliability of probability estimates

## References

1- [McKay Johns' Build Expected Goals Model](https://github.com/mckayjohns/youtube-videos/blob/main/code/build_expected_goals_model.ipynb)
2- [ConvergenceWarning: Liblinear failed to converge, increase the number of iterations](https://stackoverflow.com/questions/52670012/convergencewarning-liblinear-failed-to-converge-increase-the-number-of-iterati)
3- [Preprocessing in scikit-learn](https://scikit-learn.org/stable/modules/preprocessing.html)
4- [Logistic Regression and Different Solvers in scikit-learn](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
5- [Why ROC AUC Might Be Misleading — and What to Use Instead](https://valeman.medium.com/why-roc-auc-might-be-misleading-and-what-to-use-instead-520ea242be8a)
6- [How to calibrate your classifier in an intelligent way using Venn-Abers Conformal Prediction](https://valeman.medium.com/how-to-calibrate-your-classifier-in-an-intelligent-way-a996a2faf718)
7- [Venn-ABERS Calibration github repo](https://github.com/ip200/venn-abers)
