# ML Experiment Tracking with MLflow

## Task Description
This project focuses on tracking and comparing different machine learning experiments for a House Price Prediction problem using the California Housing dataset. We trained a Gradient Boosting Regressor three times with different hyperparameters and tracked parameters, validation metrics, and model artifacts using MLflow.

## How to Run the Project
1. Activate the virtual environment:
   ```bash
   .\my-env\Scripts\activate
   ```
2. Run the training script to log experiments:
   ```bash
   python train.py
   ```
3. Open the MLflow UI:
   ```bash
   mlflow ui
   ```
4. Navigate to `http://127.0.0.1:5000` in your browser.

## Comparison Table

| Run | Max Depth | Learning Rate | RMSE | MAE | R² |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Run 1 | 3 | 0.1 | 0.542 | 0.372 | 0.776 |
| Run 2 | 5 | 0.05 | 0.520 | 0.353 | 0.794 |
| Run 3 | 7 | 0.01 | 0.697 | 0.531 | 0.629 |

## Selected Best Model
The best performing model configuration is **Run 2 (Max Depth: 5, Learning Rate: 0.05)**.

### Explanation
We selected **Run 2** because it achieved the best overall performance across all metrics on the validation set:
* It has the **lowest RMSE (0.520)**, meaning its predictions have the lowest error magnitude.
* It has the **lowest MAE (0.353)**.
* It has the **highest \(R^2\) score (0.794)**, which proves that this configuration explains about 79.4% of the variance in house prices, making it the most accurate model among the tested experiments.
