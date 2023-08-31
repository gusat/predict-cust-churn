# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

This project aims to develop a churn prediction model using machine learning techniques. It includes utility functions for data preprocessing, visualization, model evaluation, and more.

## Project Structure

The project files are organized as follows:

```
churn_project/
│   README.md
│   churn_library_v15.py
│   churn_nbk_v15.ipynb
│   test_churn_lib_v15.py
│
├── data/
│   │   bank_data.csv
│
├── logs/
│   │   churn_script.log
│
├── models/
│   │   (Generated pkl files to storel/load ML models)
│
└── .gitignore
```

## Usage (tested on VSC with python311)

### `churn_library_v15.py`

This module contains utility functions to facilitate various steps of the churn prediction process. To use it, you can import the functions in your own scripts or notebooks.

### `churn_nbk_v15.ipynb`

This Jupyter Notebook contains the main analysis and modeling process. It imports functions from `churn_library.py` and demonstrates how to preprocess data, train models, and evaluate performance.

### `test_churn_lib_v15.py`

This module contains test cases for the functions in `churn_library.py`. You can run the tests using the `pytest` command.

## Running Files

To run the `churn_nbk_v15.ipynb` notebook, make sure you have Jupyter Notebook installed. You can then open the notebook and execute each cell to follow the analysis and modeling process.

To run the tests in `test_churn_lib_v15.py`, ensure you have the `pytest` testing framework installed. In your terminal, navigate to the project directory and run the following command:

```
pytest test_churn_lib_v15.py
```

Make sure that the required dataset `bank_data.csv` is placed in the `data/` directory.

## Author

- Mitch Gusat

## Date

- 31.08.2023
```
# Details: Churn Prediction Notebook and Library

This repository contains a Jupyter notebook `churn_nbk.ipynb` that demonstrates the process of churn prediction using machine learning. Additionally, it provides a Python library `churn_library.py` that contains various utility functions for data preprocessing, model training, and evaluation.

## Notebook Usage

The main notebook `churn_nbk_v15.ipynb` showcases the process of churn prediction using a dataset. The following libraries are used in the notebook:

- shap
- joblib
- pandas
- numpy
- matplotlib.pyplot
- seaborn
- sklearn
- scikitplot
- os

To run the notebook, ensure you have these libraries installed in your environment. You can execute the notebook cell by cell to follow along with the churn prediction process.

## Testing

The `test_churn_lib_v15.py` file contains test cases for the functions in the `churn_library.py` module. It uses the `pytest` framework to automate the testing process. To run the tests, use the following command:

```bash
pytest test_churn_lib_v15.py
```

## Churn Library

The `churn_library_v15.py` module is a collection of utility functions that facilitate various steps of the churn prediction process. The functions include:

- `import_data(pth)`: Import the dataset from the given path.
- `plot_univar_quant(data, column)`: Generate a univariate quantitative plot.
- `plot_bivariate(data, x_column, y_column)`: Generate a bivariate scatter plot.
- `separate_columns(data, target)`: Separate features and target columns.
- `plot_corr_matrix(quant_df, save_filename=None)`: Plot the correlation matrix.
- `encode_categorical_columns(df, cat_cols)`: Encode categorical columns using the mean response approach and one-hot encoding.
- `print_scores(cv_rfc, lrc, X_train, X_test, y_train, y_test, y_train_preds_rf, y_test_preds_rf, y_train_preds_lr, y_test_preds_lr)`: Print classification reports for model evaluation.
- `plot_roc_curves(rfc_model, lrc, X_test, y_test)`: Plot ROC curves for models.
- `plot_feature_importances(rfc_best_estimator, X_train)`: Plot feature importances of a Random Forest classifier.
- `choose_predictors(data)`: Choose predictor columns for modeling.
- `train_fit_models(X, y)`: Train and fit machine learning models.

Refer to the individual function documentation in the `churn_library_v15.py` module for more details on their usage.

Example run
C:.../Python311/python.exe c:/Users/970986848/Documents/U/Clean-Code/churn_library_v15.py  
Categorical Columns: ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
Quantitative Columns: ['Customer_Age', 'Dependent_count', 'Months_on_book', 'Total_Relationship_Count', 'Months_Inactive_12_mon', 'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal', 'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt', 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']
Project run completed.
PS C:\...\U\Clean-Code> 
---

## Running Files

### Notebook

To run the main notebook `churn_nbk_v15.ipynb`, follow these steps:

1. Make sure you have Jupyter Notebook installed in your Python environment.
2. Open a terminal or command prompt.
3. Navigate to the directory containing `churn_nbk_v15.ipynb`.
4. Run the following command to start Jupyter Notebook:
   
   ```bash
   jupyter notebook
   ```
   
5. A web browser window should open with the Jupyter Notebook dashboard. Click on `churn_nbk.ipynb` to open the notebook.
6. Execute each cell in the notebook sequentially to follow along with the churn prediction process.

### Testing

To run the tests for the `churn_library_v15.py` module using `pytest`, follow these steps:

1. Open a terminal or command prompt.
2. Navigate to the directory containing `test_churn_lib_v15.py`.
3. Run the following command to execute the tests:
   
   ```bash
   pytest test_churn_lib_v15.py
   ```

   The tests will automatically discover and run the test functions, providing you with the test results.

---
MIT License
