import os
import logging
import pytest
import unittest
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from scikitplot.metrics import plot_roc_curve as skplt_plot_roc_curve
from sklearn.metrics import classification_report

import churn_library as cs

output_path = './'

logging.basicConfig(
    filename='./logs/churn_script.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import_data():
    """
    Test the import_data function.

    This test case validates the import_data function by attempting to import test data.
    If successful, it checks the shape of the imported DataFrame and logs the success or error.
    """
    logging.info("Running test_import_data")

    try:
        df = cs.import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_data: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_encode_categorical_columns():
    """
    Basic test for encode_categorical_columns function.

    This test case checks whether the function correctly encodes categorical columns and appends "_Churn".
    It verifies that the expected input and output conditions are met.
    """
    logging.info("Running test_encode_categorical_columns basic")

    try:
        # Set up a dummy test dataset 
        data = {
            'Gender': [
                'Male',
                'Female',
                'Male',
                'Female',
                'Male'],
            'Education_Level': [
                'High School',
                'Graduate',
                'Graduate',
                'Doctorate',
                'Doctorate'],
            'Marital_Status': [
                'Single',
                'Married',
                'Divorced',
                'Single',
                'Married'],
            'Income_Category': [
                '$40K - $60K',
                '$60K - $80K',
                'Less than $40K',
                '$80K - $120K',
                'Unknown'],
            'Card_Category': [
                'Blue',
                'Gold',
                'Silver',
                'Blue',
                'Gold'],
            'Churn': [
                0,
                1,
                1,
                0,
                0]}
        test_df = pd.DataFrame(data)

        # Expected output column names
        expected_output = [
            'Gender_Churn',
            'Education_Level_Churn',
            'Marital_Status_Churn',
            'Income_Category_Churn',
            'Card_Category_Churn'
        ]

        # Encode categorical columns
        encoded_df = cs.encode_categorical_columns(
            test_df, [
                'Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category'])

        # Validate the encoded columns
        assert all(col in encoded_df.columns for col in expected_output)

        # Log success
        logging.info("test_encode_categorical_columns basics: SUCCESS")
    except Exception as e:
        logging.error(f"test_encode_categorical_columns basic: {e}")
        raise e


def test_plot_bivariate():
    """
    Test the plot_bivariate function.

    This test case sets up a test DataFrame, creates the bivariate scatter plot,
    validates the generated file, and logs the success or error.
    """
    logging.info("Running test_plot_bivariate")

    try:
        # Set the test data
        test_data = pd.DataFrame({
            'x_column': [1, 2, 3],
            'y_column': [4, 5, 6]
        })

        # Create the plot
        cs.plot_bivariate(test_data, 'x_column', 'y_column')

        # Validate if the output file was generated
        expected_file = f'bivariate_x_column_vs_y_column.png'
        assert os.path.exists(
            os.path.join(
                output_path, expected_file)), f"Expected file {expected_file} not found"
        logging.info("test_plot_bivariate: SUCCESS")

        # Clean up: Remove the generated file
        os.remove(os.path.join(output_path, expected_file))
    except Exception as e:
        logging.error(f"test_plot_bivariate: {e}")
        raise e


def test_plot_univar_quant():
    """
    Test the plot_univar_quant function.

    This test case sets up a test DataFrame, creates the univariate quantitative plot,
    validates the generated file, and logs the success or error.
    """
    logging.info("Running test_plot_univar_quant")

    try:
        # Create the test data
        test_data = pd.DataFrame({
            'column': [1, 2, 3, 4, 5]
        })
        # Create the plot
        cs.plot_univar_quant(test_data, 'column')

        # Validate if the output file was generated
        expected_file = f'univariate_quantitative_column.png'
        assert os.path.exists(
            os.path.join(
                output_path, expected_file)), f"Expected file {expected_file} not found"
        logging.info("test_plot_univar_quant: SUCCESS")

        # Clean up: Remove the generated file
        os.remove(os.path.join(output_path, expected_file))
    except Exception as e:
        logging.error(f"test_plot_univar_quant: {e}")
        raise e


def test_plot_corr_matrix():
    """
    Test the plot_corr_matrix function.

    This test case sets up a test DataFrame, creates the correlation matrix plot,
    validates the generated file, and logs the success or error.
    """
    logging.info("Running test_plot_corr_matrix")

    try:
        # Set the test data
        test_data = pd.DataFrame({
            'Column1': [1, 2, 3],
            'Column2': [4, 5, 6],
            'Column3': [7, 8, 9]
        })

        # Create the correlation matrix plot
        cs.plot_corr_matrix(
            test_data, save_filename=os.path.join(
                output_path, 'test_corr_matrix.png'))

        # Validate if the output file was generated
        expected_file = 'test_corr_matrix.png'
        assert os.path.exists(
            os.path.join(
                output_path, expected_file)), f"Expected file {expected_file} not found"
        logging.info("test_plot_corr_matrix: SUCCESS")

        # Clean up: Remove the generated file
        os.remove(os.path.join(output_path, expected_file))
    except Exception as e:
        logging.error(f"test_plot_corr_matrix: {e}")
        raise e


def test_plot_roc_curves():
    """
    Test the plot_roc_curves function.

    This test case sets up a test dataset and trained models, creates ROC curves,
    validates the generated file, and logs the success or error.
    """
    logging.info("Running test_plot_roc_curves")

    try:
        # Set up a test dataset
        X, y = make_classification(
            n_samples=100, n_features=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42)

        # Train a RandomForestClassifier and LogisticRegression models
        rfc = RandomForestClassifier(random_state=42)
        lrc = LogisticRegression(random_state=42)
        rfc.fit(X_train, y_train)
        lrc.fit(X_train, y_train)

        # Create the ROC curves
        cs.plot_roc_curves(rfc, lrc, X_test, y_test)

        # Validate if the output file was generated
        expected_file = 'roc_curves.png'
        assert os.path.exists(
            os.path.join(
                output_path, expected_file)), f"Expected file {expected_file} not found"
        logging.info("test_plot_roc_curves: SUCCESS")

        # Clean up: Remove the generated file
        os.remove(os.path.join(output_path, expected_file))
    except Exception as e:
        logging.error(f"test_plot_roc_curves: {e}")
        raise e


def test_print_scores():
    """
    Test the print_scores function.

    This test case sets up a test dataset, trains a RandomForestClassifier and LogisticRegression,
    makes predictions, prints classification reports, and logs the success or error.
    """
    logging.info("Running test_print_scores")

    try:
        # Set up a test dataset
        X, y = make_classification(
            n_samples=100, n_features=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42)

        # Train a RandomForestClassifier and LogisticRegression models
        rfc = RandomForestClassifier(random_state=42)
        lrc = LogisticRegression(random_state=42)
        rfc.fit(X_train, y_train)
        lrc.fit(X_train, y_train)

        # Make predictions
        y_train_preds_rf = rfc.predict(X_train)
        y_test_preds_rf = rfc.predict(X_test)
        y_train_preds_lr = lrc.predict(X_train)
        y_test_preds_lr = lrc.predict(X_test)

        # Print classification reports
        cs.print_scores(
            rfc,
            lrc,
            X_train,
            X_test,
            y_train,
            y_test,
            y_train_preds_rf,
            y_test_preds_rf,
            y_train_preds_lr,
            y_test_preds_lr)

        # Log success
        logging.info("test_print_scores: SUCCESS")
    except Exception as e:
        logging.error(f"test_print_scores: {e}")
        raise e


def test_plot_feature_importances():
    """
    Test the plot_feature_importances function.

    This test case sets up a test dataset, trains a RandomForestClassifier model, and plots
    the feature importances, while logging the success or error.
    """
    logging.info("Running test_plot_feature_importances")

    try:
        # Set up a test dataset
        X, y = make_classification(
            n_samples=100, n_features=10, random_state=42)
        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.3, random_state=42)

        # Train a RandomForestClassifier model
        rfc = RandomForestClassifier(random_state=42)
        rfc.fit(X_train, y_train)

        # Convert X_train to a Pandas DataFrame
        X_train_df = pd.DataFrame(
            X_train, columns=[
                f'feature_{i}' for i in range(
                    X_train.shape[1])])

        # Plot feature importances
        cs.plot_feature_importances(rfc, X_train_df)

        # Log success
        logging.info("test_plot_feature_importances: SUCCESS")
    except Exception as e:
        logging.error(f"test_plot_feature_importances: {e}")
        raise e


def test_train_fit_models():
    """
    Test the train_fit_models function.

    This test case sets up a test dataset, trains and fits machine learning models using
    train_fit_models function, and validates the predictions, while logging the success or error.
    """
    logging.info("Running test_train_fit_models")

    try:
        # Set up a test dataset
        X, y = make_classification(
            n_samples=200, n_features=15, random_state=42)

        # Train and fit models
        cv_rfc, lrc, X_train, X_test, y_train, y_test, y_train_preds_rf, y_test_preds_rf, y_train_preds_lr, y_test_preds_lr = cs.train_fit_models(
            X, y)

        # Validate predictions
        assert y_train_preds_rf.shape == y_train.shape
        assert y_test_preds_rf.shape == y_test.shape
        assert y_train_preds_lr.shape == y_train.shape
        assert y_test_preds_lr.shape == y_test.shape

        # Log success
        logging.info("test_train_fit_models: SUCCESS")
    except Exception as e:
        logging.error(f"test_train_fit_models: {e}")
        raise e


# def run_tests():
#     test_functions = [
#         # test_import_data,
#         # test_plot_bivariate,
#         # test_plot_univar_quant,
#         # test_print_scores,
#         # test_plot_roc_curves,
#         # test_plot_corr_matrix,
#         # test_plot_feature_importances,
#         # test_encode_categorical_columns,
#         # test_train_fit_models
#         # Add other test functions here
#         # test_separate_columns,
#         # test_choose_predictors
#     ]

#     for test_func in test_functions:
#         try:
#             test_func()
#         except Exception as err:
#             logging.error(f"Test failed for {test_func.__name__}: {str(err)}")
#         else:
#             logging.info(f"Test passed for {test_func.__name__}")

# if __name__ == "__main__":
#     run_tests()

# or automatically discover and run all tests
if __name__ == "__main__":
    pytest.main()
