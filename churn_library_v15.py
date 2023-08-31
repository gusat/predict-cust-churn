"""
Churn Library
--------------
This module is a collection of utility functions for the churn prediction.
Modules for: data preprocessing, visualization, model training, and evaluation.

Author: Mitch Gusat
Date Created: Aug. 2023

Functions:
----------
- import_data(pth):
    Import the dataset from the given path.

- plot_univar_quant(data, column):
    Generate a univariate quantitative plot.

- plot_bivariate(data, x_column, y_column):
    Generate a bivariate scatter plot.

- separate_columns(data, target):
    Separate features and target columns.

- plot_corr_matrix(quant_df, save_filename=None):
    Plot the correlation matrix.

- encode_categorical_columns(df, cat_cols):
    Encode categorical columns using the mean response approach and one-hot encoding.

- print_scores(cv_rfc, lrc, X_train, X_test, y_train, y_test, y_train_preds_rf, y_test_preds_rf, y_train_preds_lr, y_test_preds_lr):
    Print classification reports for model evaluation.

- plot_roc_curves(rfc_model, lrc, X_test, y_test):
    Plot ROC curves for models.

- plot_feature_importances(rfc_best_estimator, X_train):
    Plot feature importances of a Random Forest classifier.

- choose_predictors(data):
    Choose predictor columns for modeling.

- train_fit_models(X, y):
    Train and fit machine learning models.

Usage:
------
For usage instructions and examples of how to use these functions, please refer to the accompanying README.md file.

Note:
-----
Please ensure that the necessary libraries are imported before using the functions from this module.

"""

# import libraries
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, RocCurveDisplay

# Define categorical columns for global use
categorical_columns = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
]


def separate_columns():
    '''
    Separates columns into categorical and numerical lists.

    Returns:
        cat_columns (list): List of categorical column names.
        quant_columns (list): List of quantitative column names.
    '''
    cat_columns = categorical_columns

    quant_columns = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio'
    ]

    return cat_columns, quant_columns


def import_data(pth):
    '''
    Returns a DataFrame for the CSV found at pth.

    Args:
        pth (str): A path to the CSV file.

    Returns:
        df (pd.DataFrame): Pandas DataFrame containing the data from the CSV.
    '''
    df = pd.read_csv(pth)
    return df


def encode_categorical_columns(df, cat_cols):
    '''
    Encode categorical columns using the mean response approach and one-hot encoding.

    The "mean response approach" encodes categorical columns by mapping each category to the mean
    of the target variable (Churn) for that category. This approach captures the relationship
    between categorical features and the target variable.

    Args:
        df (pd.DataFrame): Pandas DataFrame containing the dataset.
        categorical_columns (list): List of categorical column names to be encoded.

    Returns:
        encoded_df (pd.DataFrame): Pandas DataFrame with encoded categorical columns.
    '''
    encoded_columns = [f'{col}_Churn' for col in cat_cols]

    response_mean_mapping = {}

    for col in cat_cols:
        response_mean_mapping[col] = df.groupby(col)['Churn'].mean()

    for col, encoded_col in zip(cat_cols, encoded_columns):
        df[encoded_col] = df[col].map(response_mean_mapping[col])

    # Handle remaining categorical columns using pd.get_dummies
    encoded_df = pd.get_dummies(df, columns=cat_cols, prefix=cat_cols)

    return encoded_df


def choose_predictors(df):
    """
    Choose specific predictors to keep in the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the data.

    Returns:
        df_selected (pd.DataFrame): DataFrame with selected predictor columns.
    """
    keep_cols = [
        'Customer_Age', 'Dependent_count', 'Months_on_book',
        'Total_Relationship_Count', 'Months_Inactive_12_mon',
        'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
        'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
        'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
        'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
        'Income_Category_Churn', 'Card_Category_Churn'
    ]

    df_selected = df[keep_cols]
    return df_selected


def plot_bivariate(data, x_column, y_column):
    """
    Create a bivariate scatter plot.

    Parameters:
        data (DataFrame): The DataFrame containing the data.
        x_column (str): The name of the column for the x-axis.
        y_column (str): The name of the column for the y-axis.

        Returns:
        None
    """
    plt.figure(figsize=(6, 3))
    sns.scatterplot(data=data, x=x_column, y=y_column)
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title(f'Bivariate Plot: {x_column} vs {y_column}')
    plt.savefig(f'bivariate_{x_column}_vs_{y_column}.png')
    plt.show()


def plot_univar_quant(data, column):
    """
    Create a histogram and kernel density estimate plot for a quantitative column.

    Args:
        data (pd.DataFrame): DataFrame containing the data.
        column (str): Name of the quantitative column to be plotted.

    Returns:
        None
    """
    plt.figure(figsize=(4, 3))
    sns.histplot(data[column], kde=True)
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.title(f'Univariate Quantitative Plot for {column}')
    plt.savefig(f'univariate_quantitative_{column}.png')
    plt.show()


def plot_roc_curves(rfc_model, lrc, X_test, y_test):
    """
    Plot ROC curves for RandomForestClassifier and LogisticRegression models.

    Args:
        rfc_model (RandomForestClassifier or str): The trained RandomForestClassifier model or path to a pickle file.
        lrc (LogisticRegression): The trained LogisticRegression model.
        X_test (pd.DataFrame): The test features.
        y_test (pd.Series): The true labels of the test data.

    Returns:
        None
    """
    plt.figure(figsize=(6, 4))
    ax = plt.gca()

    if isinstance(rfc_model, str):
        # Load RFC model from pickle file
        rfc_model = joblib.load(rfc_model)

    RocCurveDisplay.from_estimator(rfc_model, X_test, y_test, ax=ax)
    RocCurveDisplay.from_estimator(lrc, X_test, y_test, ax=ax)
    plt.savefig('roc_curves.png')
    plt.show()


def plot_corr_matrix(quant_df, save_filename=None):
    """
    Create and display a heatmap of the correlation matrix for quantitative columns.

    Args:
        quant_df (pd.DataFrame): DataFrame containing only quantitative columns.
        save_filename (str, optional): File name to save the plot as an image. Default is None.

    Returns:
        None
    """
    correlation_matrix = quant_df.corr()

    plt.figure(figsize=(6, 4))
    sns.heatmap(correlation_matrix, annot=False, cmap='Dark2_r', linewidths=2)

    if save_filename:
        plt.savefig(save_filename)

    plt.show()


def print_scores(
        cv_rfc,
        lrc,
        y_train,
        y_test,
        y_train_preds_rf,
        y_test_preds_rf,
        y_train_preds_lr,
        y_test_preds_lr):
    """
    Create a figure with two subplots displaying classification reports for model evaluation.

    Args:
        cv_rfc (model): Random Forest model from GridSearchCV.
        lrc (model): Logistic Regression model.
        y_train (array-like): True labels for the train set.
        y_test (array-like): True labels for the test set.
        y_train_preds_rf (array-like): Predicted labels for the training set using Random Forest.
        y_test_preds_rf (array-like): Predicted labels for the test set using Random Forest.
        y_train_preds_lr (array-like): Predicted labels for the training set using Logistic Regression.
        y_test_preds_lr (array-like): Predicted labels for the test set using Logistic Regression.

    Returns:
        None
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    models = [
        ('Random Forest', y_test_preds_rf, y_train_preds_rf, cv_rfc),
        ('Logistic Regression', y_test_preds_lr, y_train_preds_lr, lrc)
    ]

    for i, (model_name, y_test_preds, y_train_preds,
            dummy) in enumerate(models):
        ax = axes[i]
        y_train_name = 'Train'
        y_test_name = 'Test'

        ax.text(0.01, 1.25, f'{model_name} {y_train_name}', {
                'fontsize': 10}, fontproperties='monospace')
        ax.text(
            0.01, 0.05, str(
                classification_report(
                    y_train_preds, y_train)), {
                'fontsize': 10}, fontproperties='monospace')
        ax.text(0.01, 0.6, f'{model_name} {y_test_name}', {
                'fontsize': 10}, fontproperties='monospace')
        ax.text(
            0.01, 0.7, str(
                classification_report(
                    y_test_preds, y_test)), {
                'fontsize': 10}, fontproperties='monospace')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('model_scores.png')
    plt.show()
    dummy = 'functionally compulsory for this plot'


def plot_feature_importances(rfc_best_estimator, X_train):
    """
    Plot the feature importances of a Random Forest classifier.

    Args:
        rfc_best_estimator (model): Best Random Forest model from GridSearchCV.
        X_train (pd.DataFrame): Training features.

    Returns:
        None
    """
    plt.figure(figsize=(8, 6))
    importances = rfc_best_estimator.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.bar(range(X_train.shape[1]), importances[indices])
    plt.xticks(range(X_train.shape[1]), X_train.columns[indices], rotation=90)
    plt.xlabel('Features')
    plt.ylabel('Feature Importance')
    plt.title('Feature Importances')
    plt.tight_layout()
    plt.savefig('feature_importances.png')
    plt.show()


def train_fit_models(X, y):
    """
    Train and fit machine learning models.

    This function splits the data into training and testing sets, performs grid search
    for RandomForestClassifier, fits a Logistic Regression model, and makes predictions.

    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Target values.

    Returns:
        cv_rfc (GridSearchCV): Grid search instance for RandomForestClassifier.
        lrc (LogisticRegression): Fitted Logistic Regression model.
        X_train, X_test, y_train, y_test: Train and Test data splits.
        y_train_preds_rf (np.ndarray): Predicted values using RandomForest on training data.
        y_test_preds_rf (np.ndarray): Predicted values using RandomForest on testing data.
        y_train_preds_lr (np.ndarray): Predicted values using Logistic Regression on training data.
        y_test_preds_lr (np.ndarray): Predicted values using Logistic Regression on testing data.
    """
    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    # (reduced) Grid search for RandomForestClassifier
    rfc_param_grid = {
        'n_estimators': [200, 400],
        'max_features': ['sqrt'],
        'max_depth': [4, 5],
        'criterion': ['gini', 'entropy']
    }

    rfc = RandomForestClassifier(random_state=42)
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=rfc_param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    # (simple) Logistic Regression with optimized solver
    lrc = LogisticRegression(
        solver='liblinear',
        max_iter=1000,
        random_state=42)
    lrc.fit(X_train, y_train)

    # Predictions (best RFC vs. simple LRC)
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    return cv_rfc, lrc, X_train, X_test, y_train, y_test, y_train_preds_rf, y_test_preds_rf, y_train_preds_lr, y_test_preds_lr


# Define the main function to run our project
def main():

    # Load the dataset using the import_data function
    data_path = "./data/bank_data.csv"
    df = import_data(data_path)

    # Use the separate_columns function
    cat_columns, quant_columns = separate_columns()

    # Print the separated column lists
    print("Categorical Columns:", cat_columns)
    print("Quantitative Columns:", quant_columns)

    # Prepare churn from attrition
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    plot_univar_quant(df, 'Customer_Age')
    plot_univar_quant(df, 'Total_Trans_Ct')
    plot_bivariate(df, 'Customer_Age', 'Total_Trans_Amt')

    # Corr_Matrix: Select quantitative columns from the DataFrame
    quant_df = df[quant_columns]
    plot_corr_matrix(quant_df, save_filename='correlation_matrix.png')

    # get labels y
    y = df['Churn']

    # prepare empty pd fram for predictors
    X = pd.DataFrame()

    encode_categorical_columns(df, cat_columns)

    X = choose_predictors(df)

    # Train and Fit ML models
    cv_rfc, lrc, X_train, X_test, y_train, y_test, y_train_preds_rf, y_test_preds_rf, y_train_preds_lr, y_test_preds_lr = train_fit_models(
        X, y)

    print_scores(
        cv_rfc.best_estimator_,
        lrc,
        y_train,
        y_test,
        y_train_preds_rf,
        y_test_preds_rf,
        y_train_preds_lr,
        y_test_preds_lr)

    # cv_rfc.best_estimator_ or any RFC model
    plot_roc_curves(cv_rfc.best_estimator_, lrc, X_test, y_test)

    # Store best models in pkl files  => def f()
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    # Reload best models from pkl files  => def f()
    rfc_model = joblib.load('./models/rfc_model.pkl')
    lr_model = joblib.load('./models/logistic_model.pkl')

    # test the AU-ROCs of stored model vs. the prior best fit
    plot_roc_curves(rfc_model, lr_model, X_test, y_test)

    # Explain features via SHAP vals
    explainer = shap.TreeExplainer(cv_rfc.best_estimator_)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, plot_type="bar")

    # Rank feature importances
    plot_feature_importances(cv_rfc.best_estimator_, X_train)

    print("Project run completed.")


# The if __name__ == "__main__" block ensures that the main function is only executed if the script is run directly,
# but not when it's imported as a module in other scripts.
if __name__ == "__main__":
    main()
