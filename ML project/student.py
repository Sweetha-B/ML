# linear_regression_models.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# 1. SIMPLE LINEAR REGRESSION
# -----------------------------
def simple_linear_regression():
    print("\n=== Simple Linear Regression ===")

    # Sample dataset: Hours studied vs Scores
    data = {
        'Hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'Scores': [15, 18, 21, 25, 30, 33, 37, 42, 46, 50]
    }

    df = pd.DataFrame(data)

    X = df[['Hours']]  # independent variable
    y = df['Scores']   # dependent variable

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Intercept:", model.intercept_)
    print("Coefficient:", model.coef_)
    print("R² Score:", r2_score(y_test, y_pred))

    # Visualization
    plt.scatter(X, y, color='blue', label="Actual")
    plt.plot(X, model.predict(X), color='red', label="Regression Line")
    plt.xlabel("Hours Studied")
    plt.ylabel("Score")
    plt.title("Simple Linear Regression")
    plt.legend()
    plt.show()


# -----------------------------
# 2. MULTIPLE LINEAR REGRESSION
# -----------------------------
def multiple_linear_regression():
    print("\n=== Multiple Linear Regression ===")

    # Sample dataset: Hours studied, Classes attended vs Score
    data = {
        'Hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'Classes_Attended': [2, 3, 4, 5, 6, 6, 7, 8, 9, 10],
        'Scores': [20, 25, 28, 35, 40, 43, 47, 52, 56, 60]
    }

    df = pd.DataFrame(data)

    X = df[['Hours', 'Classes_Attended']]
    y = df['Scores']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Intercept:", model.intercept_)
    print("Coefficients:", model.coef_)
    print("R² Score:", r2_score(y_test, y_pred))

    # 3D Plot (optional)
    try:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(df['Hours'], df['Classes_Attended'], df['Scores'], c='blue', label="Actual")
        ax.set_xlabel("Hours")
        ax.set_ylabel("Classes Attended")
        ax.set_zlabel("Scores")
        plt.title("Multiple Linear Regression (3D view)")
        plt.show()
    except ImportError:
        print("3D plot requires mpl_toolkits.mplot3d")


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    simple_linear_regression()
    multiple_linear_regression()
