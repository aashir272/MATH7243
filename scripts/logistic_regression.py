import pandas as pd
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import matplotlib

def clean_data(df):
    remove_numeric_columns = ['Unnamed: 0', 'shotID', 'game_id', 'id', 'goalieIdForShot',
                              'shooterPlayerId', 'goal', 'HOMEZONE', 'AWAYZONE']
    cleaned_df = df.select_dtypes(include=np.number)
    return cleaned_df.drop(columns=remove_numeric_columns)

def plot_errors(errors, lambdas, title='', save_path=''):

    plt.figure()
    plt.plot(lambdas, errors, label='Validation error')
    plt.xlabel('Regularization strength')
    plt.ylabel('Error')
    plt.title(title)
    plt.savefig(save_path)


if __name__ == '__main__':
    
    base_path = os.getcwd()
    path_train = f'{base_path}\\data\\training_data.csv'
    path_valid = f'{base_path}\\data\\validation_data.csv'
    fig_save_path = f'{base_path}\\plots\\logistic_regression'
    y_col = 'goal'

    # process data
    df_train = pd.read_csv(path_train)
    df_valid = pd.read_csv(path_valid)
    df_train_cleaned = clean_data(df_train)
    df_valid_cleaned = clean_data(df_valid)
    X_train, X_valid = df_train_cleaned.values, df_valid_cleaned.values
    y_train, y_valid = df_train[y_col].values, df_valid[y_col].values

    # plot metrics by lambda
    lambdas = [0.01, 0.1, 0.5, 1, 2, 5, 10]
    penalties = ['l1', 'l2']

    for penalty in penalties:
        mse = []
        mae = []
        mape = []
        print(penalty)
        for C in lambdas:
            print(C)

            # choose solver
            if penalty == 'l1':
                solver = 'liblinear'
            else:
                solver = 'newton-cholesky'

            model = LogisticRegression(penalty=penalty, C=1/C, solver=solver, max_iter=250).fit(X_train, y_train)
            prediction = model.predict(X_valid)
            predicted_prob = model.predict_proba(X_valid)

            # expected goals = p(goal=1)
            expected_goals_valid = predicted_prob[:,1]

            # compute error metrics
            mse.append(mean_squared_error(y_valid, expected_goals_valid))
            mae.append(mean_absolute_error(y_valid, expected_goals_valid))
            mape.append(mean_absolute_percentage_error(y_valid, expected_goals_valid))
        
        plot_errors(mse, lambdas, title=f'Validation MSE by {penalty} penalty', save_path=f'{fig_save_path}\\validation_{penalty}_mse.png')
        plot_errors(mae, lambdas, title=f'Validation MAE by {penalty} penalty', save_path=f'{fig_save_path}\\validation_{penalty}_mae.png')
        plot_errors(mape, lambdas, title=f'Validation MAPE by {penalty} penalty', save_path=f'{fig_save_path}\\validation_{penalty}_mape.png')
