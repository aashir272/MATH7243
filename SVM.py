
import sys
assert sys.version_info >= (3, 5)


import sklearn
assert sklearn.__version__ >= "0.20"


import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import os
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import sympy

np.random.seed(42)
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.model_selection import GridSearchCV




def clean_data(df):
    remove_numeric_columns = ['Unnamed: 0', 'shotID', 'game_id', 'id', 'goalieIdForShot','shooterPlayerId', 'HOMEZONE', 'AWAYZONE','location_Unknown','xCord','yCord','shotGeneratedRebound','homeTeamGoals','awayTeamGoals','xGoal', 'xFroze','xRebound','xPlayContinuedInZone','xPlayContinuedOutsideZone','xShotWasOnGoal', 'shotAngle', 'shotAngleReboundRoyalRoad','defendingTeamDefencemenOnIce', 'timeDifferenceSinceChange', 'averageRestDifference']
#'xCord','yCord','shotGeneratedRebound','homeTeamGoals','awayTeamGoals','xGoal', 'xFroze','xRebound','xPlayContinuedInZone','xPlayContinuedOutsideZone', 'PlayStopped','xShotWasOnGoal', 'shotAngle', 'shotAngleReboundRoyalRoad','goal'

    cleaned_df = df.select_dtypes(include=np.number)
    return cleaned_df.drop(columns=remove_numeric_columns)


 
def remove_dep_with_qr(df: pd.DataFrame, tol=1e-9) -> pd.DataFrame:
    arr = df.to_numpy().astype(float)
    # Q, R = np.linalg.qr(arr, mode='reduced') 
    # If we want the complete decomposition, we can do:
    Q, R = np.linalg.qr(arr)
    # Identify pivot columns from the diagonal of R
    diag = np.abs(np.diag(R))
    pivot_cols = np.where(diag > tol)[0]  # columns with diagonal > tolerance
    return df.iloc[:, pivot_cols]

if __name__ == '__main__':
    
    base_path = os.getcwd()
    path_train = os.path.join(base_path,'MATH7243','data', 'training_data.csv') 
    path_valid = f'{base_path}\\MATH7243\\data\\validation_data.csv'
    fig_save_path = f'{base_path}\\MATH7243\\plots\\logistic_regression'
    y_col = 'goal'

    # process data
    df_train = pd.read_csv(path_train, nrows= 3000)
    df_valid = pd.read_csv(path_valid, nrows=3000)
    df_train_cleaned = clean_data(df_train)
    df_valid_cleaned = clean_data(df_valid)
    y_train, y_valid = df_train[y_col].values, df_valid[y_col].values

    X_train, X_valid = df_train_cleaned.values, df_valid_cleaned.values
    ####################################
 

  #  required_columns = ['period', 'shotGoalieFroze', 'shotPlayStopped', 'ShotType', 'shotOnEmptyNet', 'shotRebound',                    'shotWasOnGoal', 'time', 'timeDifferenceSinceChange', 'averageRestDifference', 'timeSinceLastEvent',
       #             'speedFromLastEvent', 'lastEventxCord', 'lastEventyCord', 'distanceFromLastEvent', 'lastEventShotAngle',
        #            'lastEventShotDistance', 'LastEventCategory', 'shooterTimeOnIce', 'ShooterLeftRight', 
         #           'PlayerPositionThatDidEvent', 'defendingTeamForwardsOnIce', 'defendingTeamDefencemenOnIce', 
          #          'defendingTeamAverageTimeOnIce', 'defendingTeamAverageTimeOnIceOfForwards', 
           #         'defendingTeamAverageTimeOnIceOfDefencemen', 'defendingTeamAverageTimeOnIceSinceFaceoff', 
            #        'shootingTeamForwardsOnIce', 'shootingTeamDefencemenOnIce', 'shootingTeamAverageTimeOnIce', 
             #       'shootingTeamAverageTimeOnIceOfForwards', 'shootingTeamAverageTimeOnIceOfDefencemen', 
              #      'shotAngle', 'shotDistance']

  #  available_columns = [col for col in required_columns if col in df_train.columns]
   # dfnew= df_train[available_columns]

   # dfnewvalid= df_valid[available_columns]
   # y_train, y_valid = df_train[y_col].values, df_valid[y_col].values
    #X_train, X_valid = dfnew.values, dfnewvalid.values

indep= remove_dep_with_qr(df_train_cleaned) 


    
rank= np.linalg.matrix_rank(X_train)



scaler = StandardScaler()


X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.fit_transform(X_valid)



############################MODEL



svm_clf = SVC(kernel="linear",gamma='scale', C=1)
svm_clf.fit(X_train_scaled, y_train)

######################Predictions
y_pred = svm_clf.predict(X_valid_scaled)


print("Accuracy:", svm_clf.score(X_valid_scaled, y_valid))
#,class_weight='balanced'