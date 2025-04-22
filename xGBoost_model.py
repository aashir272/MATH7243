"""
Created on Sun Mar 30 20:19:59 2025

@author: Ildi Hoxhallari
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#df = pd.read_csv('data.csv').dropna()
#df.sample(1000).to_csv('sample.csv', index=False)

sample_df = pd.read_csv('sample.csv')

predictors = sample_df[[
 'isPlayoffGame',
 'time',
 'timeUntilNextEvent',
 'timeSinceLastEvent',
 'period',
 'location',
 'shotAngleAdjusted',
 'shotDistance',
 'shotType',
 'shotOnEmptyNet',
 'speedFromLastEvent',
 'distanceFromLastEvent',
 'lastEventShotAngle',
 'lastEventShotDistance',
 'lastEventCategory',
 'lastEventTeam',
 'homeEmptyNet',
 'awayEmptyNet',
 'homeSkatersOnIce',
 'awaySkatersOnIce',
 'awayPenalty1TimeLeft',
 'awayPenalty1Length',
 'homePenalty1TimeLeft',
 'homePenalty1Length',
 'playerPositionThatDidEvent',
 'shooterLeftRight',
 'isHomeTeam',
 'teamCode']]

categorical_columns = [
    'location',  
    'shotType',   
    'lastEventCategory',   
    'lastEventTeam',   
    'playerPositionThatDidEvent',   
    'shooterLeftRight',  
    'teamCode']

numerical_columns = [
    'time',
    'timeUntilNextEvent',
    'timeSinceLastEvent',
    'shotAngleAdjusted',
    'shotDistance',
    'speedFromLastEvent',
    'distanceFromLastEvent',
    'lastEventShotAngle',
    'lastEventShotDistance',
    'awayPenalty1TimeLeft',
    'awayPenalty1Length',
    'homePenalty1TimeLeft',
    'homePenalty1Length']

def preprocess(X):

    # X_categorical = X[categorical_columns]
    # X_numerical = X[numerical_columns]

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))])

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_columns),
            ('cat', categorical_transformer, categorical_columns)])

    X_processed = preprocessor.fit_transform(X)

    categorical_column_names = preprocessor.transformers_[1][1].named_steps['onehot'].get_feature_names_out(categorical_columns)

    all_column_names = numerical_columns + list(categorical_column_names)

    X_processed_df = pd.DataFrame(X_processed, columns=all_column_names)
    
    X_processed_df[[
 'isPlayoffGame',
 'period',
 'shotOnEmptyNet',
 'homeEmptyNet',
 'awayEmptyNet',
 'homeSkatersOnIce',
 'awaySkatersOnIce',
 'isHomeTeam']] = X[[
 'isPlayoffGame',
 'period',
 'shotOnEmptyNet',
 'homeEmptyNet',
 'awayEmptyNet',
 'homeSkatersOnIce',
 'awaySkatersOnIce',
 'isHomeTeam']]
    
    return X_processed_df

X = preprocess(predictors)
y = sample_df['goal']

import xgboost as xgb
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, classification_report

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

df_train = pd.concat([X_train, y_train], axis=1)

df_minority_train = df_train[df_train['goal'] == 1]
df_majority_train = df_train[df_train['goal'] == 0]

df_majority_train_downsampled = resample(df_majority_train, 
                                         replace=False,    
                                         n_samples=len(df_minority_train),  
                                         random_state=42)

df_train_balanced = pd.concat([df_majority_train_downsampled, df_minority_train])

X_train_balanced = df_train_balanced.drop('goal', axis=1)
y_train_balanced = df_train_balanced['goal']

model = xgb.XGBClassifier(random_state=42)
model.fit(X_train_balanced, y_train_balanced)

y_pred = model.predict(X_test)

print("Accuracy on test set:", accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['No Goal', 'Goal'], yticklabels=['No Goal', 'Goal'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
