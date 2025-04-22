import io
import zipfile
import requests
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score, ConfusionMatrixDisplay

n = 1000
# df = pd.read_csv('shots_2012.csv')

url = "https://peter-tanner.com/moneypuck/downloads/shots_2012.zip"

response = requests.get(url)
zip_file = zipfile.ZipFile(io.BytesIO(response.content))
zip_file.extractall("extracted_files")

csv_file_name = [file for file in zip_file.namelist() if file.endswith('.csv')][0]
csv_path = f"extracted_files/{csv_file_name}"
df = pd.read_csv(csv_path)
print(df.head())

df = df.sample(n)

predictors = df[[
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

def prep_data(df, numerical_columns, categorical_columns):
    """
    Preprocesses a DataFrame by normalizing numerical columns and encoding categorical columns.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    numerical_columns (list): List of numerical column names.
    categorical_columns (list): List of categorical column names.

    Returns:
    pd.DataFrame: Preprocessed DataFrame with normalized numerical data and one-hot encoded categorical data.
    """
    # Preprocessing for numerical data: Normalize
    numerical_transformer = StandardScaler()

    # Preprocessing for categorical data: One-Hot Encode
    categorical_transformer = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    # Combine preprocessors in a column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_columns),
            ('cat', categorical_transformer, categorical_columns)
        ]
    )

    # Create a pipeline with the preprocessor
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

    # Fit and transform the data
    preprocessed_data = pipeline.fit_transform(df)

    # Convert the processed data to a DataFrame with column names
    preprocessed_df = pd.DataFrame(
        preprocessed_data, 
        columns=pipeline.named_steps['preprocessor'].get_feature_names_out()
    )

    return preprocessed_df

X = prep_data(predictors, numerical_columns, categorical_columns).dropna()
y = df['goal']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

model = MLPClassifier(hidden_layer_sizes=(64,32), activation='relu', solver='adam', max_iter=500, random_state=42)
model.fit(X_train, y_train)

# Calculate the Prediction Probability
y_proba = model.predict_proba(X_test)
y_scores = y_proba[:, 1]

y_pred = (y_scores >= 0.5).astype(int)

print("Goal Probability in the first 10 samples")
print(y_scores[:10])

print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

#Figures
#ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
auc_score = roc_auc_score(y_test, y_scores)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.2f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()

# Probility Distributions
plt.figure(figsize=(6, 4))
plt.hist(y_scores[y_test == 0], bins=20, alpha=0.6, label="Class 0 (No Goal)")
plt.hist(y_scores[y_test == 1], bins=20, alpha=0.6, label="Class 1 (Goal)")
plt.xlabel("Predicted Probability")
plt.ylabel("Count")
plt.title("Predicted Probability Distribution")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Confusion Matrix
cm_display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred), display_labels=["No Goal", "Goal"])
cm_display.plot(cmap="Blues")
plt.title("Confusion Matrix for MLPClassifier")
plt.grid(False)
plt.tight_layout()
plt.show()
