import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, ConfusionMatrixDisplay

chunk_size = 1000
chunk_file = pd.read_csv('/Users/Bluewiseid/7243 Final Project/training_data.csv', chunksize=chunk_size)

first_chunk = next(chunk_file)

# print(first_chunk.dtypes)
# print(first_chunk.columns.tolist())

#Check Column Index
for idx, col in enumerate(first_chunk.columns):
    print(f"{idx}: {col}")

selected_columns_indecies = list(set([
    17, 115, 26, 40, 41, 42, 44, 22, 34, 35, 64, 65, 48, 13, 49, 50, 55, 56, 53, 54, 114, 11, 12, 26, 27, 28, 29, 30, 31, 32
]))

df_select = first_chunk.iloc[:, selected_columns_indecies].dropna()

X = df_select.iloc[:, 1:]
y = df_select.iloc[:, 0]

X_Scaled = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_Scaled, y, test_size=0.2, random_state = 42)

model = MLPClassifier(hidden_layer_sizes=(64,32), activation='relu', solver='adam', max_iter=500, random_state=42)
model.fit(X_train, y_train)

# Calculate the Prediction Probability
y_proba = model.predict_proba(X_test)
y_scores = y_proba[:, 1]

y_pred = (y_scores >= 0.5).astype(int)

#Print Prob
print("Goal Probability in the first 10 samples")
print(y_scores[:10])

#Print Classification Report
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
plt.title("Confusion Matrix")
plt.grid(False)
plt.tight_layout()
plt.show()