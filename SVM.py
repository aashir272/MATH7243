
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
from sklearn.decomposition import PCA
import sympy
from sklearn.svm import LinearSVC

np.random.seed(42)
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score



def clean_data(df):
    remove_numeric_columns = ['Unnamed: 0', 'shotID', 'game_id', 'id', 'goalieIdForShot','shooterPlayerId', 'HOMEZONE', 'AWAYZONE','location_Unknown','xCord','yCord','shotGeneratedRebound','homeTeamGoals','awayTeamGoals','xGoal', 'xFroze','xRebound','xPlayContinuedInZone','xPlayContinuedOutsideZone','xShotWasOnGoal', 'shotAngle', 'shotAngleReboundRoyalRoad','defendingTeamDefencemenOnIce', 'timeDifferenceSinceChange', 'averageRestDifference','goal','season', 'homeTeamWon','shotPlayContinuedOutsideZone', 'shotPlayContinuedInZone','shotPlayStopped', 'speedFromLastEvent', 'lastEventxCord', 'lastEventyCord', 'homeEmptyNet', 'awayEmptyNet', 'homeSkatersOnIce', 'awaySkatersOnIce', 'awayPenalty1TimeLeft', 'awayPenalty1Length', 'homePenalty1TimeLeft', 'homePenalty1Length', 'playerNumThatDidEvent', 'playerNumThatDidLastEvent', 'offWing', 'arenaAdjustedShotDistance', 'arenaAdjustedXCord', 'arenaAdjustedYCord', 'arenaAdjustedYCordAbs', 'xPlayStopped', 'isHomeTeam', 'arenaAdjustedXCordABS', 'location_AWAYZONE', 'location_HOMEZONE', 'location_Neu. Zone']
#'xCord','yCord','shotGeneratedRebound','homeTeamGoals','awayTeamGoals','xGoal', 'xFroze','xRebound','xPlayContinuedInZone','xPlayContinuedOutsideZone', 'PlayStopped','xShotWasOnGoal', 'shotAngle', 'shotAngleReboundRoyalRoad','goal'

# 'season', 'homeTeamWon','shotPlayContinuedOutsideZone', 'shotPlayContinuedInZone','shotPlayStopped', 'speedFromLastEvent', 'lastEventxCord', 'lastEventyCord', 'homeEmptyNet', 'awayEmptyNet', 'homeSkatersOnIce', 'awaySkatersOnIce', 'awayPenalty1TimeLeft', 'awayPenalty1Length', 'homePenalty1TimeLeft', 'homePenalty1Length', 'playerNumThatDidEvent', 'playerNumThatDidLastEvent', 'offWing', 'arenaAdjustedShotDistance', 'arenaAdjustedXCord', 'arenaAdjustedYCord', 'arenaAdjustedYCordAbs', 'xPlayStopped', 'isHomeTeam', 'arenaAdjustedXCordABS', 'location_AWAYZONE', 'location_HOMEZONE', 'location_Neu. Zone'

    cleaned_df = df.select_dtypes(include=np.number)
    return cleaned_df.drop(columns=remove_numeric_columns)




 
def remove_dep_with_qr(df: pd.DataFrame, tol=1e-9) -> pd.DataFrame:
    arr = df.to_numpy().astype(float)
    # Q, R = np.linalg.qr(arr, mode='reduced') 
    Q, R = np.linalg.qr(arr)
    diag = np.abs(np.diag(R))
    pivot_cols = np.where(diag > tol)[0]  # columns with diagonal > tolerance
    return df.iloc[:, pivot_cols]

if __name__ == '__main__':
    
    base_path = os.getcwd()
    path_train = os.path.join(base_path,'data', 'training_data.csv') 
    path_valid = f'{base_path}\\data\\validation_data.csv'
    fig_save_path = f'{base_path}\\plots\\logistic_regression'
    y_col = 'goal'

    # process data
    df_trainunsampled = pd.read_csv(path_train)
    df_validunsampled = pd.read_csv(path_valid)


###############################sampled data for SVC not linear SVC
 #   df_train_1= df_trainunsampled[df_trainunsampled['goal'] == 1]
  #  df_train_0= df_trainunsampled[df_trainunsampled['goal'] == 0]
   # dftrain_1s= df_train_1.sample(n=48000, random_state=42)             
   # dftrain_0s= df_train_0.sample(n=48000, random_state=42)   

    #df_valid_1= df_trainunsampled[df_trainunsampled['goal'] == 1]
    #df_valid_0= df_validunsampled[df_validunsampled['goal'] == 0]
    #dfvalid_1s= df_valid_1.sample(n=10000, random_state=42)
    #dfvalid_0s= df_valid_0.sample(n=10000, random_state=42)

    df_train= df_trainunsampled #pd.concat([dftrain_1s, dftrain_0s], ignore_index=True)
    df_valid= df_validunsampled #pd.concat([dfvalid_1s, dfvalid_0s], ignore_index=True)


    df_train_cleaned = clean_data(df_train)
    df_valid_cleaned = clean_data(df_valid)
    
    y_train, y_valid = df_train[y_col].values, df_valid[y_col].values
    X_train, X_valid = df_train_cleaned.values, df_valid_cleaned.values

scaler =  StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)

#clf = LinearSVC(C=100, max_iter=10000)
#clf.fit(X_train_scaled, y_train)
#acc = clf.score(X_valid_scaled, y_valid)
#print(f"Accuracy without PCA = {acc:.4f}")


#y_pred = clf.predict(X_valid_scaled)



####################PCA


pca = PCA(n_components=30) ############################# NUMBER OF PCA COMPONENTS
X_train_pca = pca.fit_transform(X_train_scaled)
X_valid_pca = pca.transform(X_valid_scaled)

feature_names = df_train_cleaned.columns

# PCA results###################################
feature_names = df_train_cleaned.columns
loadings_df = pd.DataFrame(pca.components_, columns=feature_names)
for i, component in enumerate(loadings_df.values):
    sorted_indices = np.argsort(np.abs(component))[::-1]  
    top_features = feature_names[sorted_indices[:5]]
    top_values = component[sorted_indices[:5]]
    
    print(f"Principal Component {i+1}")
    for feature, value in zip(top_features, top_values):
        direction = "↑" if value > 0 else "↓"
        print(f"  {feature:40s} {direction} ({value:.4f})")

print(f"Explained variance by each component: {pca.explained_variance_ratio_}")
print(f"Total variance explained by the selected components: {np.sum(pca.explained_variance_ratio_):.4f}")



# SVM AFTER PCA #############################################################
C_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
accuracies = []

for C in C_values:
    clf = LinearSVC(C=C, max_iter=10000)
    clf.fit(X_train_pca, y_train)
    acc = clf.score(X_valid_pca, y_valid)
    accuracies.append(acc)
    print(f"C = {C:7} → Accuracy = {acc:.4f}")

#######################PLOT 
plt.figure(figsize=(8, 5))
plt.plot(C_values, accuracies, marker='o')
plt.xscale('log')
plt.xlabel('C')
plt.ylabel('Validation Accuracy')
plt.title('Effect of L2 Regularization Strength (C) on Accuracy')
plt.grid(True)
plt.tight_layout()
plt.show()



#############################TEST variables

clf = LinearSVC(C=10, max_iter=10000)
clf.fit(X_train_pca, y_train)
y_pred = clf.predict(X_valid_pca)


# confusion matrix
cm = confusion_matrix(y_valid, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

# Plot
plt.figure(figsize=(6, 5))
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix for SVM")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.grid(False)
plt.tight_layout()
plt.show()




print(classification_report(y_valid, y_pred, target_names=["No Goal", "Goal"]))

####################################BAlanced class weights#############################
clf = LinearSVC(C=10, class_weight='balanced', max_iter=10000)
clf.fit(X_train_pca, y_train)
y_pred = clf.predict(X_valid_pca)
acc = clf.score(X_valid_pca, y_valid)
print(f"Balancede accuracy:{acc}")

cm = confusion_matrix(y_valid, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

# Plot
plt.figure(figsize=(6, 5))
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix for SVM balanced")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.grid(False)
plt.tight_layout()
plt.show()

print('balanced report')
print(classification_report(y_valid, y_pred, target_names=["No Goal", "Goal"]))


calibrated = CalibratedClassifierCV(clf, cv=5)
calibrated.fit(X_train_pca, y_train)

probs = calibrated.predict_proba(X_valid_pca)[:, 1]

precisions, recalls, thresholds = precision_recall_curve(y_valid, probs)

# Plot######################
plt.plot(thresholds, precisions[:-1], label="Precision")
plt.plot(thresholds, recalls[:-1], label="Recall")
plt.xlabel("Threshold")
plt.legend()
plt.title("Precision-Recall vs. Threshold")
plt.grid()
plt.show()
######################################


decision_scores = clf.decision_function(X_valid_pca)

threshold = 0.3
y_pred_custom = (decision_scores > threshold).astype(int)

accuracy = accuracy_score(y_valid, y_pred_custom)
print(f"Custom threshold accuracy (threshold={threshold}): {accuracy:.4f}")

precision = precision_score(y_valid, y_pred_custom)
recall = recall_score(y_valid, y_pred_custom)
f1 = f1_score(y_valid, y_pred_custom)

print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

#svm_clf = LinearSVC(C=100, max_iter=10000)  # SVC(kernel="linear", C=10)
#svm_clf.fit(X_train_pca, y_train)
#y_pred = svm_clf.predict(X_valid_pca)


#accuracy = svm_clf.score(X_valid_pca, y_valid)
#print(f"SVM Accuracy after PCA transformation: {accuracy:.4f}")




#################GRID SEARCH#######################################


#param_grid = {
#    'C': [0.1, 1, 10, 100],
#    'kernel': ['linear'], #, 'rbf', 'poly'],
#    'gamma': ['scale', 'auto'],
#}

#grid_search = GridSearchCV(SVC(), param_grid, cv=5, n_jobs=-1)
#print(f"Best parameters: {grid_search.best_params_}")
#Best parameters: {'C': 100, 'gamma': 'scale', 'kernel': 'linear'}

#best_svm = grid_search.best_estimator_
#test_accuracy = best_svm.score(X_valid_pca, y_valid)
#print(f"Test accuracy with best parameters: {test_accuracy:.4f}")


########################################################################


############################MODEL


#svm_clf = SVC(kernel="linear",gamma='scale', C=100)
#svm_clf.fit(X_train_pca, y_train)

######################Predictions########################################
#y_pred = svm_clf.predict(X_valid_pca)

#score = svm_clf.score(X_valid_pca, y_valid)

#print("Accuracy:", score)
