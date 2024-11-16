import numpy as np
import pandas as pd
import seaborn as sns
import missingno as mso
import warnings
import os
import scipy

from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, classification_report, confusion_matrix
from scipy import stats
from scipy.stats import pearsonr, ttest_ind
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier

# Load the dataset
df = pd.read_csv('glucovision.csv')

# Display column names
print(df.columns)

# Data exploration
print(df.head())
print(df.describe())
print(df.info())
print(f"Shape of the dataset: {df.shape}")
print(f"Number of missing values:\n{df.isna().sum()}")
print(f"Number of unique values per column:\n{df.nunique()}")
print(f"Number of duplicated rows: {df.duplicated().sum()}")
print(f"Outcome value counts:\n{df['Outcome'].value_counts()}")

# List of numerical columns
num = df.select_dtypes(include=np.number).columns.tolist()

# Boxplot and distribution plot for each numerical column using seaborn
for col in num:
    sns.boxplot(x=df[col])
    sns.set(style="whitegrid")
    sns.despine(left=True, bottom=True)
    plt.title(f'Boxplot of {col}')
    plt.show()

for col in num:
    sns.histplot(x=df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()

# Normalizing specific numerical columns
num = ['Glucose', 'Age', 'BloodPressure', 'SkinThickness', 'Insulin']
scaler = MinMaxScaler()
df[num] = scaler.fit_transform(df[num])

# Splitting the data
X = df.drop(columns='Outcome', axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Visualization of the Outcome distribution
sns.countplot(x='Outcome', data=df)
plt.title('Outcome Distribution')
plt.show()

# Correlation heatmap
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title('Correlation Heatmap')
plt.show()

# Support Vector Machine (SVM) with linear kernel
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)

# Evaluating the SVM model
accuracy = accuracy_score(y_test, y_pred)
print(f"Akurasi SVM: {accuracy * 100:.2f}%")

f1 = f1_score(y_test, y_pred, average='weighted')
print(f"F1-Score: {f1}")

precision = precision_score(y_test, y_pred, average='weighted')
print(f"Precision: {precision}")

recall = recall_score(y_test, y_pred, average='weighted')
print(f"Recall: {recall}")

# Naive Bayes (Gaussian) Model
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)

# Evaluating the Naive Bayes model
accuracy = accuracy_score(y_test, y_pred)
print(f"Akurasi Naive Bayes: {accuracy * 100:.2f}%")

f1 = f1_score(y_test, y_pred, average='weighted')
print(f"F1-Score: {f1:.2f}")

precision = precision_score(y_test, y_pred, average='weighted')
print(f"Precision: {precision}")

recall = recall_score(y_test, y_pred, average='weighted')
print(f"Recall: {recall}")

# K-Nearest Neighbors (KNN) - Tuning for the best K value
scoreListknn = []
best_knn_model = None

for i in range(1, 21):
    KNclassifier = KNeighborsClassifier(n_neighbors=i)
    KNclassifier.fit(X_train, y_train)
    
    # Calculate accuracy and keep the best model
    score = KNclassifier.score(X_test, y_test)
    scoreListknn.append(score)
    
    if best_knn_model is None or score > best_knn_model.score(X_test, y_test):
        best_knn_model = KNclassifier

# Plotting the KNN accuracy
sns.lineplot(x=range(1, 21), y=scoreListknn, marker='o')
plt.xticks(np.arange(1, 21, 1))
plt.xlabel("K value")
plt.ylabel("Accuracy Score")
plt.title("KNN Accuracy for Different K Values")
plt.show()

knn_acc = max(scoreListknn)
print("KNN best accuracy: {:.2f}%".format(knn_acc * 100))

# Evaluating the best KNN model
y_pred = best_knn_model.predict(X_test)
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"F1-Score: {f1}")

precision = precision_score(y_test, y_pred, average='weighted')
print(f"Precision: {precision}")

recall = recall_score(y_test, y_pred, average='weighted')
print(f"Recall: {recall}")

# Sample prediction using Naive Bayes model
input_data = (2, 264, 70, 21, 176, 26.9, 0.671, 40)
input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)

prediction = nb.predict(input_data_as_numpy_array)
print(prediction)

if prediction[0] == 0:
    print('bukan penderita diabetes')
else:
    print('penderita diabetes')

# Save Naive Bayes model using pickle
import pickle

filename = 'trained_model.sav'
pickle.dump(nb, open(filename, 'wb'))

# Loading the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

# Predicting a sample data point using the loaded model
prediction = loaded_model.predict(input_data_as_numpy_array)
print(prediction)

if prediction[0] == 0:
    print('bukan penderita diabetes')
else:
    print('penderita diabetes')
