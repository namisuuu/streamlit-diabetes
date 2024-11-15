import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import missingno as mso
import seaborn as sns
import warnings
import os
import scipy

from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from scipy import stats
from scipy.stats import pearsonr
from scipy.stats import ttest_ind
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

df= pd.read_csv('diabetes.csv')

df.columns
Index(['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'],
      dtype='object')

df.head()
df.describe()
df.info()
df.shape
df.isna().sum()
df.nunique()
df.duplicated().sum()
df['Outcome'].value_counts()
num = df.select_dtypes(include=np.number).columns.tolist()


for col in num:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
plt.show()

for col in num:
    plt.figure(figsize=(8, 6))  # Adjust figure size if needed
    sns.histplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()

from sklearn.preprocessing import MinMaxScaler
num= num = ['Glucose','Age','BloodPressure','SkinThickness','Insulin']

scaler = MinMaxScaler()
df[num] = scaler.fit_transform(df[num])
df

x = df.drop(columns = 'Outcome', axis=1)
y = df['Outcome']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, stratify=y, random_state=2)

sns.countplot(x='Outcome', data=df)

sns.heatmap(df.corr(), annot=True, cmap="coolwarm")

from sklearn.svm import SVC
# Inisialisasi model SVM dengan kernel linear
svm = SVC(kernel='linear')

# Train model
svm.fit(X_train, y_train)

# Lakukan prediksi pada data testing
y_pred = svm.predict(X_test)

# Evaluasi akurasi model
accuracy = accuracy_score(y_test, y_pred)
print(f"Akurasi SVM: {accuracy * 100:.2f}%")
# Menghitung F1-score
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"F1-Score: {f1}")

# Menghitung precision
precision = precision_score(y_test, y_pred, average='weighted')
print(f"Precision: {precision}")

# Menghitung recall
recall = recall_score(y_test, y_pred, average='weighted')
print(f"Recall: {recall}")

# Inisialisasi model Naive Bayes (Gaussian)
nb = GaussianNB()

# Train model
nb.fit(X_train, y_train)

# Lakukan prediksi pada data testing
y_pred = nb.predict(X_test)

# Evaluasi akurasi model
accuracy = accuracy_score(y_test, y_pred)
print(f"Akurasi Naive Bayes: {accuracy * 100:.2f}%")

# Menghitung F1-score
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"F1-Score: {f1:.2f}")

# Menghitung precision
precision = precision_score(y_test, y_pred, average='weighted')
print(f"Precision: {precision}")

# Menghitung recall
recall = recall_score(y_test, y_pred, average='weighted')
print(f"Recall: {recall}")

scoreListknn = []
best_knn_model = None

for i in range(1, 21):
    KNclassifier = KNeighborsClassifier(n_neighbors=i)
    KNclassifier.fit(X_train, y_train)

    # Menghitung akurasi dan menyimpan model terbaik
    score = KNclassifier.score(X_test, y_test)
    scoreListknn.append(score)

    # Simpan model jika akurasinya lebih baik
    if best_knn_model is None or score > best_knn_model.score(X_test, y_test):
        best_knn_model = KNclassifier

# Plotting
plt.plot(range(1, 21), scoreListknn)
plt.xticks(np.arange(1, 21, 1))
plt.xlabel("K value")
plt.ylabel("Score")
plt.title("KNN Accuracy for Different K Values")
plt.show()

KNAcc = max(scoreListknn)
print("KNN best accuracy: {:.2f}%".format(KNAcc * 100))

# Menghitung F1-score menggunakan model terbaik
y_pred = best_knn_model.predict(X_test)
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"F1-Score: {f1}")

# Menghitung precision
precision = precision_score(y_test, y_pred, average='weighted')
print(f"Precision: {precision}")

# Menghitung recall
recall = recall_score(y_test, y_pred, average='weighted')
print(f"Recall: {recall}")

input_data = (2,264,70,21,176,26.9,0.671,40)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediksi = nb.predict(input_data_reshaped)
print(prediksi)

if (prediksi[0] == 0):
  print('bukan penderita diabetes')
else:
  print('penderita diabetes')

import pickle

filename = 'trained_model.sav'
pickle.dump(nb, open(filename, 'wb'))

# loading the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

input_data = (2,264,70,21,176,26.9,0.671,40)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediksi = loaded_model.predict(input_data_reshaped)
print(prediksi)

if (prediksi[0] == 0):
  print('bukan penderita diabetes')
else:
  print('penderita diabetes')

