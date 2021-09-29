import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

# importing dataset
dataset = pd.read_csv('Data.csv')
print(dataset.head())

X = dataset.iloc[:, :-1].values  # feature variable
y = dataset.iloc[:, -1].values   # dependent variable

# splitting into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# creating regression model
classifier = GaussianNB()
classifier.fit(X_train, y_train)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)


# testing accuracy
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy of the Model is : {accuracy * 100: .2f} %', )