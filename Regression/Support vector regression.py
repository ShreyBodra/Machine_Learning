import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score

# importing dataset
dataset = pd.read_excel('Folds5x2_pp.xlsx')

X = dataset.iloc[:, :-1].values  # feature variable
y = dataset.iloc[:, -1].values   # dependent variable
y = y.reshape(len(y),1)

# splitting into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# feature scaling
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
y_train = sc_y.fit_transform(y_train)

# creating regression model
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train, y_train.ravel())
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(X_test)))

# testing accuracy
accuracy = r2_score(y_test, y_pred)
print(f'Accuracy of the Model is : {accuracy * 100: .2f} %', )
