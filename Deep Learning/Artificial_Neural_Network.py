import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score

def data_preprocessing():
    dataset = pd.read_csv("Churn_Modelling.csv")
    x = dataset.iloc[:,3:-1].values  # feature variable
    y = dataset.iloc[:,-1].values  # dependent variable

    # encoding categorical data
    le = LabelEncoder()
    x[:,2] = le.fit_transform(x[:,2])

    ct = ColumnTransformer(transformers=[('encode', OneHotEncoder(), [1])], remainder='passthrough')
    x = np.array(ct.fit_transform(x))

    # splitting into training and testing dataset
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.2, random_state=0)

    # standardization
    sc =StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    building_ann(x_train, x_test, y_train, y_test)

def building_ann(x_train, x_test, y_train, y_test):
    ann = tf.keras.models.Sequential()

    # adding layers to model
    ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
    ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

    # output layer
    ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

    # compile model
    ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # fit model
    ann.fit(x_train, y_train, batch_size=32, epochs=25)

    # predicting for test set
    y_pred = ann.predict(x_test)
    y_pred = y_pred > 0.5
    # testing accuracy
    print(f'Accuracy of the model is : {accuracy_score(y_pred ,y_test)}')

if __name__ == "__main__":
    data_preprocessing()