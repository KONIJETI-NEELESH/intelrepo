import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

#reading the CSV file and stores its contents in a dataset - "Pandas DataFrame object"
dataset = pd.read_csv('E:\Mobile Price Classification\Data/train.csv')

#selecting all rows and all columns except for the last column of the DataFrame and assigns it to the variable x.
x=dataset.iloc[:,:-1].values

#selecting all rows and only the last column of the DataFrame, representing the target variable, and assigns it to the variable y.
y=dataset.iloc[:,-1].values

#dividing the dataset in a way that 75% of the data is used for training (x_train and y_train), and the remaining 25% is used for testing (x_test and y_test)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)

sc=StandardScaler()
#computing the mean and standard deviation of the training data and scaling the feature values 
x_train = sc.fit_transform(x_train)
#applying the same scaling transformation to the testing data using the mean and standard deviation computed from the training data
x_test = sc.transform(x_test)
#now x_train and x_test will contain the scaled feature values.

#training an object 'classifier1' on the training data to make the predictions using K Nearest Neighbor Classifier
model = KNeighborsClassifier(n_neighbors=30,metric='minkowski')
model.fit(x_train,y_train)

#predicting labels for the testing data based on the trained K Nearest Neighbors Model.
y_pred = model.predict(x_test)

#displaying the accuracy score which represents the proportion of correctly predicted labels compared to the total number of labels in the testing data
print("KNN:",accuracy_score(y_test,y_pred))
