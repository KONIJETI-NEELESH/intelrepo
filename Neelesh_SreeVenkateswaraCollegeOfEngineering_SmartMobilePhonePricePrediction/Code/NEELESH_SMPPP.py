import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

#reading the CSV file and stores its contents in a dataset - "Pandas DataFrame object"
dataset = pd.read_csv('E:\Mobile Price Classification\Data/train.csv')

#displaying the first five records of the dataset
print(dataset.head())

# computing summary statistics of the numerical columns in the dataset
print(dataset.describe())

#internal memory vs price range
print(sns.pointplot(y='int_memory',x='price_range',data=dataset,color='blue'))

#ram vs price range
print(sns.jointplot(y='ram',x='price_range',data=dataset,color='red',kind='kde'))

#battery power vs price range
sns.boxplot(x='battery_power',y='price_range',data=dataset)

#checking whether any NaN values are there or not
print(dataset.isna().sum())

#displaying correlation matrix of the dataset
corr_mat = dataset.corr()
plt.figure(figsize=(20,20))
print(sns.heatmap(corr_mat,annot=True))

#selecting all rows and all columns except for the last column of the DataFrame and assigns it to the variable x.
x=dataset.iloc[:,:-1].values

#selecting all rows and only the last column of the DataFrame, representing the target variable, and assigns it to the variable y.
y=dataset.iloc[:,-1].values

#displaying the first record of the feature values array 'x' and the target variable array 'y'
print(x[0])
print(y[0])

#dividing the dataset in a way that 75% of the data is used for training (x_train and y_train), and the remaining 25% is used for testing (x_test and y_test)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)

#displaying the shape of x_train(number of rows and columns)
print(x_test.shape)

#displaying the shape of x_test(number of rows and columns)
print(x_train.shape)

sc=StandardScaler()
#computing the mean and standard deviation of the training data and scaling the feature values 
x_train = sc.fit_transform(x_train)
#applying the same scaling transformation to the testing data using the mean and standard deviation computed from the training data
x_test = sc.transform(x_test)
#now x_train and x_test will contain the scaled feature values.

#training an object 'classifier1' on the training data to make the predictions using K Nearest Neighbor Classifier
classifier1 = KNeighborsClassifier(n_neighbors=30,metric='minkowski')
classifier1.fit(x_train,y_train)

#predicting labels for the testing data based on the trained K Nearest Neighbors Model.
y_pred1 = classifier1.predict(x_test)
#print(y_pred1)

#displaying the accuracy score which represents the proportion of correctly predicted labels compared to the total number of labels in the testing data
print("KNN",accuracy_score(y_test,y_pred1))

#training an object 'classifier2' on the training data to make the predictions using Random Forest Classifier
classifier2 = RandomForestClassifier(n_estimators = 25, criterion = 'entropy')
classifier2.fit(x_train,y_train)

#predicting labels for the testing data based on the trained random forest model.
y_pred2 = classifier2.predict(x_test)
#print(y_pred2)

#representing the confusion matrix, with the true labels on the y-axis and the predicted labels on the x-axis 
cm = confusion_matrix(y_test,y_pred2)
print(sns.heatmap(cm,annot=True))
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()

#displaying the accuracy score which represents the proportion of correctly predicted labels compared to the total number of labels in the testing data
print("RandomForest",accuracy_score(y_test,y_pred2))
# The accuracy score ranges from 0 to 1, where a higher score indicates better performance

# Initialize the SVM classifier
classifier3 = SVC()

# Train the model on the training data
classifier3.fit(x_train, y_train)

# Make predictions on the testing data
y_pred3 = classifier3.predict(x_test)

# Evaluate the model's accuracy using the accuracy_score metric
accuracy = accuracy_score(y_test, y_pred3)

# Print the accuracy score
print("SVM", accuracy)

# Assuming you have already split your data into training and testing sets: x_train, x_test, y_train, y_test

# Initialize the Linear Regression model
classifier4 = LinearRegression()

# Train the model on the training data
classifier4.fit(x_train, y_train)

# Make predictions on the testing data
y_pred4 = classifier4.predict(x_test)

# Evaluate the model's accuracy using metrics such as Mean Squared Error (MSE) and R-squared (R2)
mse = mean_squared_error(y_test, y_pred4)
r2 = r2_score(y_test, y_pred4)

# Print the evaluation metrics
print("Mean Squared Error:", mse)
print("LinearRegression", r2)

# Initialize the Logistic Regression model
classifier5 = LogisticRegression()

# Train the model on the training data
classifier5.fit(x_train, y_train)

# Make predictions on the testing data
y_pred5 = classifier5.predict(x_test)

# Evaluate the model's accuracy using the accuracy_score metric
accuracy = accuracy_score(y_test, y_pred5)

# Print the accuracy score
print("LogisticRegression:", accuracy)

#print("KNN,RandomForest,SVM,LinearRegresion,LogisticRegression")

