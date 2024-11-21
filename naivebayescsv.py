import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

iris = pd.read_csv("Iris.csv")
x = iris.iloc[:, 1:-1].values
y = iris.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

C_NB = GaussianNB()
C_NB.fit(x_train, y_train)

y_pred = C_NB.predict(x_test)

print("Accuracy :", metrics.accuracy_score(y_test, y_pred))

print("Enter the sample data:")
a = float(input("Enter sepal length in cm = "))
b = float(input("Enter sepal width in cm = "))
c = float(input("Enter petal length in cm = "))
d = float(input("Enter petal width in cm = "))

Sample = [[a, b, c, d]]
pred = C_NB.predict(Sample)


print(pred)