from sklearn.datasets import load_iris
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

iris=load_iris()
x=iris.data
y=iris.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=1)
G_nb = GaussianNB()
G_nb.fit(x_train,y_train)

y_pred = G_nb.predict(x_test)


print("accuracy",metrics.accuracy_score(y_test,y_pred))
print("enter sample data: ")
a = int(input("enter sepal length in cm : "))
b = int(input("enter sepal width in cm : "))
c = int(input("enter petal length in cm : "))
d = int(input("enter petal width in cm : "))

sample = [[a,b,c,d]]

pred = G_nb.predict(sample)
pred_v = [iris.target_names[p] for p in pred]
print(pred_v)