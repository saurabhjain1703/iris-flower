
# libraries used
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# upload dataset
url= "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length','sepal-width','petal-length','petal-width','class']
dataset = pandas.read_csv(url,names=names)

# split-out validation dataset
array=dataset.values
X = array[:,0:4]
#print(X)
Y=array[:,4]
#print(Y)
validation_size = 0.20
seed = 7
X_train,X_validation,Y_train,Y_validation = model_selection.train_test_split(X,Y,test_size=validation_size,random_state=seed)

#check accuracy
seed=7
scoring = 'accuracy'

# spot check algorithms

models = []
models.append(('LR',LogisticRegression()))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('CART',DecisionTreeClassifier()))
models.append(('NB',GaussianNB()))
models.append(('SVM',SVC()))

# evaluate each model

results = []
names = []
for name,model in models:
    kfold = model_selection.KFold(n_splits=10,random_state=seed)
    cv_results = model_selection.cross_val_score(model,X_train,Y_train,cv=kfold,scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name,cv_results.mean(),cv_results.std())
    #print(msg)
    
# make prediction on validation dataset
a=[]
b=list(map(float,input().split()))
a.append(b)
knn = KNeighborsClassifier()
knn.fit(X_train,Y_train)
predictions = knn.predict(a)
print(predictions)
#print((accuracy_score(Y_validation,predictions))*100)
#print(classification_report(Y_validation,predictions))
