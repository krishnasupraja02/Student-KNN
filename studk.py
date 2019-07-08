import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

dt=pd.read_csv('knn.csv')
dt.head()

knn = KNeighborsClassifier(n_neighbors=4)

stud_cat = ["maths student", "physics student", "social student","biology student"]
data=dt.drop(['stu_id','type'],axis=1)
target=dt['type']
knn.fit(data,target)
#x=[[56,45,34,78],[89,78,44,48],[98,44,45,87]]
print(data)
prediction = knn.predict(data)
print(prediction)
#for val in prediction:
	#print(val)
#		print("Predicted subject: ",stud_cat[val])