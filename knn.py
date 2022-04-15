import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier



# Verisetini yükleme 2006 2020 Türkiye verileri
dataset = pd.read_csv('..\MP2\data.csv',sep=";")
dataset = dataset.apply(lambda x:x.str.replace(",","."))

dataset['issizlik'] = dataset['issizlik'].astype('float')
dataset['enflasyon'] = dataset['enflasyon'].astype('float')


x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


# Test ve train verileri Bulunmakta X ve y kümeleri test_size yüzde 90 ogrenme yzude 10 test olarak bolunmesi
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)


# Basit Doğrusal Regresyon Algoritmaya train verilerinin gönderilmesi

knn = KNeighborsClassifier(n_neighbors=5,metric='minkowski')
knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print(cm)

