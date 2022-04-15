import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler




# Verisetini yükleme 2006 2020 Türkiye verileri
dataset = pd.read_csv('..\MP2\data1.csv',sep=";")
dataset = dataset.apply(lambda x:x.str.replace(",","."))


# Verisetindeki değişkenlerin türünü değiştirme
dataset['issizlik'] = dataset['issizlik'].astype('float')
dataset['enflasyon'] = dataset['enflasyon'].astype('float')


# Veriseti Hakkında Bilgi edinme
dataset.shape
dataset.describe()
dataset.info()
print(" ")


X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Test ve train verileri Bulunmakta X ve y kümeleri test_size yüzde 90 ogrenme yzude 10 test olarak bolunmesi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)


# Basit Doğrusal Regresyon Algoritmaya train verilerinin gönderilmesi

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

regressor = LinearRegression()
regressor.fit(X_train, y_train)


r_sq = regressor.score(X, y)
print('Dogruluk orani:', r_sq)
print(" ")


# Eğim değeri
print('Egim Degeri: ',regressor.coef_)
print(" ")

# Enflasyon için tahminde bulunma
y_pred = regressor.predict(X_test)

# Tahmin edilen ve Gerçek verilerin Karşılaştırılması
df = pd.DataFrame({'Gercek': y_test, 'Tahmin': y_pred})
print(df)
print(" ")

# print('Ortalama Mutlak Hatası:', metrics.mean_absolute_error(y_test, y_pred))

