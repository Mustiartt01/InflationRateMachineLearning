from turtle import color
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

# Verisetini yükleme 2006 2020 Türkiye verileri
dataset = pd.read_csv('..\MP2\data.csv',sep=";")
dataset = dataset.apply(lambda x:x.str.replace(",","."))

# Verisertindeki attributeların türlerini değiştirme
dataset['issizlik'] = dataset['issizlik'].astype('float')
dataset['enflasyon'] = dataset['enflasyon'].astype('float')


# Veriseti Hakkında Bilgi edinme
print(dataset.shape)
print(" ")
print(dataset.describe())
print(" ")
dataset.info()
print(" ")

# X bagımsız degısklenler y bagımlı degıskenler temsil etmekte
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


# Test ve train verileri Bulunmakta X ve y kümeleri test_size yüzde 90 ogrenme yzude 10 test olarak bolunmesi
# X_train, X_test, y_train, y_test = train_test_split(x y, test_size=0.1, random_state=0)


# Basit Doğrusal Regresyon Algoritmaya train verilerinin gönderilmesi

# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

# regressor = LinearRegression()
# regressor.fit(X_train, y_train)

# lineer Regresyon egıtım modelını olusturmakta
lin_reg = LinearRegression()
lin_reg.fit(x,y)


# Eğim değeri
print('Egim Degeri: ',lin_reg.coef_)
print(" ")


# Kordinat sisteminde Dogrusal verilerin görünmesi
plt.scatter(x,y,color = 'red')
plt.plot(x,lin_reg.predict(x),color = 'blue')
plt.xlabel('İssizlik')
plt.ylabel('Enflasyon')
plt.show()


# X verisetini polinom denklemine çevirme
poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x)

lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)



# Kordinat sisteminde polinom verilerin görünmesi
plt.scatter(x,y,color = 'red')
plt.plot(x,lin_reg2.predict(poly_reg.fit_transform(x)),color = 'blue')
plt.xlabel('İssizlik')
plt.ylabel('Enflasyon')
plt.show()


# Veriler üzerinde tahmin yaparak dogruluk oranının tespiti
print(lin_reg.predict([[25]]))
print(lin_reg.predict([[5]]))
print(lin_reg.score(x,y))

print(" ")

print(lin_reg2.predict(poly_reg.fit_transform([[25]])))
print(lin_reg2.predict(poly_reg.fit_transform([[5]])))
print(lin_reg2.score(x_poly,y))