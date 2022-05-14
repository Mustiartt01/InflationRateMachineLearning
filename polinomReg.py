from ctypes import alignment
from doctest import master
from tokenize import String
from turtle import color
from unittest import result
from numpy import pad
import pandas as pd
import matplotlib.pyplot as plt
from pyparsing import col
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from tkinter import *

# Verisetini yükleme 2006 2020 Türkiye verileri
dataset = pd.read_csv('data.csv',sep=";")
dataset = dataset.apply(lambda x:x.str.replace(",","."))

# Verisertindeki attributeların türlerini değiştirme
dataset['issizlik'] = dataset['issizlik'].astype('float')
dataset['enflasyon'] = dataset['enflasyon'].astype('float')


# Veriseti Hakkında Bilgi edinme
# print(dataset.shape)
# print(" ")
# print(dataset.describe())
# print(" ")
# dataset.info()
# print(" ")

# X bagımsız degısklenler y bagımlı degıskenler temsil etmekte
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


# lineer Regresyon egıtım modelını olusturmakta
lin_reg = LinearRegression()
lin_reg.fit(x,y)

# Kordinat sisteminde Dogrusal verilerin görünmesi
# plt.scatter(x,y,color = 'red')
# plt.plot(x,lin_reg.predict(x),color = 'blue')
# plt.xlabel('İssizlik')
# plt.ylabel('Enflasyon')
# plt.show()


# X verisetini polinom denklemine çevirme
poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x)

poly_reg2= LinearRegression()
poly_reg2.fit(x_poly,y)



# Kordinat sisteminde polinom verilerin görünmesi
# plt.scatter(x,y,color = 'red')
# plt.plot(x,lin_reg2.predict(poly_reg.fit_transform(x)),color = 'blue')
# plt.xlabel('İssizlik')
# plt.ylabel('Enflasyon')
# plt.show()


# Veriler üzerinde tahmin yaparak dogruluk oranının tespiti

print(lin_reg.predict([[5]]))
print(lin_reg.score(x,y))

print(" ")

print(poly_reg2.predict(poly_reg.fit_transform([[5]])))
print(poly_reg2.score(x_poly,y))




app = Tk()

issizlik_type = float()

input_label = Label(app,text='Işşizlik Oranı',font=('bold',14),padx=20,pady=20)
input_label.grid(row=0,column=0,sticky=W)

issizlik_entry = Entry(app,textvariable=issizlik_type)
issizlik_entry.grid(row=0,column=1)



def Show_result():

    deneme1 = poly_reg2.predict(poly_reg.fit_transform([[issizlik_entry.get()]]))
    res = deneme1[0]

    myLabel = Label(app,text = res,font=('bold',14),padx=20,pady=20)
    myLabel.grid(row=2,column=1)
    

submit_btn = Button(app,text='SUBMIT',width=12,command= Show_result)
submit_btn.grid(row=1,column=1)

app.title('Enflasyon Tahmini')
app.geometry('400x400')

app.mainloop()