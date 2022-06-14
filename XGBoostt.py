import re
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from tkinter import *


dataset = pd.read_csv('data.csv',sep=";")
dataset = dataset.apply(lambda x:x.str.replace(",","."))

dataset['issizlik'] = dataset['issizlik'].astype('float')
dataset['enflasyon'] = dataset['enflasyon'].astype('float')


x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)

reg = XGBRegressor()

reg.fit(x,y)


# reg.fit(X_train,y_train)
# y_pred = reg.predict(X_test)
# print(type(y_pred))
# print(y_pred)
# print(reg.score(x,y))


app = Tk()

issizlik_type = float()

input_label = Label(app,text='Işşizlik Oranı',font=('bold',14),padx=20,pady=20)
input_label.grid(row=0,column=0,sticky=W)

input_label1 = Label(app,text='XGBoostReg Tahmin: ',font=('bold',14),padx=20,pady=20)
input_label1.grid(row=2,column=0,sticky=W)

input_label2 = Label(app,text='Score: ',font=('bold',14),padx=20,pady=20)
input_label2.grid(row=3,column=0,sticky=W)

issizlik_entry = Entry(app,textvariable=issizlik_type)
issizlik_entry.grid(row=0,column=1)



def Show_result():

    arr = np.array([float(issizlik_entry.get())])
    deneme1 = reg.predict(arr)

    myLabel = Label(app,text = deneme1[0],font=('bold',14),padx=20,pady=20)
    myLabel.grid(row=2,column=1)

    myLabel = Label(app,text = reg.score(x,y),font=('bold',14),padx=20,pady=20)
    myLabel.grid(row=3,column=1)

    

submit_btn = Button(app,text='SUBMIT',width=12,command= Show_result)
submit_btn.grid(row=1,column=1)

app.title('Enflasyon Tahmini')
app.geometry('400x400')

app.mainloop()

