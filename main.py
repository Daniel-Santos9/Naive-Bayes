#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 11:35:47 2019

@author: Daniel Santos
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from  sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

#importando dataset
base = pd.read_csv('Prostate_Cancer.csv')

#dividindo base entre previsores e classe
previsores = base.iloc[:,2:10].values
classe = base.iloc[:,1].values

#normalizando os dados
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

#alterando os dados categóricos em numeros
LE_classe = LabelEncoder()
classe[:] = LE_classe.fit_transform(classe[:])
class_number=classe.astype(str).astype(int)

#dividindo a base em train e test
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores,class_number, test_size=0.30,random_state=0)

#treinando e classificando os dados da base com o Naive Bayes 
classificador = GaussianNB()
classificador.fit(previsores_treinamento,classe_treinamento)
predict = classificador.predict(previsores_teste)

#calculando a acurácia
acc_dataTrain = classificador.score(previsores_treinamento, classe_treinamento)

acc_dataTest = classificador.score(previsores_teste,classe_teste) 

acc = accuracy_score(classe_teste, predict)

print(acc_dataTrain)
print(acc_dataTest)
print(acc)



