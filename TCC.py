import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np 
rfr = RandomForestRegressor()
from sklearn import metrics 
arv = tree.DecisionTreeRegressor() 

st.title('Predição Da Produção Agrícola Com Base Em Dados Pluviométricos Para A Cidade De Maringá-PR')
df2 = pd.read_csv("df2.csv")
porcent_test = 0.3
X_train, x_test = train_test_split(df2, test_size=porcent_test)
cols_train_x=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
cols_train_y=['Producao']

arv.fit(X_train[cols_train_x].values,X_train[cols_train_y].values)
x=arv.predict(x_test[cols_train_x].values)

rfr.fit(X_train[cols_train_x].values,X_train[cols_train_y].values)
y=rfr.predict(x_test[cols_train_x].values)

st.write('Predição com Árvore de decisão')

fig, ax = plt.subplots(figsize=(10,5))           
ax.scatter(df2['ANO'], df2['Producao'], label='Produção')
plt.xlabel('Anos')  
plt.ylabel('Produção (Milhões de toneladas)')
plt.title('Produção(T) por Ano')
plt.plot(x_test.index+2007, x,'b.', color='r', label='Predição')
ax.legend()
st.pyplot(fig)

st.write('Predição com Floresta aleatória')

fig, ax = plt.subplots(figsize=(10,5))           
ax.scatter(df2['ANO'], df2['Producao'], label='Produção')
plt.xlabel('Anos')  
plt.ylabel('Produção (Milhões de toneladas)')
plt.title('Produção(T) por Ano')
plt.plot(x_test.index+2007, y,'b.', color='r', label='Predição')
ax.legend()
st.pyplot(fig)

st.write('Diferença entre Árvore de decisão e Floresta aleatória')

fig, ax = plt.subplots(figsize=(10,5))
ax.scatter(df2['ANO'], df2['Producao'], label='Produção')          
plt.xlabel('Anos')  
plt.ylabel('Produção (Milhões de toneladas)')
plt.title('Produção(T) por Ano')
plt.plot(x_test.index+2007, y,'b.', color='r', label='Floresta aleatória')
plt.plot(x_test.index+2007, x,'g*', color='g', label='Árvore de decisão')
ax.legend()
st.pyplot(fig)

ano=st.number_input('qual ano sera feita a predicao?')

ja=st.number_input('Quanto choveu em janeiro desse ano?')
fe=st.number_input('Quanto choveu em fevereiro desse ano?')
ma=st.number_input('Quanto choveu em marco desse ano?')
ab=st.number_input('Quanto choveu em abril desse ano?')
mai=st.number_input('Quanto choveu em maio desse ano?')
jun=st.number_input('Quanto choveu em junho desse ano?')
jul=st.number_input('Quanto choveu em julho desse ano?')
ago=st.number_input('Quanto choveu em agosto desse ano?')
se=st.number_input('Quanto choveu em setembro desse ano?')
out=st.number_input('Quanto choveu em outubro desse ano?')
nov=st.number_input('Quanto choveu em novembro desse ano?')
dez=st.number_input('Quanto choveu em dezembro desse ano?')

a=np.array([[ja,fe,ma,ab,mai,jun,jul,ago,se,out,nov,dez]])

z=rfr.predict(a.values)

st.write("A producao estimada para {}, é de {} toneladas".format(ano,z))