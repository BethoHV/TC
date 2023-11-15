# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import lime.lime_tabular
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
import sklearn.model_selection
import sklearn.utils.metaestimators

from classifica_imoveis_go import classificaP,classificaPDB,classifica_antigo
from metodos_go import RandonForrest, DecisionTree, GradientBoost
from plotagens_go import plotBairro_TD,plotBairro_T, plot_TDB



#%%
# lendo o aquivo csv que foi exportado do banco 

imoveis = pd.read_csv('../dados/imoveis_goiania.csv',sep=',',index_col=False)

nomes = {'DATE':'Data','PRICE':'Preco','ADDRESS':'Endereco','AREAS':'Area','BEDROOMS':'Dormitorios','PARKING-SPACES':'Garagens','BATHROOMS':'Banheiros','CONDOMÍNIO':'Condominio','IPTU':'IPTU','TIPO':'Tipo'}
imoveis = imoveis.rename(columns=nomes)

#%%
imoveis['Preco'] = imoveis['Preco'].str.replace('R$ ', '')
imoveis['Preco'] = imoveis['Preco'].str.replace('.', '')
imoveis['Area'] = imoveis['Area'].str.replace(' m²', '')

imoveis['Dormitorios'] = imoveis['Dormitorios'].str.split(' - ', expand=True)[0]
imoveis['Garagens'] = imoveis['Garagens'].str.split(' - ', expand=True)[0]
imoveis['Banheiros'] = imoveis['Banheiros'].str.split(' - ', expand=True)[0]
imoveis['Area'] = imoveis['Area'].str.split(' - ', expand=True)[0]

imoveis['Bairro'] = imoveis['Endereco'].str.split(', ', expand=True)[1]

#%%
# transforma os valores de preco em valores numericos
imoveis['Preco'] = pd.to_numeric(imoveis['Preco'], errors='coerce')
imoveis['Dormitorios'] = pd.to_numeric(imoveis['Dormitorios'], errors='coerce')
imoveis['Banheiros'] = pd.to_numeric(imoveis['Banheiros'], errors='coerce')
imoveis['Garagens'] = pd.to_numeric(imoveis['Garagens'], errors='coerce')
imoveis['Area'] = pd.to_numeric(imoveis['Area'], errors='coerce')

# remove colunas inuteis
imoveis = imoveis.drop(['Condominio','IPTU','Data','Endereco'], axis=1)
#%%
# limpando valores NAN para 0 nas colunas:
fill = {'Dormitorios': 0,'Banheiros': 0,'Garagens': 0,'Area': 0}
imoveis.fillna(fill, inplace=True)

print(imoveis.isnull().sum())

# %%
# padronizando nome das cidades

imoveis['Cidade'] = 'Goiânia'

imoveis['Cidade'] = imoveis['Cidade'].str.title()
imoveis["Cidade"].value_counts() 


# %%
# verifica e deleta colunas duplicadas
duplicados = imoveis.duplicated().sum()
if duplicados > 0:
    imoveis.drop_duplicates(inplace=True)

# verifica e deleta colunas com bairro nulo
bairroVazio = imoveis['Bairro'].isnull().sum()
if bairroVazio > 0:
    imoveis = imoveis.drop(imoveis[imoveis['Bairro'].isnull()].index, axis=0)


print(imoveis.isnull().sum())

# %%

# dados faltantes por coluna
print('dados faltantes por coluna:')
print(imoveis.isna().sum().sort_values(ascending=False))

# porcentagem de dados faltantes por coluna
print('porcentagem de dados faltantes por coluna:')
print(((imoveis.isna().sum() / len(imoveis)) * 100).sort_values(ascending=False))

# limpa imoveis com preço nan
imoveis = imoveis.drop(imoveis[imoveis['Preco'].isnull()].index)

# limpa imoveis com preço menor que 10.000
imoveis = imoveis.loc[imoveis['Preco'] >= 10000]

# %%
# dados faltantes por coluna depois da limpeza
print('dados faltantes por coluna depois da limpeza:')
print(imoveis.isna().sum().sort_values(ascending=False))

# porcentagem de dados faltantes por coluna depois da limpeza
print('porcentagem de dados faltantes por coluna depois da limpeza:')
print(((imoveis.isna().sum() / len(imoveis)) * 100).sort_values(ascending=False))


# %%
# verifica se ainda existe preco zerado
contagem_zeros = (imoveis['Preco'] == 0).sum()
print(contagem_zeros)


print(imoveis.isnull().sum())

imoveis.describe()

# %%

# verifica os outlier dos preços
colunas = ['Preco']
plt.boxplot(imoveis[colunas])
plt.xticks([1], colunas)
plt.title('Outlier antes de remover')
plt.show()
print(f'Total de colunas com outlier: {imoveis.shape[0]}')

# %%


Q1 = imoveis[colunas].quantile(0.25)
Q3 = imoveis[colunas].quantile(0.75)
IQR = Q3 - Q1

# elimina outlier do preço
imoveis = imoveis[~((imoveis[colunas] < (Q1 - 1.5 * IQR)) | (imoveis[colunas] > (Q3 + 1.5 * IQR))).any(axis=1)]

colunas = ['Preco']
plt.boxplot(imoveis[colunas])
plt.xticks([1], colunas)
plt.title('Outlier depois de remover')
plt.show()
print(f'Total de colunas com outlier: {imoveis.shape[0]}')


#%%
#plotagem de graficos:

# bairro por tipo e dormitorios
plotBairro_TD(imoveis)

#%%

# bairro por tipo
plotBairro_T(imoveis)

#%%

# bairro por tipo,dormitorios e banheiros
plot_TDB(imoveis)
# %%

# transformando valor das cidades e bairros em numericos 

label_encoder = LabelEncoder()
imoveis['Cidade'] = label_encoder.fit_transform(imoveis['Cidade'])
imoveis['Bairro'] = label_encoder.fit_transform(imoveis['Bairro'])
imoveis['Tipo'] = label_encoder.fit_transform(imoveis['Tipo'])


#%%
#classificação do padrão considerando apenas preço
classificaP(imoveis)
print(imoveis['Padrao'].value_counts())

# %%
#classificação do padrão considerando preço, dormitorios e quartos
classificaPDB(imoveis)
print(imoveis['Padrao'].value_counts())

# %%
#classificação do padrão considerando preço, dormitorios e quartos
classifica_antigo(imoveis)
print(imoveis['Padrao'].value_counts())



# %%
# utilizando metodo RandonForrest
RandonForrest(imoveis)

# %%
# utilizando metodo DecisionTree
DecisionTree(imoveis)

# %%
# utilizando metodo GradientBoosting
GradientBoost(imoveis)

# %%

numatributos = len(imoveis.columns) - 1
atributos = list(imoveis.columns[0:numatributos])

X = imoveis[atributos].values
y = imoveis['Padrao'].values

# dividindo os dados em conjuntos de treinamento e teste
treinamento_x, validacao_x, treinamento_y, validacao_y = train_test_split(X, y, test_size = 0.20)
print(treinamento_x)


#%%
# Criar um objeto Lime
classificador_forest = RandomForestClassifier()
resultado = classificador_forest.fit(treinamento_x, treinamento_y)

expl = lime.lime_tabular.LimeTabularExplainer(treinamento_x, feature_names=['index', 'Preco', 'Area', 'Dormitorios', 'Garagens', 'Banheiros', 'Tipo', 'Bairro', 'Cidade'],class_names=['Baixo', 'Medio', 'Alto'])

prever = lambda x: classificador_forest.predict_proba(x).astype(float)

print(validacao_x[1])
print(prever(validacao_x[1].reshape(1,-1)))
print(classificador_forest.predict(validacao_x[1].reshape(1,-1)))

previsto = classificador_forest.predict(validacao_x[5].reshape(1,-1))
probabilidades = classificador_forest.predict_proba(validacao_x[5].reshape(1,-1))
print(previsto)
print(probabilidades)
#%%
exp = expl.explain_instance(validacao_x[5], prever, num_features=4)
exp.show_in_notebook(show_all=True)


# %%

import eli5
from eli5 import show_prediction

classificador_forest = RandomForestClassifier()

eli5.show_weights(classificador_forest, feature_names = imoveis.columns)

# %%
