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

from classifica_imoveis import classificaP,classificaPDB
from metodos import RandonForrest, DecisionTree, GradientBoost
from plotagens import plotBairro_TD,plotBairro_T, plot_TDB



#%%
# lendo o aquivo csv que foi exportado do banco 
imoveis = pd.read_csv('dados/imoveis_vendidos_cx.csv',sep=';',index_col=False)

#%%
# transforma os valores de preco em valores numericos
imoveis['preco'] = pd.to_numeric(imoveis['preco'], errors='coerce')
imoveis['QtdDormitorio'] = pd.to_numeric(imoveis['QtdDormitorio'], errors='coerce')
imoveis['QtdSuite'] = pd.to_numeric(imoveis['QtdSuite'], errors='coerce')
imoveis['QtdBanheiro'] = pd.to_numeric(imoveis['QtdBanheiro'], errors='coerce')
imoveis['QtdBoxes'] = pd.to_numeric(imoveis['QtdBoxes'], errors='coerce')
imoveis['AreaTotal'] = pd.to_numeric(imoveis['AreaTotal'], errors='coerce')
imoveis['AreaPrivativa'] = pd.to_numeric(imoveis['AreaPrivativa'], errors='coerce')

# remove colunas inuteis
imoveis = imoveis.drop(["referencia","descricao","areaUtil","numero","endereco","complemento","Estado","DataCadastro","preco_loca","precoCondominio"], axis=1)

# limpando valores NAN para 0 nas colunas:
fill = {'QtdDormitorio': 0,'QtdSuite': 0,'QtdBanheiro': 0,'QtdBoxes': 0,'AreaTotal': 0,'AreaPrivativa': 0, 'precoCondominio': 0}
imoveis.fillna(fill, inplace=True)

print(imoveis.isnull().sum())

# %%
# padronizando nome das cidades
imoveis['Cidade'] = imoveis['Cidade'].str.title()

imoveis["Cidade"].value_counts() 

#%%

# arrumando as colunas com dormitorio 0 e contendo suite
# pois se o imovel tem dorm = 0 e suite = 2, significa que ele possui 2 dorm sendo eles suites
def preencher_dorm(row):
    if row['QtdDormitorio'] == 0 and row['QtdSuite'] > 0:
        return row['QtdSuite']
    else:
        return row['QtdDormitorio']

imoveis['QtdDormitorio'] = imoveis.apply(preencher_dorm, axis=1)

# %%
# verifica e deleta colunas duplicadas
duplicados = imoveis.duplicated().sum()
if duplicados > 0:
    imoveis.drop_duplicates(inplace=True)

# verifica e deleta colunas com cidade nula
cidadeVazia = imoveis['Cidade'].isnull().sum()
if cidadeVazia > 0:
    imoveis = imoveis.drop(imoveis[imoveis['Cidade'].isnull()].index, axis=0)

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
imoveis = imoveis.drop(imoveis[imoveis['preco'].isnull()].index)

# limpa imoveis com preço menor que 10.000
imoveis = imoveis.loc[imoveis['preco'] >= 10000]

# %%
# dados faltantes por coluna depois da limpeza
print('dados faltantes por coluna depois da limpeza:')
print(imoveis.isna().sum().sort_values(ascending=False))

# porcentagem de dados faltantes por coluna depois da limpeza
print('porcentagem de dados faltantes por coluna depois da limpeza:')
print(((imoveis.isna().sum() / len(imoveis)) * 100).sort_values(ascending=False))


# %%
# verifica se ainda existe preco zerado
contagem_zeros = (imoveis['preco'] == 0).sum()
print(contagem_zeros)


print(imoveis.isnull().sum())

imoveis.describe()

# %%

# verifica os outlier dos preços
colunas = ['preco']
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

colunas = ['preco']
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

expl = lime.lime_tabular.LimeTabularExplainer(treinamento_x, feature_names=['id', 'Tipo', 'Cidade', 'Bairro', 'QtdDormitorio', 'QtdSuite', 'QtdBanheiro', 'QtdBoxes', 'AreaTotal', 'AreaPrivativa', 'preco'],class_names=['Baixo', 'Medio', 'Alto'])

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
