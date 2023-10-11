# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



# lendo o aquivo csv que foi exportado do banco 
dados_imoveis = pd.read_csv('dados/imoveis_vendidos_cx.csv',sep=';')

# transforma os valores de preco em valores numericos
dados_imoveis['preco'] = pd.to_numeric(dados_imoveis['preco'], errors='coerce')
dados_imoveis['QtdDormitorio'] = pd.to_numeric(dados_imoveis['QtdDormitorio'], errors='coerce')
dados_imoveis['QtdSuite'] = pd.to_numeric(dados_imoveis['QtdSuite'], errors='coerce')
dados_imoveis['QtdBanheiro'] = pd.to_numeric(dados_imoveis['QtdBanheiro'], errors='coerce')
dados_imoveis['QtdBoxes'] = pd.to_numeric(dados_imoveis['QtdBoxes'], errors='coerce')
dados_imoveis['AreaTotal'] = pd.to_numeric(dados_imoveis['AreaTotal'], errors='coerce')
dados_imoveis['AreaPrivativa'] = pd.to_numeric(dados_imoveis['AreaPrivativa'], errors='coerce')

# remove colunas inuteis
dados_imoveis = dados_imoveis.drop(["referencia","descricao","areaUtil","numero","endereco","complemento","Tipo","Estado","DataCadastro","preco_loca","precoCondominio"], axis=1)

# limpando valores NAN para 0 nas colunas:
fill = {'QtdDormitorio': 0,'QtdSuite': 0,'QtdBanheiro': 0,'QtdBoxes': 0,'AreaTotal': 0,'AreaPrivativa': 0, 'precoCondominio': 0}
dados_imoveis.fillna(fill, inplace=True)

print(dados_imoveis.isnull().sum())
# %%
# padronizando nome das cidades
dados_imoveis['Cidade'] = dados_imoveis['Cidade'].str.title()

dados_imoveis["Cidade"].value_counts() 

# %%

# transformando valor das cidades e bairros em numericos 

label_encoder = LabelEncoder()
dados_imoveis['Cidade'] = label_encoder.fit_transform(dados_imoveis['Cidade'])

label_encoder = LabelEncoder()
dados_imoveis['Bairro'] = label_encoder.fit_transform(dados_imoveis['Bairro'])


#%%

# arrumando as colunas com dormitorio 0 e contendo suite
# pois se o imovel tem dorm = 0 e suite = 2, significa que ele possui 2 dorm sendo eles suites
def preencher_dorm(row):
    if row['QtdDormitorio'] == 0 and row['QtdSuite'] > 0:
        return row['QtdSuite']
    else:
        return row['QtdDormitorio']

dados_imoveis['QtdDormitorio'] = dados_imoveis.apply(preencher_dorm, axis=1)

# %%
# verifica e deleta colunas duplicadas
duplicados = dados_imoveis.duplicated().sum()
if duplicados > 0:
    dados_imoveis.drop_duplicates(inplace=True)

# verifica e deleta colunas com cidade nula
cidadeVazia = dados_imoveis['Cidade'].isnull().sum()
if cidadeVazia > 0:
    dados_imoveis = dados_imoveis.drop(dados_imoveis[dados_imoveis['Cidade'].isnull()].index, axis=0)

# verifica e deleta colunas com bairro nulo
bairroVazio = dados_imoveis['Bairro'].isnull().sum()
if bairroVazio > 0:
    dados_imoveis = dados_imoveis.drop(dados_imoveis[dados_imoveis['Bairro'].isnull()].index, axis=0)


print(dados_imoveis.isnull().sum())

# %%

# dados faltantes por coluna
print('dados faltantes por coluna:')
print(dados_imoveis.isna().sum().sort_values(ascending=False))

# porcentagem de dados faltantes por coluna
print('porcentagem de dados faltantes por coluna:')
print(((dados_imoveis.isna().sum() / len(dados_imoveis)) * 100).sort_values(ascending=False))

# limpa imoveis com preço nan
dados_imoveis = dados_imoveis.drop(dados_imoveis[dados_imoveis['preco'].isnull()].index)

# limpa imoveis com preço menor que 10.000
dados_imoveis = dados_imoveis.loc[dados_imoveis['preco'] >= 10000]

# %%
# dados faltantes por coluna depois da limpeza
print('dados faltantes por coluna depois da limpeza:')
print(dados_imoveis.isna().sum().sort_values(ascending=False))

# porcentagem de dados faltantes por coluna depois da limpeza
print('porcentagem de dados faltantes por coluna depois da limpeza:')
print(((dados_imoveis.isna().sum() / len(dados_imoveis)) * 100).sort_values(ascending=False))


# %%
# verifica se ainda existe preco zerado
contagem_zeros = (dados_imoveis['preco'] == 0).sum()
print(contagem_zeros)


print(dados_imoveis.isnull().sum())

dados_imoveis.describe()

# %%

# verifica os outlier dos preços
colunas = ['preco']
plt.boxplot(dados_imoveis[colunas])
plt.xticks([1], colunas)
plt.title('Outlier antes de remover')
plt.show()
print(f'Total de colunas com outlier: {dados_imoveis.shape[0]}')

# %%


Q1 = dados_imoveis[colunas].quantile(0.25)
Q3 = dados_imoveis[colunas].quantile(0.75)
IQR = Q3 - Q1

# elimina outlier do preço
dados_imoveis = dados_imoveis[~((dados_imoveis[colunas] < (Q1 - 1.5 * IQR)) | (dados_imoveis[colunas] > (Q3 + 1.5 * IQR))).any(axis=1)]

colunas = ['preco']
plt.boxplot(dados_imoveis[colunas])
plt.xticks([1], colunas)
plt.title('Outlier depois de remover')
plt.show()
print(f'Total de colunas com outlier: {dados_imoveis.shape[0]}')
# %%

#classificação do padrão

# define limites
# !!! SUJEITO A MUDANÇAS !!!
limites = {
    'QtdDormitorio': [2,4],
    'QtdBanheiro': [1,2],
    'QtdSuite': [1,2],
    'preco': [250000,500000]
}

def classificar_imovel(row):
    pontos = 0
    pontos += (row['QtdDormitorio'] > limites['QtdDormitorio'][0]) + (row['QtdDormitorio'] > limites['QtdDormitorio'][1])
    pontos += (row['QtdBanheiro'] > limites['QtdBanheiro'][0]) + (row['QtdBanheiro'] > limites['QtdBanheiro'][1])
    pontos += (row['QtdSuite'] > limites['QtdSuite'][0]) + (row['QtdSuite'] > limites['QtdSuite'][1])
    pontos += (row['preco'] > limites['preco'][0]) + (row['preco'] > limites['preco'][1])

    if pontos <= 3:
        return 'Baixo'
    elif pontos <= 5:
        return 'Médio'
    else:
        return 'Alto'
    
dados_imoveis['padrão'] = dados_imoveis.apply(classificar_imovel, axis=1)

#%%

# metodo de classificação 1

# pré-processamento dos dados
X = dados_imoveis.iloc[:, :-1].values
y = dados_imoveis.iloc[:, -1].values

# dividindo os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia: {accuracy}")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


#%%

# metodo de classificação 2

# pré-processamento dos dados
X = dados_imoveis.iloc[:, :-1].values
y = dados_imoveis.iloc[:, -1].values

# dividindo os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# normalizando os recursos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(X_train)
print(X_test)

# construindo o classificador Random Forest
classificador = RandomForestClassifier(n_estimators=100)
classificador.fit(X_train, y_train)

# fazendo previsões no conjunto de teste
y_pred = classificador.predict(X_test)

# avaliando o modelo
cm = confusion_matrix(y_test, y_pred)
print(cm)

# %%
