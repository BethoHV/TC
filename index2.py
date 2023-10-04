# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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
dados_imoveis = dados_imoveis.drop(["referencia","descricao","areaUtil","numero","endereco","complemento"], axis=1)

# limpando valores NAN para 0 nas colunas:
fill = {'QtdDormitorio': 0,'QtdSuite': 0,'QtdBanheiro': 0,'QtdBoxes': 0,'AreaTotal': 0,'AreaPrivativa': 0, 'precoCondominio': 0}
dados_imoveis.fillna(fill, inplace=True)

print(dados_imoveis.isnull().sum())

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
    elif pontos <= 6:
        return 'Médio'
    else:
        return 'Alto'
    
dados_imoveis['padrão'] = dados_imoveis.apply(classificar_imovel, axis=1)

# %%
