# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# lendo o aquivo csv que foi exportado do banco 
dados_imoveis = pd.read_csv('dados/imoveis_vendidos_cx.csv',sep=';')

# transforma os valores de preco em valores numericos
dados_imoveis['preco'] = pd.to_numeric(dados_imoveis['preco'], errors='coerce')

print(dados_imoveis.isnull().sum())

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

# remove colunas inuteis
dados_imoveis = dados_imoveis.drop(["referencia","descricao","areaUtil","numero",], axis=1)

# limpando valores NAN para 0 nas colunas:
fill = {'QtdDormitorio': 0,'QtdSuite': 0,'QtdBanheiro': 0,'QtdBoxes': 0,'AreaTotal': 0,'AreaPrivativa': 0, 'precoCondominio': 0}
dados_imoveis.fillna(fill, inplace=True)

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

colunas = ['QtdDormitorio','QtdSuite','QtdBanheiro','preco']
plt.boxplot(dados_imoveis[colunas])
plt.xticks([1, 2, 3, 4], colunas)
plt.title('Outlier antes de remover')
plt.show()
print(f'Total de colunas com outlier: {dados_imoveis.shape[0]}')

# %%
