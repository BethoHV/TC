# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


dados_imoveis = pd.read_csv('dados/imoveis_vendidos_cx.csv',sep=';')

print(dados_imoveis.head)

# %%

# dados faltantes por coluna
print('dados faltantes por coluna:')
print(dados_imoveis.isna().sum().sort_values(ascending=False))

# porcentagem de dados faltantes por coluna
print('porcentagem de dados faltantes por coluna:')
print(((dados_imoveis.isna().sum() / len(dados_imoveis)) * 100).sort_values(ascending=False))

# remove colunas inuteis
dados_imoveis = dados_imoveis.drop(["referencia","descricao","areaUtil"], axis=1)

# limpando valores None para 0 nas colunas:
fill = {'QtdSuite': 0,'QtdBanheiro': 0,'QtdBoxes': 0,'AreaTotal': 0,'AreaPrivativa': 0}
dados_imoveis.fillna(fill, inplace=True)

# limpa imoveis com preço menor que 10.000
dados_imoveis = dados_imoveis.drop(dados_imoveis[dados_imoveis["preco"].map(dados_imoveis["preco"]) < 10000]["preco"].index)


# %%
# dados faltantes por coluna depois da limpeza
print('dados faltantes por coluna depois da limpeza:')
print(dados_imoveis.isna().sum().sort_values(ascending=False))

# porcentagem de dados faltantes por coluna depois da limpeza
print('porcentagem de dados faltantes por coluna depois da limpeza:')
print(((dados_imoveis.isna().sum() / len(dados_imoveis)) * 100).sort_values(ascending=False))

print(dados_imoveis['preco'].sort_values())
