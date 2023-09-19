# %%
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# conexão e consulta da tabela no banco de dados:
db_uri = "mysql+mysqlconnector://root:@localhost/imoveis"
engine = create_engine(db_uri)
sql = "SELECT * FROM imoveis_vendidos"
dados_imoveis = pd.read_sql_query(sql,engine)

print(dados_imoveis)

# %%

# dados faltantes por coluna
print('dados faltantes por coluna:')
print(dados_imoveis.isna().sum().sort_values(ascending=False))

# porcentagem de dados faltantes por coluna
print('porcentagem de dados faltantes por coluna:')
print(((dados_imoveis.isna().sum() / len(dados_imoveis)) * 100).sort_values(ascending=False))

# remove colunas inuteis
dados_imoveis = dados_imoveis.drop(["referencia","descricao"], axis=1)

# limpando valores None para 0 nas colunas:
fill = {'QtdSuite': 0,'QtdBanheiro': 0,'QtdBoxes': 0,'AreaTotal': 0,'AreaPrivativa': 0,'areaUtil': 0}
dados_imoveis.fillna(fill, inplace=True)

# %%
# dados faltantes por coluna depois da limpeza
print('dados faltantes por coluna depois da limpeza:')
print(dados_imoveis.isna().sum().sort_values(ascending=False))

# porcentagem de dados faltantes por coluna depois da limpeza
print('porcentagem de dados faltantes por coluna depois da limpeza:')
print(((dados_imoveis.isna().sum() / len(dados_imoveis)) * 100).sort_values(ascending=False))

# %%
plt.figure(figsize=(10,6))
sns.boxplot(data=dados_imoveis, orient="h");


engine.dispose()