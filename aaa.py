#%%
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix


import pandas as pd
# Carregamento dos dados
iris_data = load_iris()

print(iris_data.feature_names)
# Atributos
X = iris_data.data
# Classe que indica o tipo da planta
y = iris_data.target
# %%

from sklearn.model_selection import train_test_split
treinamento_x, validacao_x, treinamento_y, validacao_y = train_test_split(X, y, test_size = 0.20)
print(treinamento_x)

# %%
