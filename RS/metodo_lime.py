from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
import lime.lime_tabular



def Lime(X,y):

    # dividindo os dados em conjuntos de treinamento e teste
    treinamento_x, validacao_x, treinamento_y, validacao_y = train_test_split(X, y, test_size = 0.20)

    # Criar um objeto Lime
    classificador_forest = RandomForestClassifier()
    resultado = classificador_forest.fit(treinamento_x, treinamento_y)

    expl = lime.lime_tabular.LimeTabularExplainer(treinamento_x, feature_names=['Tipo', 'Cidade', 'Bairro', 'QtdDormitorio', 'QtdSuite', 'QtdBanheiro','QtdBoxes', 'AreaTotal', 'AreaPrivativa'],class_names=['Baixo', 'Medio', 'Alto'])

    prever = lambda x: classificador_forest.predict_proba(x).astype(float)

    print(validacao_x[0])
    print(prever(validacao_x[0].reshape(1,-1)))
    print(classificador_forest.predict(validacao_x[0].reshape(1,-1)))

    previsto = classificador_forest.predict(validacao_x[5].reshape(1,-1))
    probabilidades = classificador_forest.predict_proba(validacao_x[5].reshape(1,-1))
    print(previsto)
    print(probabilidades)
    
    exp = expl.explain_instance(validacao_x[5], prever, num_features=4)
    exp.show_in_notebook(show_all=True)




