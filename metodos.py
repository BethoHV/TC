from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import lime.lime_tabular

def RandonForrest(df):

    # metodo de classificação randon forrest
    model = RandomForestClassifier()

    numatributos = len(df.columns) - 1
    atributos = list(df.columns[0:numatributos])

    X = df[atributos]
    y = df['Padrao']

    # dividindo os dados em conjuntos de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    resultado = model.fit(X_train, y_train)
    predicted = cross_val_predict(model, X_train, y_train, cv=10)
    expected = y_train.values
    print(confusion_matrix(expected, predicted))

    print("Esperado:\n")
    print(expected)
    print("Previsto:\n")
    print(predicted)

    print(classification_report(expected, predicted))
    print(accuracy_score(expected, predicted))

    # Visualizar as features mais importantes
    print(model.feature_importances_)
    for feature,importancia in zip(df.columns,model.feature_importances_):
        print("{}:{}".format(feature, importancia))

    #validação no conjunto de dados que foi definido como validação
    predicted = model.predict(X_test)
    expected = y_test.values

    print(confusion_matrix(expected, predicted))

    print("\nEsperado:\n")
    print(expected)
    print("Previsto:\n")
    print(predicted)

    print(classification_report(expected, predicted))
    print(accuracy_score(expected, predicted))

    expl = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=X_train.columns,class_names=['Medio','Alto','Baixo'])

    

def DecisionTree(df):
    
    classificador_tree = tree.DecisionTreeClassifier(random_state=1, max_depth=10)
    numatributos = len(df.columns) - 1
    atributos = list(df.columns[0:numatributos]) 

    X = df[atributos]
    y = df['Padrao']    

    # processo de aprendizado

    resultado = classificador_tree.fit(X, y)
    predicted = cross_val_predict(classificador_tree, X, y, cv=10)

    #matriz de confusão
    expected = y.values
    print('Matriz de confusão')
    print(confusion_matrix(expected, predicted))

    print("\nEsperado:")
    print(expected)
    print("\n Previsto:")
    print(predicted)

    # for esperado,previsto in zip(expected, predicted):
    #     print("{}->{}".format(esperado, previsto))

    print('\n Métricas')
    print(classification_report(expected, predicted))
    print('\n Acurácia')
    print(accuracy_score(expected, predicted))

    # Visualizar as features mais importantes
    print(classificador_tree.feature_importances_)
    
    for feature,importancia in zip(df.columns,classificador_tree.feature_importances_):
        print("{}:{}".format(feature, importancia))
        

def GradientBoost(df):
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

    numatributos = len(df.columns) - 1
    atributos = list(df.columns[0:numatributos])

    X = df[atributos]
    y = df['Padrao']

    # dividindo os dados em conjuntos de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    resultado = model.fit(X_train, y_train)
    predicted = cross_val_predict(model, X_train, y_train, cv=10)
    expected = y_train.values
    print(confusion_matrix(expected, predicted))

    print("Esperado:\n")
    print(expected)
    print("Previsto:\n")
    print(predicted)

    print(classification_report(expected, predicted))
    print(accuracy_score(expected, predicted))

    # Visualizar as features mais importantes
    print(model.feature_importances_)
    for feature,importancia in zip(df.columns,model.feature_importances_):
        print("{}:{}".format(feature, importancia))

    #validação no conjunto de dados que foi definido como validação
    predicted = model.predict(X_test)
    expected = y_test.values

    print(confusion_matrix(expected, predicted))

    print("\nEsperado:\n")
    print(expected)
    print("Previsto:\n")
    print(predicted)

    print(classification_report(expected, predicted))
    print(accuracy_score(expected, predicted))
