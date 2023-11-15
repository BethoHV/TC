import pandas as pd 
import matplotlib.pyplot as plt

def plotBairro_T(df):
    df_grouped = df.groupby(['Bairro', 'Tipo']).size().reset_index(name='contagem')
    df_grouped['contagem'] = pd.to_numeric(df_grouped['contagem'], errors='coerce')

    # Ordene os resultados e pegue a quantidade de quartos mais frequente por bairro
    df_grouped = df_grouped.sort_values(['contagem'], ascending=[False])
    df_grouped = df_grouped.drop_duplicates(subset='Bairro', keep='first')

    df_top10 = df_grouped.head(10)

    plt.figure(figsize=(10,5))
    plt.barh(df_top10['Bairro'] + ' - ' + df_top10['Tipo'], df_top10['contagem'])
    plt.xlabel('Contagem')
    plt.ylabel('Bairro - Tipo de Imóvel')
    plt.title('Os 10 tipos de imóveis mais comuns por bairro')
    plt.show()


def plotBairro_TD(df):

    df_grouped = df.groupby(['Bairro', 'Tipo','Dormitorios']).size().reset_index(name='contagem')
    df_grouped['contagem'] = pd.to_numeric(df_grouped['contagem'], errors='coerce')

    # Ordene os resultados e pegue a quantidade de quartos mais frequente por bairro
    df_grouped = df_grouped.sort_values(['contagem'], ascending=[False])
    df_grouped = df_grouped.drop_duplicates(subset='Bairro', keep='first')

    df_top10 = df_grouped.head(10)

    plt.figure(figsize=(10,5))
    plt.barh(df_top10['Bairro'] + ' - ' + df_top10['Tipo'] + ' - ' + df_top10['Dormitorios'].astype(str) + ' quartos', df_top10['contagem'])
    plt.xlabel('Contagem')
    plt.ylabel('Bairro - Tipo de Imóvel - Quartos')
    plt.title('Os 10 tipos de imóveis mais comuns por bairro e número de quartos')
    plt.show()

def plot_TDB(df):
     
    df_grouped = df.groupby(['Tipo','Dormitorios','Banheiros']).size().reset_index(name='contagem')
    df_grouped['contagem'] = pd.to_numeric(df_grouped['contagem'], errors='coerce')

    # Ordene os resultados e pegue a quantidade de quartos mais frequente por bairro
    df_grouped = df_grouped.sort_values(['contagem'], ascending=[False])

    df_top10 = df_grouped.head(10)

    plt.figure(figsize=(10,5))
    plt.barh(df_top10['Tipo'] + ' - ' + df_top10['Dormitorios'].astype(str) + ' quartos' + df_top10['Banheiros'].astype(str) + ' banheiros', df_top10['contagem'])
    plt.xlabel('Contagem')
    plt.ylabel('Bairro - Tipo de Imóvel - Quartos')
    plt.title('Os 10 tipos de imóveis mais comuns por bairro e número de quartos')
    plt.show()