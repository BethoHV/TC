
def classificaP(df):

    media_preco = df['Preco'].mean()

    limiar_alto = 1.5 * media_preco
    limiar_baixo = 0.7 * media_preco

    def classificar_padrao(preco):
        if preco >= limiar_alto:
            return 'Alto'
        elif preco <= limiar_baixo:
            return 'Baixo'
        else:
            return 'Medio'
    
    df['Padrao'] = df['Preco'].apply(classificar_padrao)

    return df


def classificaPDB(df):

    media_preco = df['Preco'].mean()
    media_banheiros = df['Banheiros'].mean()
    media_dorm = df['Dormitorios'].mean()

    limiar_alto_p = 1.5 * media_preco
    limiar_baixo_p = 0.7 * media_preco

    limiar_alto_b = 1.5 * media_banheiros
    limiar_baixo_b = 0.7 * media_banheiros

    limiar_alto_d = 1.5 * media_dorm
    limiar_baixo_d = 0.7 * media_dorm

    def classificar_padrao(preco, banheiros, quartos):
        if preco >= limiar_alto_p and banheiros >= limiar_alto_b and quartos >= limiar_alto_d:
            return 'Alto'
        elif preco <= limiar_baixo_p and banheiros <= limiar_baixo_b and quartos <= limiar_baixo_d:
            return 'Baixo'
        else:
            return 'Medio'
    
    df['Padrao'] = df.apply(lambda row: classificar_padrao(row['Preco'], row['Banheiros'], row['Dormitorios']), axis=1)

    return df

def classifica_antigo(df):
    
    limites = {
    'Dormitorios': [2,4],
    'Banheiros': [1,2],
    'Preco': [180000,500000]
    }

    def classificar_imovel(row):
        pontos = 0
        pontos += (row['Dormitorios'] >= limites['Dormitorios'][0]) + (row['Dormitorios'] >= limites['Dormitorios'][1])
        pontos += (row['Banheiros'] >= limites['Banheiros'][0]) + (row['Banheiros'] >= limites['Banheiros'][1])
        pontos += (row['Preco'] >= limites['Preco'][0]) + (row['Preco'] >= limites['Preco'][1])

        if pontos <= 3:
            return 'Baixo'
        elif pontos <= 6:
            return 'Médio'
        else:
            return 'Alto'
        
    df['Padrao'] = df.apply(classificar_imovel, axis=1)

    return df
