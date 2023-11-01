
def classificaP(df):

    media_preco = df['preco'].mean()

    limiar_alto = 1.5 * media_preco
    limiar_baixo = 0.7 * media_preco

    def classificar_padrao(preco):
        if preco >= limiar_alto:
            return 'Alto'
        elif preco <= limiar_baixo:
            return 'Baixo'
        else:
            return 'Medio'
    
    df['Padrao'] = df['preco'].apply(classificar_padrao)

    return df


def classificaPDB(df):

    media_preco = df['preco'].mean()
    media_banheiros = df['QtdBanheiro'].mean()
    media_dorm = df['QtdDormitorio'].mean()

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
    
    df['Padrao'] = df.apply(lambda row: classificar_padrao(row['preco'], row['QtdBanheiro'], row['QtdDormitorio']), axis=1)

    return df

def classifica_antigo(df):
    
    limites = {
    'QtdDormitorio': [2,4],
    'QtdBanheiro': [1,2],
    'QtdSuite': [1,2],
    'preco': [180000,500000]
    }

    def classificar_imovel(row):
        pontos = 0
        pontos += (row['QtdDormitorio'] >= limites['QtdDormitorio'][0]) + (row['QtdDormitorio'] >= limites['QtdDormitorio'][1])
        pontos += (row['QtdBanheiro'] >= limites['QtdBanheiro'][0]) + (row['QtdBanheiro'] >= limites['QtdBanheiro'][1])
        pontos += (row['QtdSuite'] >= limites['QtdSuite'][0]) + (row['QtdSuite'] >= limites['QtdSuite'][1])
        pontos += (row['preco'] >= limites['preco'][0]) + (row['preco'] >= limites['preco'][1])

        if pontos <= 3:
            return 'Baixo'
        elif pontos <= 6:
            return 'Médio'
        else:
            return 'Alto'
        
    df['Padrao'] = df.apply(classificar_imovel, axis=1)

    return df
