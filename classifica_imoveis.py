
def classificaP(df):

    media_preco = df['preco'].mean()

    limiar_alto = 1.5 * media_preco
    limiar_baixo = 0.7 * media_preco

    def classificar_padrao(preco):
        if preco >= limiar_alto:
            return 'Alto'
        elif preco <= limiar_baixo:
            return 'Baixo '
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
            return 'Médio'
    
    df['Padrao'] = df.apply(lambda row: classificar_padrao(row['preco'], row['QtdBanheiro'], row['QtdDormitorio']), axis=1)

    return df