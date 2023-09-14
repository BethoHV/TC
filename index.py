from sqlalchemy import create_engine
import pandas as pd

db_uri = "mysql+mysqlconnector://root:@localhost/imoveis"
engine = create_engine(db_uri)

sql = "SELECT * FROM imoveis_vendidos"

df = pd.read_sql_query(sql,engine)
print(df)

engine.dispose()