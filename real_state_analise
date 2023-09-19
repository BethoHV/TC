from sqlalchemy import create_engine
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import os 

for dirname, _, filenames in os.walk('dados/'):
     for filename in filenames: 
        print(os.path.join(dirname, filename))

real_estate = pd.read_csv('dados/realtor-data.zip.csv')

missing_values = real_estate.isna().sum().sort_values(ascending=False) 

missing_values_percentages = missing_values/ len(real_estate)

print(real_estate['price'].plot(kind='box', vert=False, figsize=(20,7)))
