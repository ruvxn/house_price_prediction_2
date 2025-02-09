import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#import dataset
house_data_original = pd.read_csv('dataset/kc_house_data.csv')

#understand dataste
print(house_data_original.head(10))
print(house_data_original.info())
print(house_data_original.describe())


