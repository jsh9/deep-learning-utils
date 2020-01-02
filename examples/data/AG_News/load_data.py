# -*- coding: utf-8 -*-
"""
Load the raw CSV files into memory.

Created on Wed Jan  1 22:21:44 2020
"""
import pandas as pd

column_names = ['label', 'title', 'text']

df_train = pd.read_csv('train.csv', header=None, names=column_names)
df_test = pd.read_csv('test.csv', header=None, names=column_names)
