import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

p_class = {'1':'A', '2':'B', '3':'C'}
survived = {'0': True, '1': False}

def change_to_boolean(data):
    if data in survived:
        return survived[data]
    else:
        return np.nan


def change_labels(data):
    if data in p_class:
        return p_class[data]
    else:
        return np.nan

    
titanic = pd.read_csv(
    filepath_or_buffer="./Titanic.csv",
    index_col='PassengerId',
    converters={
        'Survived': change_to_boolean,
        'Pclass' : change_labels,
    },
    usecols=lambda col: col not in ['SibSp', 'Ticket', 'Parch', 'Cabin', 'Embarked', 'Name'],
    dtype={
        'Sex':'category',
    },
    skip_blank_lines = True,
)

titanic.Survived = titanic.Survived.astype('category')
titanic.Pclass = titanic.Pclass.astype('category')
scaler = MinMaxScaler()
scaler.fit(titanic[['Fare']]);
titanic[['Fare']] = scaler.transform(titanic[['Fare']])
titanic.Age = titanic.groupby(['Pclass'])['Age'].transform(lambda x: x.fillna(x.mean()))
titanic.Age = titanic.Age.astype('int64')