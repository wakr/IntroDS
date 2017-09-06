import pandas as pd
import json

df = pd.read_csv("train.csv")
#print(data.to_json(orient='records', lines=True))

# 2
df.drop('Name', axis=1, inplace=True)
df.drop('Ticket', axis=1, inplace=True)

# 3
df['Cabin'] = df['Cabin'].fillna(df.mode()['Cabin'].iloc[0])

df['Deck'] = df['Cabin'].map(lambda x: str(x)[0]) # first letter as a deck


# 4
#print(df)

df['Deck'] = df['Deck'].astype('category').cat.codes
df['Cabin'] = df['Cabin'].astype('category').cat.codes
df['Sex'] = df['Sex'].astype('category').cat.codes
df['Survived'] = df['Survived'].astype('category').cat.codes
df['Embarked'] = df['Embarked'].astype('category').cat.codes

#print(df)

# 5

df['Age'] = df['Age'].fillna(df.mean()['Age'])
df['Fare'] = df['Fare'].fillna(df.mean()['Fare'])

discrete_names = ['PassengerId', 'Survived', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Cabin', 'Embarked', 'Deck']
for x in discrete_names:
    df[x].fillna(df.mode()[x].iloc[0], inplace=True)

print(df)

# 6

df.to_csv('passengers.csv', index=False)
df.to_json('passengers.json', orient='records')