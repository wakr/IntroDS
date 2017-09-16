import pandas as pd
import json
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

df = pd.read_csv("week2/passengers.csv")

# 1
categ = ["PassengerId", "Survived", "Pclass", "Sex", "SibSp", "Parch", "Cabin", "Embarked", "Deck"]
numer = ["Age", "Fare"]


df[categ].mode()[:1]

df[numer].mean()

# 2

men = df.query("Sex == 1")
women = df.query("Sex == 0")

def create_avg(d):
    avg_df = d[categ].mode()[:1]
    means = d[numer].mean()
    avg_df['Age'] = means.Age
    avg_df['Fare'] = means.Fare
    return avg_df

non_sur_men = df.query("Survived == 0")
sur_wom = df.query("Survived == 1")

avg_joe = create_avg(df)
avg_non_joe = create_avg(non_sur_men)
avg_jane = create_avg(sur_wom)


# 3 - 4

df.Age.plot(kind="hist", title="Whole age dist")
df.Survived.plot(kind="hist", title="Whole survived dist")
df.Fare.plot(kind="hist", title="Whole fare dist")

avg_non_joe
non_sur_men.Age.plot(kind="hist", title="Non-survived age dist")
non_sur_men.Fare.plot(kind="hist", title="Non-survived fare dist")
non_sur_men.Deck.plot(kind="hist", title="Non-survived Deck dist")

avg_jane
sur_wom.Age.plot(kind="hist", title="Survived age dist")
sur_wom.Fare.plot(kind="hist", title="Survived fare dist")
sur_wom.Deck.plot(kind="hist", title="Survived deck dist")


# other group 

non_sur_men.query("Fare >= 40 and Age >= 28")
sur_wom.query("Fare <= 23 and Age <= 31")



# 5

asd = sur_wom.plot.scatter(x="Age", y="Fare", color="DarkBlue", label="Survivors", alpha=0.1)
non_sur_men.plot.scatter(x="Age", y="Fare", color="Green", label="Non-survivors", alpha=0.1, ax=asd)

df.plot.scatter(x="Age", y="Fare", c="Survived")

# 6

# women = 0, men = 1

ss = sur_wom.Sex.astype("category")
(ss.value_counts() / ss.size).plot.pie()

sn = non_sur_men.Sex.astype("category")
(sn.value_counts() / sn.size).plot.pie()

# 7


# Use of average e.g. in fares will make the avg column stand out alot, but it still
# allows to use the whole data and works quite like the data in that column didnt affect
# others because its the avg instead of e.g. 0 or 9999999 
 
