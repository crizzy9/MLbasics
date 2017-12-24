import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rnd
import pandas as pd
from sklearn.linear_model import LinearRegression
import os

# univariate linear regression

rnd.seed(42)

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "fundamentals"

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)


datapath = '../datasets/lifesat/'

# Load the data
oecd_bli = pd.read_csv(os.path.join(datapath,"oecd_bli_2015.csv"), thousands=',')
gdp_per_capita = pd.read_csv(datapath+"gdp_per_capita.csv", thousands=',', delimiter='\t', encoding='latin1', na_values='n/a')

oecd_bli = oecd_bli[oecd_bli['INEQUALITY'] == 'TOT']
oecd_bli = oecd_bli.pivot(index='Country', columns='Indicator', values='Value')
oecd_bli = oecd_bli[['Life satisfaction']]

print(oecd_bli['Life satisfaction'].head())

gdp_per_capita.set_index('Country', inplace=True)
gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
gdp_per_capita = gdp_per_capita[['GDP per capita']]

print(gdp_per_capita.head())

full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita, left_index=True, right_index=True)

print(full_country_stats.head())


remove_indices = [0, 1, 6, 8, 33, 34, 35]
keep_indices = list(set(range(36)) - set(remove_indices))

train_data = full_country_stats.iloc[keep_indices]
test_data = full_country_stats.iloc[remove_indices]
print(train_data)
print(test_data)


# train_data.plot(kind='scatter', x='GDP per capita', y='Life satisfaction')
# plt.show()

X_train = np.array(train_data['GDP per capita']).reshape(-1,1)
X_test = np.array(test_data['GDP per capita']).reshape(-1,1)
y_train = np.array(train_data['Life satisfaction'])
y_test = np.array(test_data['Life satisfaction'])

clf = LinearRegression(n_jobs=5)
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)

t0, t1 = clf.intercept_, clf.coef_

print(accuracy)
print(clf.predict(np.array([[22587.0]])))


# train_data.plot(kind='scatter', x='GDP per capita', y='Life satisfaction')
# plt.plot(X_train, t0 + t1*X_train, 'b')
# plt.show()

train_data.plot(kind='scatter', x="GDP per capita", y='Life satisfaction', figsize=(5,3))
plt.annotate('BEAUTY', xy=(20000, 2.5), xytext=(18000, 1.7),
            arrowprops=dict(facecolor='black', width=0.5, shrink=0.1, headwidth=5))
plt.plot(20000, 2.5, "ro")
plt.axis([0, 60000, 0, 10])
position_text = {
    "Hungary": (5000, 1),
    "Korea": (18000, 1.7),
    "France": (29000, 2.4),
    "Australia": (40000, 3.0),
    "United States": (52000, 3.8),
}
for country, pos_text in position_text.items():
    if country in train_data.index:
        pos_data_x, pos_data_y = train_data.loc[country]
    else:
        pos_data_x, pos_data_y = test_data.loc[country]

    country = "U.S." if country == "United States" else country
    plt.annotate(country, xy=(pos_data_x, pos_data_y), xytext=pos_text,
            arrowprops=dict(facecolor='black', width=0.5, shrink=0.1, headwidth=5))
    plt.plot(pos_data_x, pos_data_y, "ro")
# save_fig('money_happy_scatterplot')
plt.show()

# reference: hands on ml
