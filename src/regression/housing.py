import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import os
from sklearn.preprocessing import Imputer, LabelBinarizer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


style.use('ggplot')

plt.rcParams["patch.force_edgecolor"] = True

housing_path = '../../datasets/housing'

housing = pd.read_csv(os.path.join(housing_path, 'housing.csv'))

# print(housing.head())

ocp = housing['ocean_proximity'].value_counts()

# adding extra attributes for normalization and get more effectiveness

# for stratified split
housing['income_cat'] = np.ceil(housing['median_income']/1.5)
housing['income_cat'].where(housing['income_cat'] < 5, 5.0, inplace=True)


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # here X will be a numpy array otherwise X.iloc should've been used
        rooms_per_household = X[:, self.rooms_ix]/X[:, self.household_ix]
        population_per_household = X[:, self.population_ix]/X[:, self.household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, self.bedrooms_ix]/X[:, self.rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


# adding more useful attributes with higher correlation
# housing['rooms_per_household'] = housing['total_rooms']/housing['households']
# housing['bedrooms_per_room'] = housing['total_bedrooms']/housing['total_rooms']
# housing['population_per_household'] = housing['population']/housing['households']

# adding transformer to the pipeline
# creating extra attributes using a transformer
# attr_adder = CombinedAttributesAdder()
# housing = attr_adder.transform(housing)
# print(housing.head())



# for normal split but income category will require stratified shuffle split
# train_data, test_data = train_test_split(housing, test_size=0.2, random_state=42)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

for set in (strat_train_set, strat_test_set):
    set.drop(["income_cat"], axis=1, inplace=True)

train_set = strat_train_set.copy()

corr_matrix = train_set.corr()

print(corr_matrix['median_house_value'].sort_values(ascending=False))


# density based plot
# train_set.plot(kind='scatter', x='longitude', y='latitude', alpha=0.1)

# density color and size based plot
# train_set.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4, s=train_set['population']/100, label='population',
#                c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True)
# plt.legend()

# ocean_proximity pie chart
# plt.pie(ocp.values, labels=ocp.index, autopct='%1.3f%%')
# housing.hist(bins=50, figsize=(20,15))
# plt.show()

# scatter matrix
# scatter_matrix(train_set[['median_house_value', 'median_income', 'bedrooms_per_room', 'rooms_per_household']], figsize=(12, 8))
# plt.show()

new_housing = strat_train_set.drop('median_house_value', axis=1)
new_housing_labels = strat_train_set.copy()
housing_labels = strat_train_set["median_house_value"].copy()
test_housing_labels = strat_test_set["median_house_value"].copy()

# can be done using imputer
# median = new_housing.total_bedrooms.median()
# new_housing.total_bedrooms.fillna(median)

# using Imputer
imputer = Imputer(strategy="median")
housing_num = new_housing.drop('ocean_proximity', axis=1)
imputer.fit(housing_num)
print("Housing num \n{}".format(housing_num.head()))

X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns)
print(housing_tr.head())

# LabelEncoder
encoder = LabelEncoder()
housing_cat = new_housing['ocean_proximity']
housing_cat_encoded = encoder.fit_transform(housing_cat)
# print(housing_cat_encoded)
# print(encoder.classes_)

# One Hot Encoder
encoder1hot = OneHotEncoder()
housing_cat_1hot = encoder1hot.fit_transform(housing_cat_encoded.reshape(-1, 1))
# print(housing_cat_1hot)


# Label Binarizer
encoder_lb = LabelBinarizer(sparse_output=True)
housing_cat_lb = encoder_lb.fit_transform(housing_cat)
print(housing_cat_lb)


# num_pipeline = Pipeline([
#     ('imputer', Imputer(strategy='median')),
#     ('attribs_adder', CombinedAttributesAdder()),
#     ('std_scalar', StandardScaler())
# ])
#
# housing_num_tr = num_pipeline.fit_transform(housing_num)


# Extending the pipeline to do all the tasks at once with custom class DataFrameSelector and CombinedAttribsAdder
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.attribute_names].values


class CustomLabelBinarizer(BaseEstimator, TransformerMixin):
    def __init__(self, sparse_output=False):
        self.sparse_output = sparse_output
        self.encoder = LabelBinarizer(sparse_output=self.sparse_output)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return self.encoder.fit_transform(X)


num_attribs = list(housing_num)
cat_attribs = ['ocean_proximity']

num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('imputer', Imputer(strategy='median')),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scalar', StandardScaler())
])

# Label Binarizer is not meant to be used like this hence Custom Label Binarizer is used
# to remove Label Binarizer requires 2 positional arguments but 3 were given
cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('label_binarizer', CustomLabelBinarizer())
])

full_pipeline = FeatureUnion(transformer_list=[
    ('num_pipeline', num_pipeline),
    ('cat_pipeline', cat_pipeline)
])


housing_prepared = full_pipeline.fit_transform(new_housing)
print(housing_prepared)
print(housing_prepared.shape)

# recreating dataframe

categories = num_attribs + ['rooms_per_household', 'population_per_household', 'bedrooms_per_room'] + list(cat_pipeline.steps[1][1].encoder.classes_)
housing_final = pd.DataFrame(data=housing_prepared, columns=categories)
print(housing_final.head())

# preparing test data
test_housing_prepared = full_pipeline.fit_transform(strat_test_set)
test_housing_final = pd.DataFrame(data=test_housing_prepared, columns=categories)

# removing unwanted columns
# no need to remove right now but can put a selector and DataFrameCreator in pipeline
# imp_columns = list(filter(lambda x: x not in ['total_rooms', 'total_bedrooms'], categories))


# Applying regression
lin_reg = LinearRegression()
lin_reg.fit(housing_final, housing_labels)

print("Lin Reg")
print("Predictions:\t", lin_reg.predict(test_housing_final.tail()))
print("Labels:\t\t", list(test_housing_labels.tail()))

accuracy_lin_reg = lin_reg.score(test_housing_final, test_housing_labels)
print(accuracy_lin_reg)

# Applying Decision Tree Regressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_final, housing_labels)

print("Tree Reg")
print("Predictions:\t", tree_reg.predict(test_housing_final.tail()))
print("Labels:\t\t", list(test_housing_labels.tail()))

accuracy_tree_reg = tree_reg.score(test_housing_final, test_housing_labels)
print(accuracy_tree_reg)

# Applying Random Forest Regressor
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_final, housing_labels)

print("Forest Reg")
print("Predictions:\t", forest_reg.predict(test_housing_final.tail()))
print("Labels:\t\t", list(test_housing_labels.tail()))

accuracy_forest_reg = forest_reg.score(test_housing_final, test_housing_labels)
print(accuracy_forest_reg)


# reference: hands on ml
