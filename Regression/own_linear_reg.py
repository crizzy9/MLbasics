from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('ggplot')


class OwnLinearRegression:

    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys
        self.m = None
        self.b = None
        self.ys_line = None
        self.best_fit_slope_and_intercept()

    # calculate b and m
    def best_fit_slope_and_intercept(self):
        self.m = ((mean(self.xs) * mean(self.ys)) - mean(self.xs * self.ys)) / (mean(self.xs) ** 2 - mean(self.xs ** 2))
        self.b = mean(self.ys) - self.m * mean(self.xs)

    # calculate cost
    # could make it private method
    def root_mean_squared_error(self, ys_new):
        return sum((ys_new - self.ys) ** 2) / len(self.ys)**(1/2)

    # the accuracy of regression line
    def coefficient_of_determination(self):
        mn = mean(self.ys)
        y_mean_line = np.array([mn]*len(self.ys))
        rmse_regr = self.root_mean_squared_error(self.ys_line)
        rmse_mean = self.root_mean_squared_error(y_mean_line)
        return 1 - (rmse_regr / rmse_mean)

    # hypothesis of single point according to regression line
    def predict(self, x):
        return (self.m * x) + self.b

    # calculate regression line
    def reg_line(self):
        self.ys_line = [self.predict(x) for x in self.xs]
        return [self.predict(x) for x in self.xs]

    # this only works on a small dataset and tends to be linear for larger datasets
    # if correlation is false the data wont increment linearly since the numbers will only be in the range of variance
    # if variance increases the the data is spaced among a larger range hence the data is more scattered
    # if steps increase the variance matters less because on a larger plot the variation will still be scattered
    # across a small range and the rate at which steps increase y is less than that of variance so
    # its scattered data across a range that increments linearly
    @staticmethod
    def create_dataset(hm, variance, step=2, correlation=None):
        val = 1
        ys = []
        for _ in range(hm):
            y = val + random.randrange(-variance, variance)
            ys.append(y)
            if correlation == 'pos':
                val += step
            elif correlation == 'neg':
                val -= step
        xs = [i for i in range(len(ys))]

        return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


# xs = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
# ys = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)


xs, ys = OwnLinearRegression.create_dataset(400, 8000, 10, correlation='pos')

print(xs)
print(ys)

clf = OwnLinearRegression(xs, ys)

regline = clf.reg_line()
print(regline)

cod = clf.coefficient_of_determination()
print(cod)

predict_x = 8
predict_y = clf.predict(predict_x)
print(predict_y)

plt.scatter(xs, ys)
plt.scatter(predict_x, predict_y, color='g')
plt.plot(xs, regline, 'b')
plt.show()


# reference: sentdex
