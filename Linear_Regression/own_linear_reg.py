from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')


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


xs = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
ys = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)

clf = OwnLinearRegression(xs, ys)

regline = clf.reg_line()
print(regline)

cod = clf.coefficient_of_determination()
print(cod)

predict_y = clf.predict(8)
print(predict_y)

plt.scatter(xs, ys)
plt.plot(xs, regline)
plt.show()
