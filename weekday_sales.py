import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

calendar = pd.read_csv('./calendar.csv')
sales_train = pd.read_csv('./sales_train_validation.csv')
sell_prices = pd.read_csv('./sell_prices.csv')

d_ = [col for col in sales_train.columns if "d_" in col]

stv = pd.DataFrame(sales_train[d_].T.sum(axis=1)).rename({0:"sales"}, axis=1)\
                    .merge(calendar.set_index("d"), how="left", left_index=True,
                    right_index=True, validate="1:1")

weekday = stv.groupby(["weekday"])["sales"].sum()
weekday = pd.DataFrame(weekday).reset_index().sort_values(by=['sales'])

sum = sum(weekday["sales"])

plt.subplots(figsize=(10,5))
bar = plt.barh(weekday["weekday"], weekday["sales"]/sum*100, color='mediumpurple')

sales = weekday["sales"].tolist()

i = 0
for c in bar:
    width = c.get_width()
    height = c.get_height()
    x, y = c.get_xy()
    plt.text(x + width / 1.2, y + height * 0.4, f"{sales[i]/sum*100:.2f}%", fontsize=11, color="white", fontweight="bold")
    i += 1

plt.title("Weekday Sales", fontsize=15)
plt.ylabel("Day of the week", fontsize=13)
plt.xlabel("Sales [%]", fontsize=13)
plt.show()

