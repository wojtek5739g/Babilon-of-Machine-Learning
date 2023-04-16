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

month = stv.groupby(["month"])["sales"].sum()
month = pd.DataFrame(month).reset_index().sort_values(by=['sales'])

sum = sum(month["sales"])

plt.subplots(figsize=(10,5))
months = ["Jan.", "Feb.",	"Mar.",	"Apr.",	"May",	"June",	"July",	"Aug.",	"Sept.", "Oct.",	"Nov.",	"Dec."]
bar = plt.barh(month["month"], month["sales"]/sum*100, color='mediumpurple')

sales = month["sales"].tolist()

i = 0
for c in bar:
    width = c.get_width()
    height = c.get_height()
    x, y = c.get_xy()
    plt.text(x + width / 1.2, y + height * 0.4, f"{sales[i]/sum*100:.2f}%", fontsize=11, color="white", fontweight="bold")
    i += 1
plt.yticks(month.index + 1, months, fontsize=11)
plt.title("Month Sales", fontsize=15)
plt.ylabel("Month", fontsize=13)
plt.xlabel("Sales [%]", fontsize=13)
plt.show()

