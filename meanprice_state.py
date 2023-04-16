import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

calendar = pd.read_csv('./calendar.csv')
sales_train = pd.read_csv('./sales_train_validation.csv')
sell_prices = pd.read_csv('./sell_prices.csv')


stores = ['CA_', 'TX_', 'WI_']
colors = ['mediumpurple', 'mediumseagreen', 'palevioletred']

fig, axs = plt.subplots(1, 3, figsize=(16,5))

for i, store in enumerate(stores):
    sell_prices_store = sell_prices[sell_prices['store_id'].str.startswith(store)]
    mean_prices = sell_prices_store.drop_duplicates(subset='item_id').merge(
        sales_train[['item_id', 'cat_id']], on='item_id'
    ).groupby('cat_id')['sell_price'].mean()

    axs[i].bar(mean_prices.index, mean_prices.values, color=colors[i])

    for j, val in enumerate(mean_prices.values):
        axs[i].text(j, val-0.5, f'{val:.2f}', ha='center', fontsize=11, color='black', fontweight='bold')

    axs[i].set_title(f'Mean price of the categories for {store} State')

plt.show()

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# import warnings
# warnings.filterwarnings("ignore")

# calendar = pd.read_csv('./calendar.csv')
# sales_train = pd.read_csv('./sales_train_validation.csv')
# sell_prices = pd.read_csv('./sell_prices.csv')


# ca_sell_prices = sell_prices[sell_prices['store_id'].str.startswith('CA_')]
# ca_mean_prices = ca_sell_prices.drop_duplicates(subset='item_id').merge(
#     sales_train[['item_id', 'cat_id']], on='item_id'
# ).groupby('cat_id')['sell_price'].mean()

# tx_sell_prices = sell_prices[sell_prices['store_id'].str.startswith('TX_')]
# tx_mean_prices = tx_sell_prices.drop_duplicates(subset='item_id').merge(
#     sales_train[['item_id', 'cat_id']], on='item_id'
# ).groupby('cat_id')['sell_price'].mean()

# wi_sell_prices = sell_prices[sell_prices['store_id'].str.startswith('WI_')]
# wi_mean_prices = wi_sell_prices.drop_duplicates(subset='item_id').merge(
#     sales_train[['item_id', 'cat_id']], on='item_id'
# ).groupby('cat_id')['sell_price'].mean()


# fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
# bar1 = ax1.bar(ca_mean_prices.index, ca_mean_prices.values, color='mediumpurple')
# bar2 = ax2.bar(tx_mean_prices.index, tx_mean_prices.values, color='mediumseagreen')
# bar3 = ax3.bar(wi_mean_prices.index, wi_mean_prices.values, color='palevioletred')
# i = 0
# for c in bar1:
#     width = c.get_width()
#     height = c.get_height()
#     x, y = c.get_xy()
#     ax1.text(x + width * 0.4, y + height / 1.2, f"{ca_mean_prices.values[i]:.2f}", fontsize=11, color="white", fontweight="bold")
#     i += 1
# i = 0
# for c in bar2:
#     width = c.get_width()
#     height = c.get_height()
#     x, y = c.get_xy()
#     ax2.text(x + width * 0.4, y + height / 1.2, f"{tx_mean_prices.values[i]:.2f}", fontsize=11, color="white", fontweight="bold")
#     i += 1
# i = 0
# for c in bar3:
#     width = c.get_width()
#     height = c.get_height()
#     x, y = c.get_xy()
#     ax3.text(x + width * 0.4, y + height / 1.2, f"{wi_mean_prices.values[i]:.2f}", fontsize=11, color="white", fontweight="bold")
#     i += 1

# ax1.set_title("Mean price of the categories for CA State")
# ax2.set_title("Mean price of the categories for TX State")
# ax3.set_title("Mean price of the categories for WI State")


# plt.show()
