# TESTING MODEL


import environment

import tensorflow as tf
import pandas as pd
import yfinance as yf
import numpy as np


pre_stock = yf.Ticker("voo").history(period='60d', interval='1h')
pre_stock.reset_index(inplace=True)
pre_stock = pre_stock.drop(['Dividends', 'Stock Splits'], axis = 1)

pre_stock['Hour'] = pd.DatetimeIndex(pre_stock['index'].values).hour
pre_stock['Month'] = pd.DatetimeIndex(pre_stock['index'].values).month
pre_stock['Year'] = pd.DatetimeIndex(pre_stock['index'].values).year
pre_stock = pre_stock.drop(['index'], axis = 1)

stock = pd.get_dummies(pre_stock, columns = ['Year', 'Month', 'Hour'])

close_stocks = stock.Close.values
went_up = [0]
for i in range(1, len(close_stocks)):
  up = 0
  if close_stocks[i] > close_stocks[i-1]:
    up = 1
  went_up.append(up)
stock['Went Up'] = went_up



train_split = int(len(stock) * 0.75)
stock_train = stock[:train_split]
stock_test = stock[train_split:]


num_actions = 3
iterations = 100


env = environment.Environment(stock_test)
profit_iter = []
model = tf.keras.models.load_model("model.h5")


env.reset()
env.dat = stock_test
env.train = False

total_reward = 0
curr_state, _, _ = env.observe()
profit = 0
buy_days = []
buy_costs = []
sell_days = []
sell_costs = []


for i in range(0, len(env.dat) - 1):
  env.iter = i
  q_vals = model.predict(curr_state)
  action = np.argmax(q_vals[0])
  # Cannot buy if we already bought and cannot sell if we did not buy anything
  if action == 1 and env.buy == True:
      action = 2
  elif action == 2 and env.buy == False:
      action = 1
  next_state, reward, game_over = env.update(action)
  if action == 1:
      buy_days.append(i)
      buy_costs.append(env.dat.Close.values[i])
  elif action == 2:
      sell_days.append(i)
      sell_costs.append(env.dat.Close.values[i])
  total_reward += reward
  curr_state = next_state

print("\n")
print("Total Profit: " + str(env.profit))
print("Total Reward: " + str(total_reward))
print(buy_days)
print(sell_days)
print(buy_costs)
print(sell_costs)
print("\n")

