# !pip install yfinance

import pandas as pd
import numpy as np
import yfinance as yf
import time

import environment
import dqn
import brain


# SETTING UP CLASSES




# TRAINING MODEL


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



# Setting the paramaters
epsilon = 0.3
num_actions = 3
num_epochs = 10
max_memory = 150
batch_size = 16
learning_rate = 0.001
discount = 0.85
num_inputs = len(stock_train.columns)


# Build Environment
env = environment.Environment(stock_train)
brain = brain.Brain(lr = learning_rate, num_actions = num_actions, num_inputs = num_inputs)
dqn = dqn.DQN(max_memory = max_memory, discount = discount)
train = True

# 0 = do nothing, 1 = buy, 2 = sell

env.train = train
if env.train:
  
  end = time.time()
  for epoch in range(1, num_epochs):
    start = time.time()
    print("Start: " + str(start))
    total_reward = 0.0
    loss = 0.0
    env.reset()
    game_over = False
    curr_state, _, _ = env.observe()
    i = 0 
    profit = 0
    buy_dates = []
    sell_dates = []
    while((not game_over) and i < len(env.dat) - 1):
      env.iter = i
      if np.random.rand() <= epsilon:
        action = np.random.randint(0, num_actions - 1)
        if env.buy and action == 1:
          action = 2
        elif env.buy == False and action == 1:
          action = 1
      else:
        q_vals = brain.model.predict(curr_state)
        action = np.argmax(q_vals[0])
        # Cannot buy if we already bought and cannot sell if we did not buy anything
        if action == 1 and env.buy == True:
          action = 2
        elif action == 2 and env.buy == False:
          action = 1
      next_state, reward, game_over = env.update(action)
      total_reward += reward
      if action == 1:
          buy_dates.append(i)
      elif action == 2:
          sell_dates.append(i)
      dqn.remember([curr_state, action, reward, next_state], game_over)

      inputs, targets = dqn.get_batch(brain.model, batch_size = batch_size)

      loss += brain.model.train_on_batch(inputs, targets)
      i += 1
      curr_state = next_state
    end = time.time()
    print("\n")
    print("Epoch: {:03d}/{:03d}".format(epoch, num_epochs))
    print("Total Reward: {:.2f}".format(total_reward))
    print("Profit: " + str(env.profit))
    print("Loss: " + str(loss))
    print("End: " + str(end))
    print("Time Diff: " + str(end - start))
    print(buy_dates)
    print(sell_dates)
    print("\n")
    brain.model.save("model.h5")
    
