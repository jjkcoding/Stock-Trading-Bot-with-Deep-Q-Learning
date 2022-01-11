import numpy as np

class Environment(object):
  def __init__(self, dat):
    self.dat = dat
    self.max_loss = -50
    self.dep = 2
    self.game_over = 0
    self.reward = 0.0
    self.iter = 0
    self.buy = False
    self.buy_price = 0.0
    self.train = 1
    self.hold = 0
    self.profit = 0.0
    self.scaler = 0.001
    self.wait_days = 5
    self.trades = 1

  def update(self, action):
    if action == 1:
      self.buy = True
      self.buy_price = self.dat.Close.values[self.iter]
      self.reward = 0.0
      self.hold = 1
    elif action == 2:
      self.buy = False
      self.reward = (self.dat.Close.values[self.iter] - self.buy_price) * self.hold / self.trades
      self.profit += self.dat.Close.values[self.iter] - self.buy_price
      self.buy_price = 0.0
      self.hold = 0
      self.trades += 1
    else:
      self.hold += 1
      if self.buy:
        self.reward = self.dep
      else:
          if self.hold > self.wait_days:
              # make bot buy more than usual
              self.reward = -1 * self.dep
          else:
              self.reward = 0
    self.reward *= self.scaler
    next_state = np.matrix(self.dat.values[self.iter + 1])
    return next_state, self.reward, self.game_over

# make it impossible to buy wen selling or selling wen we dont have one
# 0 = do nothing, 1 = buy, 2 = sell

  def reset(self):
    self.game_over = 0
    self.reward = 0.0
    self.iter = 0
    self.buy = False
    self.buy_price = 0.0
    self.train = 1
    self.profit = 0.0
    self.trades = 1


  def observe(self):
    curr_state = np.matrix(self.dat.iloc[self.iter].values)
    return curr_state, self.reward, self.game_over

