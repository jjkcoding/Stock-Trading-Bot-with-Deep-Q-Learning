# Stock Trading Bot with Deep Q-Learning: Project Overview
* Transformed and prepared data for Deep Q-Learning
* Developed an environment, model, and Deep Q-Network to predict Apple Stock Data
* Trained and tested stock bot with yfinance Apple stock data 
* Predicted best moments to buy and sell stocks
* **Coding Language:** Python
* **Main Packages:** pandas, numpy, tensorflow, yfinance

## Preparing Data:
* Collected Date and Time, Open, Close, High, Low, and Volume from yfinance hourly stock data
* Converted and dropped Date and Time into Hour, Month, and Year
* Added a binary "Went Up" variable (1 = current closing price is higher than previous, 0 = current closing price is lower than previous)
* Split data into 75% training data and 25% testing data

## Training Algorithm:
* If the bot has not traded for 5 days, the bot is negatively rewarded
* If the bot is holding a stock, the bot is rewarded positively
* If the bot sold a stock, the bot is rewarded by the equation:
* (Current Closing Price - Buying Closing Price) * (Number of days stock was held) / (Number of Trades Occurred By Bot)
* Reward is scaled by 0.001 before placing into Deep Q-Network model
* Used three output values (0 = Hold current stock or Do not buy, 1 = Buy Stock, 2 = Sell Stock)
* Bot cannot buy stock if it already bot and cannot sell a stock if it did not buy one

## Model:
* Modeling based on an input layer, two Dense layers (units = 64, 32) with sigmoid activation layers, two Dropout layers (rate = 0.1, 0.1) in between and after the Dense layers, and a final Dense layer with three outputs (0 = Hold current stock or Do not buy, 1 = Buy Stock, 2 = Sell Stock)
* Left thirty percent chance of going for a randomly selected output
* One Epoch Example: \
Epoch: 002/010 \
Total Reward: 0.14 \
Profit: 6.691009521484375 \
Loss: 0.31766826641978696 \
End: 1641861533.8103352 \
Time Diff: 287.1830909252167 

## Testing:
* Continued with model from training, but used testing data
* Example Output: \
Total Profit: 11.099609375 \
Total Reward: 0.0009725590234480766

## Conclusion:
* Apple stock prices have an overall positive trend making the stock bot buy and sell stocks consecutively
* Attempted to stop above problem by multiplying the reward for holding stock prices, but failed
* Overall, I learned more about Q-Learning and neural networks while having fun



