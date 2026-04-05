### Summary of the Paper

* The study predicts the next day stock closing price for five companies across different sectors using Artificial Neural Network (ANN) and Random Forest (RF) techniques.
* The companies analyzed are Nike, Goldman Sachs, Johnson and Johnson, Pfizer, and JP Morgan Chase and Co.
* Historical stock data from 04/05/2009 to 04/05/2019 was collected from Yahoo Finance.
* The models were evaluated using Root Mean Square Error (RMSE), Mean Absolute Percentage Error (MAPE), and Mean Bias Error (MBE).
* The comparative analysis demonstrates that the ANN model provided better prediction results than the RF model for four of the five companies.

### What the Paper Did to Create the Model

* The researchers collected historical daily data encompassing High, Low, Open, Close, Adjacent close, and Volume metrics.
* They extracted the day-wise closing prices to use as the base for their analysis.
* The dataset was divided chronologically into a training set (04/06/2009 to 04/03/2017) and a testing set (04/04/2017 to 04/05/2019).
* Six new predictive variables were engineered from the raw data: Stock High minus Low price (H-L), Stock Close minus Open price (O-C), 7 days moving average, 14 days moving average, 21 days moving average, and 7 days standard deviation.
* For the ANN model, they constructed a three-layer architecture consisting of an input layer, a hidden layer, and an output layer.
* The six engineered variables, along with Volume, were fed into the input layer of the ANN.
* The output layer of the ANN consisted of a single neuron designed to output the predicted closing stock price.
* For the RF model, the engineered variables were used to train multiple decision trees.
* The RF model combined these multiple decision trees to determine the final output regression, which reduces the variance in the model.

### Plan to Re-create the Paper from Code-Level

1. **Data Collection:** Utilize the `yfinance` library in Python to programmatically download daily historical stock data for the tickers NKE, GS, JNJ, PFE, and JPM covering the period from 2009-04-05 to 2019-04-05.
2. **Feature Engineering:** Use `pandas` to calculate the six derived features outlined in the paper: `High - Low`, `Close - Open`, `Rolling Mean (7-day)`, `Rolling Mean (14-day)`, `Rolling Mean (21-day)`, and `Rolling Std Dev (7-day)`. Drop any `NaN` rows resulting from the rolling window calculations.
3. **Data Splitting:** Filter the `pandas` DataFrame by date to explicitly separate the data into a training set (2009-04-06 to 2017-04-03) and a testing set (2017-04-04 to 2019-04-05).
4. **ANN Implementation:** Use `TensorFlow/Keras` or `scikit-learn` (`MLPRegressor`) to instantiate a feed-forward neural network. Define an input layer matching the number of features (the 6 new variables + Volume), one hidden layer, and a single-node output layer with a linear activation function. Train using the Mean Squared Error loss function.
5. **RF Implementation:** Instantiate a `RandomForestRegressor` from `scikit-learn`. Feed the exact same training features into the model to fit the ensemble of decision trees.
6. **Evaluation:** Generate predictions on the testing dataset for both models. Write custom Python functions or use `scikit-learn.metrics` to calculate the RMSE, MAPE, and MBE for each company and model combination.

### Key Findings and Results
1. **Companies Evaluated:** The models were tested on five major companies: JP Morgan, Nike, Johnson and Johnson, Goldman Sachs, and Pfizer.
2. **Performance Evaluation Metrics**: The researchers measured the prediction errors using Root Mean Square Error (RMSE), Mean Absolute Percentage Error (MAPE), and Mean Bias Error (MBE).
3. **Overall Performance:** The comparative analysis clearly showed that the Artificial Neural Network (ANN) provided better overall stock price predictions compared to the Random Forest (RF) model.
4. **Specific Improvements:** For Nike, JP Morgan and Co., Johnson & Johnson, and Pfizer Inc., the ANN model was the superior technique, yielding lower RMSE and MAPE values.
5. **Best Metrics Achieved:** The most accurate performance obtained by the ANN model resulted in an RMSE of 0.42, a MAPE of 0.77, and an MBE of 0.013.

### Ways to Improve the Code for Better Accuracy

1. **Implement Time-Series Cross-Validation:** Instead of using a single, static train/test split, utilize `TimeSeriesSplit` from `scikit-learn`. This applies expanding or rolling windows to validate the model more robustly against the sequential nature of financial data.
2. **Hyperparameter Optimization:** Rather than using default model architectures, apply grid search (`GridSearchCV`) or Bayesian optimization (`Optuna`) to systematically tune the ANN (learning rate, number of hidden nodes, batch size) and the RF (number of trees, maximum depth, minimum samples per leaf).
3. **Integrate Sequential Deep Learning Models:** Financial data is highly temporally dependent. Replace the standard feed-forward ANN with architectures explicitly designed to retain memory over time, such as Long Short-Term Memory (LSTM) networks or Gated Recurrent Units (GRUs).
4. **Incorporate Qualitative Sentiment Features:** Enhance the dataset by integrating Natural Language Processing (NLP) models like FinBERT. You can scrape and score financial news headlines or social media sentiment regarding the specific companies to capture market psychology, which pure technical indicators miss.
5. **Apply Feature Scaling:** Neural networks are highly sensitive to the scale of input data. Implement `MinMaxScaler` or `StandardScaler` on your feature set before training the ANN to ensure all variables contribute equally to the weight updates and to speed up gradient descent convergence.