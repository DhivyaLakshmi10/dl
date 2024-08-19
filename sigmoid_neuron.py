import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

class Sigmoid:
  def __init__(self, data, target, eta, test_size=0.2, random_state=42):
        self.eta = eta
        self.target = target

        # Preprocessing the data
        self.data = self.preprocess_data(data)

        # Splitting the data
        train_data, test_data = train_test_split(self.data, test_size=test_size, random_state=random_state)
        self.x_train = train_data.drop(columns=[target])
        self.y_train = train_data[target]
        self.x_test = test_data.drop(columns=[target])
        self.y_test = test_data[target]

        self.n = self.x_train.shape[0]
        self.d = self.x_train.shape[1]

        # Adding bias term
        self.x_train.insert(0, 'x0', 1)
        self.x_test.insert(0, 'x0', 1)
        # Initialize weights
        self.w = np.random.rand(1, self.d + 1)[0]

  def preprocess_data(self, data):
      if self.target not in data.columns:
          raise KeyError(f"Target column '{self.target}' not found in the DataFrame")

      # Separate features and target
      features = data.drop(columns=[self.target])
      target_col = data[self.target]

      # Identify numerical and categorical features
      numerical_features = features.select_dtypes(include=['int64', 'float64']).columns.tolist()
      categorical_features = features.select_dtypes(include=['object', 'category']).columns.tolist()

      # Apply One-Hot Encoding to categorical features
      encoder = OneHotEncoder(drop='first', sparse_output=False)
      encoded_cats = pd.DataFrame(encoder.fit_transform(features[categorical_features]),
                                  columns=encoder.get_feature_names_out(categorical_features))

      # Apply Standardization to numerical features
      scaler = StandardScaler()
      scaled_nums = pd.DataFrame(scaler.fit_transform(features[numerical_features]),
                                  columns=numerical_features)

      # Combine the processed numerical and categorical features
      processed_features = pd.concat([scaled_nums, encoded_cats], axis=1)

      # Remove multicollinearity using the correlation matrix approach
      corr_matrix = processed_features.corr().abs()
      upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

      # Find features with correlation greater than 0.8
      to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
      processed_features.drop(columns=to_drop, inplace=True)

      # Add target back
      processed_data = pd.concat([processed_features, target_col.reset_index(drop=True)], axis=1)

      return processed_data

  def get_sigmoid(self, wtx):
    return 1/(1 + np.exp(-wtx))

  def build_model(self):
    wold = self.w
    wnew = self.w
    epochs = 0
    x_t = self.x_train
    y_t = self.y_train

    while True:
      print("Epoch", epochs)
      wold = wnew
      Tg = 0
      for i in range(self.n):
        xi = x_t.iloc[i]
        yi = y_t.iloc[i]
        wTx = wnew.T.dot(xi)
        yhat = self.get_sigmoid(wTx)
        wnew = wnew + self.eta*(yi-yhat)*np.array(xi)

      if np.allclose(wnew, wold):
        break

      epochs+=1
    self.w = wnew


    return self.w

  def test_model(self, test_data):
    if test_data is None:
      test_x = self.x_test
    output = []

    for i in range(test_x.shape[0]):
      predicted = 1 if self.get_sigmoid(self.w.dot(test_x.iloc[i]))>0.5 else 0
      output.append((self.y_test.iloc[i], predicted))

    return output
for x1, y1 in zip(s1.test_model(test_df, 'y'), test_df['y']):
  print(x1, y1)
  