import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

class SoftMax:
  def __init__(self, data, target, eta):
    self.data = data
    self.n = data.shape[0]
    self.x = data.drop(columns=[target])
    self.y = data[target]
    self.x.insert(0, 'x0', 1)
    self.d = len(self.x.columns)
    self.k = len(self.y.unique())
    self.w = np.random.rand(self.k,self.d)
    self.e = eta

  def get_softmax(self, k_idx, output_neuron_values):
    return np.exp(output_neuron_values[k_idx])/np.sum(np.exp(output_neuron_values))

  def build_model(self):
    n = self.n
    k = self.k
    e = self.e
    wolds = self.w.copy()
    wnews = self.w.copy()
    epochs = 0
    x_t = self.x
    y_t = self.y

    while True:
      if epochs%1000 == 0:
        print("Epoch", epochs)
      wolds = wnews.copy()
      for i in range(n):
        x = x_t.iloc[i]
        y = y_t.iloc[i]
        output_neuron_values = []
        for j in range(k):
          wjtx = wnews[j].T.dot(x)
          output_neuron_values.append(wjtx)

        for j in range(k):
          y_act = 1 if j==y else 0
          y_hat = self.get_softmax(j, output_neuron_values)
          wnews[j] = wnews[j] + e*(y_act-y_hat)*np.array(x) # lasso: lambd*np.sign(wnews[j]) # ridge: lambda*wnews[j]

      break_flag = True
      for j in range(self.k):
        if not np.allclose(wolds[j], wnews[j]):
          break_flag = False

      if break_flag:
        break
      epochs+=1
    self.w = wnews[:]
    return self.w

  def test_model(self, test_data, target):
    output = []

    test_x = test_data.drop(columns=[target])

    test_x.insert(0, 'x0', 1)

    for i in range(test_x.shape[0]):
      outputs = []
      for j in range(self.k):
        wjtx = self.w[j].T.dot(test_x.iloc[i])
        outputs.append(wjtx)
      output.append(np.argmax(outputs))

    return output

train_df, test_df = train_test_split(df, test_size=0.33)
so1 = SoftMax(train_df, 'y', 0.1)
so1.build_model()

for x1, y1 in zip(so1.test_model(test_df, 'y'), test_df['y']):
  print(x1, y1)