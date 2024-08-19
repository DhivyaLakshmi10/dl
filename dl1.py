# 1. PERCEPTRON - CLASSIFICATION

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def PERCEPTRONNNNNNNN(X, y, W):
    epochs = 0
    history = [W.copy()]
    while True:
        epochs += 1
        iteration_misclassifications = 0
        for iteration in range(X.shape[0]):
            wTx = np.dot(W, X.iloc[iteration])
            yHat = 1 if wTx >= 0 else 0
            if y.iloc[iteration][0] != yHat:
                iteration_misclassifications += 1
                if y.iloc[iteration][0] == 1 and yHat == 0:
                    W = W + np.array(X.iloc[iteration])
                elif y.iloc[iteration][0] == 0 and yHat == 1:
                    W = W - np.array(X.iloc[iteration])
        history.append(W.copy())
        if iteration_misclassifications == 0: break
    return W, epochs, history

def plot_progression(weight_history):
    fig, ax = plt.subplots()
    
    ax.scatter(X[y['y'] == 0]['x1'], X[y['y'] == 0]['x2'], color='blue', label='y = 0')
    ax.scatter(X[y['y'] == 1]['x1'], X[y['y'] == 1]['x2'], color='red', label='y = 1')
    
    for i in range(1, len(weight_history) - 1):
        w0, w1, w2 = weight_history[i]
        if w2 != 0:
            x_vals = np.linspace(-5, 5, 100)
            y_vals = (-w1 * x_vals - w0) / w2
            ax.plot(x_vals, y_vals, linestyle='--', color='gray')

    w0, w1, w2 = weight_history[-1]
    if w2 != 0:
        x_vals = np.linspace(-5, 5, 100)
        y_vals = (-w1 * x_vals - w0) / w2
        ax.plot(x_vals, y_vals, linestyle='-', color='green', linewidth=2, label='Final Decision Boundary')

    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_title('Progression of Decision Boundary')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.legend()
    plt.show()

# OR Gate
X = pd.DataFrame({'x0': [1, 1, 1, 1], 'x1': [0, 0, 1, 1], 'x2': [0, 1, 0, 1]})
y = pd.DataFrame({'y': [0, 1, 1, 1]})
W, epochs, weight_history = PERCEPTRONNNNNNNN(X, y, [1, 1, 1])
print(f"Final W: {list(W)} in {epochs} epochs.")
plot_progression(weight_history)

#2. PERCEPTRON - REGRESSION
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Dataset
# x1 = Soap | y = Sud
df = pd.DataFrame({"x0":np.ones(7), "x1": np.linspace(4.0, 7.0, 7), "y": np.array([33, 42, 45, 51, 53, 61, 62])})

X = df.drop(columns=["y"])
y = df["y"]

def PERCEPTRONNFORREGRESSION(X, y, W, η):
    # convergence thru diff threshold
    # while not converged
    epochs = 1
    history = [W.copy()]
    while True:
        # init iteration_gradient
        iteration_gradient = np.array([0.0, 0.0])
        # for each point in the dataset (x0, x1)
        for iteration in range(X.shape[0]):
            # find yHat = W.T.X
            yHat = np.dot(W.T, np.array(X.iloc[iteration]))
            # find grad = d/dW (Error fn)
            grad = (y.iloc[iteration] - yHat) * np.array(X.iloc[iteration])
            # update iteration_gradient
            iteration_gradient += grad
        # check for convergence
        wNew = W + η * iteration_gradient
        history.append(wNew.copy())
        if np.allclose(wNew, W):
            break
        # update W with the iteration_gradient
        W = wNew
        epochs += 1
    return W, epochs, history

def plot_line_progression(W_history, X):
    # Assuming W_history is a list of weight vectors over epochs
    _, ax = plt.subplots(figsize=(8, 8))
    
    # Plot each line corresponding to W in W_history
    for W in W_history:
        if np.abs(W[1]) > 1e-8:  # Avoid division by zero if w1 is close to zero
            x0_vals = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
            x1_vals = -W[0] / W[1] * x0_vals
            ax.plot(x0_vals, x1_vals, label=f'Line: {W[0]:.2f}x0 + {W[1]:.2f}x1 = 0')
    
    # Scatter plot of the data points (assuming X is a 2D array)
    ax.scatter(X[:, 0], X[:, 1], color='red', label='Data Points')
    
    ax.set_xlabel('x0')
    ax.set_ylabel('x1')
    ax.set_title('Progression of Line Equation wTx = 0')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

W = np.array([0, 0])
W, epochs, w_history = PERCEPTRONNFORREGRESSION(X, y, W, 0.0001)
print(f"{W} in {epochs} epochs.")

# Predictions using Perceptron
X_vals = np.linspace(min(X["x1"]), max(X["x1"]), 100)
X_pred = np.column_stack((np.ones_like(X_vals), X_vals))
y_pred = np.dot(X_pred, W)

# Calculate metrics for Perceptron
sse_perceptron = np.sum((y - np.dot(X, W)) ** 2)
mse_perceptron = mean_squared_error(y, np.dot(X, W))
r2_perceptron = r2_score(y, np.dot(X, W))

print(f"Perceptron - SSE: {sse_perceptron}, MSE: {mse_perceptron}, R-squared: {r2_perceptron}")


# Plot Perceptron model
plt.scatter(X["x1"], y, color='red', label='Data Points')
plt.plot(X_vals, y_pred, label='Perceptron Model', color='blue')
plt.xlabel('Soap')
plt.ylabel('Sud')
plt.title('Perceptron Regression')
plt.legend()
plt.show()

# Train Simple Linear Regression model
lr = LinearRegression()
lr.fit(X, y)
y_pred_lr = lr.predict(X)

# Predictions using Linear Regression
y_pred_lr_vals = lr.predict(np.column_stack((np.ones_like(X_vals), X_vals)))

# Calculate metrics for Simple Linear Regression
sse_lr = np.sum((y - y_pred_lr) ** 2)
mse_lr = mean_squared_error(y, y_pred_lr)
r2_lr = r2_score(y, y_pred_lr)

print(f"Linear Regression - SSE: {sse_lr}, MSE: {mse_lr}, R-squared: {r2_lr}")

# Plot Linear Regression model
plt.scatter(X["x1"], y, color='red', label='Data Points')
plt.plot(X_vals, y_pred_lr_vals, label='Linear Regression Model', color='green')
plt.xlabel('Soap')
plt.ylabel('Sud')
plt.title('Linear Regression')
plt.legend()
plt.show()


print(f"Perceptron - SSE: {sse_perceptron}, MSE: {mse_perceptron}, R-squared: {r2_perceptron}")
print(f"Linear Regression - SSE: {sse_lr}, MSE: {mse_lr}, R-squared: {r2_lr}")

#3. SIGMOID NEURON 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def SIGMOOIDDDNEURONNN(X, y, W, η, plot=True): 
    # keep track of epochs
    epochs = 0
    # on infinite loop
    while True:
        epochs += 1
        wNew = W
        # for each item in the dataset (xi, yi)
        for iteration in range(X.shape[0]):
            # calculate dot pdt. of W and xi [W.T, X]
            wTx = np.dot(W, X.iloc[iteration])
            # calculate the Sigmoid(wTx)
            yHat = 1/(1 + np.exp(-wTx))
            # Update the weights
            wNew = wNew + η * (y.iloc[iteration][0] - yHat) * np.array(X.iloc[iteration])
        # check for convergence based on the convergence criteria
        # here, it's diff between W vectors
        # break the outer loop if convergence
        if np.allclose(wNew, W):
            break
        W = wNew
    
    if plot:
        fig, ax = plt.subplots()
        ax.scatter(X[y['y'] == 0]['x1'], X[y['y'] == 0]['x2'], color='blue', label='y = 0')
        ax.scatter(X[y['y'] == 1]['x1'], X[y['y'] == 1]['x2'], color='red', label='y = 1')

        w0, w1, w2 = W
        if w2 != 0:
            x_vals = np.linspace(-5, 5, 100)
            y_vals = (-w1 * x_vals - w0) / w2
            ax.plot(x_vals, y_vals, linestyle='-', color='green', linewidth=2, label='Final Decision Boundary')
        ax.set_xlim([-5, 5])
        ax.set_ylim([-5, 5])
        ax.set_title('Plot of Decision Boundary')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.legend()
        plt.grid(True)
        plt.show() 
    return W, epochs

# OR Gate
X = pd.DataFrame({'x0': [1, 1, 1, 1], 'x1': [0, 0, 1, 1], 'x2': [0, 1, 0, 1]})
y = pd.DataFrame({'y': [0, 1, 1, 1]})
W, epochs = SIGMOOIDDDNEURONNN(X, y, [1, 1, 1], 0.05)
print(f"Final W: {list(W)} in {epochs} epochs.")

# SIGMOID NEURON - 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def SIGMOOIDDDNEURONNN(X, y, W, η, plot=True): 
# keep track of epochs
  epochs = 0
  # on infinite loop
  while True:
      epochs += 1
      wNew = W
      # for each item in the dataset (xi, yi)
      for iteration in range(X.shape[0]):
          # calculate dot pdt. of W and xi [W.T, X]
          wTx = np.dot(W, X.iloc[iteration])
          # calculate the Sigmoid(wTx)
          yHat = 1/(1 + np.exp(-wTx))
          # Update the weights
          wNew = wNew + η * (y.iloc[iteration][0] - yHat) * np.array(X.iloc[iteration])
      print(epochs, wNew)
      # check for convergence based on the convergence criteria
      # here, it's diff between W vectors
      # break the outer loop if convergence
      if np.allclose(wNew, W):
          break
      W = wNew
  
  if plot:
      fig, ax = plt.subplots()
      ax.scatter(X[y['y'] == 0]['x1'], X[y['y'] == 0]['x2'], color='blue', label='y = 0')
      ax.scatter(X[y['y'] == 1]['x1'], X[y['y'] == 1]['x2'], color='red', label='y = 1')

      w0, w1, w2 = W
      if w2 != 0:
          x_vals = np.linspace(-5, 5, 100)
          y_vals = (-w1 * x_vals - w0) / w2
          ax.plot(x_vals, y_vals, linestyle='-', color='green', linewidth=2, label='Final Decision Boundary')
      
      ax.set_xlim([-5, 5])
      ax.set_ylim([-5, 5])
      ax.set_title('Plot of Decision Boundary')
      ax.set_xlabel('x1')
      ax.set_ylabel('x2')
      ax.legend()
      plt.grid(True)
      plt.show() 
  return W, epochs

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
X['x0'] = np.ones(X.shape[0])
y = pd.DataFrame((iris.target == 0).astype(int), columns=['y']) # Class 0 vs all others (binary classification)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

W_IRIS, epochs = SIGMOOIDDDNEURONNN(pd.DataFrame(X_train), pd.DataFrame(y_train), np.ones(X_train.shape[1]), 0.05, False)

# 4. MLP - XOR


import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0]).reshape(-1, 1)  # XOR problem

N, ip_dim = X.shape
hidden_layer_dim = 4
W1 = np.random.random((ip_dim, hidden_layer_dim))  # Weights for input to hidden layer
op_dim = len(y.T)
W2 = np.random.random((hidden_layer_dim, op_dim))  # Weights for hidden to output layer

no_of_epochs = 10000
η = 1  # Learning rate

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

history = []

for _ in range(no_of_epochs):
    # Forward pass
    hidden_layer_output = sigmoid(np.dot(X, W1))  # Output of hidden layer
    output_layer_output = sigmoid(np.dot(hidden_layer_output, W2))  # Output of network
    
    # Compute the error
    error = y - output_layer_output
    delta_output = error * sigmoid_prime(output_layer_output)
    
    # Compute the error for hidden layer
    delta_hidden = np.dot(delta_output, W2.T) * sigmoid_prime(hidden_layer_output)
    
    # Update weights
    W1 += η * np.dot(X.T, delta_hidden)
    W2 += η * np.dot(hidden_layer_output.T, delta_output)
    
    # Track the loss
    history.append(np.sum(error ** 2))

plt.plot(history)
plt.xlabel('Epochs')
plt.ylabel('Sum of Squared Error')
plt.title('Training Progress')
plt.show()

# Testing the trained network
for x, yy in zip(X, y):
    hidden_layer_prediction = sigmoid(np.dot(x, W1))  # Feedforward
    final_prediction = sigmoid(np.dot(hidden_layer_prediction, W2))  # Feedforward
    print(f"Input: {x}, Prediction: {(final_prediction > 0.5).astype(int)}, Actual: {yy}")

# 5. SOFTMAX NEURON 

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import random
from sklearn.metrics import classification_report

iris = load_iris()

X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.DataFrame(iris.target, columns=['y'])

df = pd.DataFrame(X)
df['y'] = y

class Perceptron:
    def __init__(self, df):
        x, y = df.loc[:, df.columns!='y'], df['y']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)
        self.x_train = x_train
        self.x_train.insert(0,'x0',1)
        self.x_test = x_test
        self.x_test.insert(0,'x0',1)
        self.y_train = y_train
        self.y_test = y_test
        self.d_features = len(x.columns)
        self.n_datapoints = self.x_train.shape[0]
        self.l_classes = len(set(y))
        self.Ws = [np.array([0 for _ in range(self.d_features+1)]) for _ in range(self.l_classes)]

    def display_attr(self):
        # print(self.Ws)
        for i, elt in enumerate(self.Ws):
            print(i, elt)

    def softmax(self, index, output_neuron_values):
        numerator = np.exp(output_neuron_values[index])
        denominator = 0

        for val in output_neuron_values:
            denominator += np.exp(val)

        return numerator/denominator
    
    def argmax(self, output_neuron_values):
        maxidx = -1
        maxval = -1e12

        for i, elt in enumerate(output_neuron_values):
            if elt>maxval:
                maxval = elt
                maxidx = i

        return maxidx

    def build_model(self):
        n = self.n_datapoints
        d = self.d_features
        l = self.l_classes

        Wolds = self.Ws[:]
        Wnews = self.Ws[:]
        epochs = 0
        eta = random.choice(np.linspace(0, 0.1, 1000))
        
        x = self.x_train
        y = self.y_train

        printer_steps = set(int(np.floor(i)) for i in np.linspace(0, n, 50))

        while True:
            if epochs!=0:
                for i, elt in enumerate(Wolds):
                  Wnews[i] = elt

            if epochs%1000==0:
                print(f"Running Epoch {epochs+1} ", end='')

            for i in range(n):
                if epochs%1000==0 and i in printer_steps:
                    print("==", end='')

                x_vec = x.iloc[i]
                y_class = y.iloc[i]
                y_true = 0

                output_neuron_values = []

                for j in range(l):
                    Wj = Wolds[j]
                    WjTx = Wj.T.dot(x_vec)
                    output_neuron_values.append(WjTx)

                for j in range(l):
                    y_true = 1 if j==y_class else 0

                    softmax_probability = self.softmax(j, output_neuron_values)
                    Wolds[j] = Wolds[j] - eta*(softmax_probability-y_true)*np.array(x_vec)

            if epochs%1000==0:
                print(">")
                for i, elt in enumerate(Wolds):
                   print(f"Old vs New Weight of Hyperplane{i+1}")
                   print(f"Wold {i} {Wolds[i]}, Wnew {i} {Wnews[i]}")

            brkflg = True
            epochs += 1

            # print(Wolds, Wnews)
            for i, elt in enumerate(Wolds):
                if not np.allclose(Wolds[i],Wnews[i]):
                    brkflg = False

            if brkflg:
                break

        self.Ws = Wnews

        print(f"Model training Successful at epochs {epochs}")
        print(f"Final Weights: ")
        for i, wi in enumerate(self.Ws):
            print(i, wi)

    def test_model(self):
        x = self.x_test
        n = x.shape[0]
        Ws = self.Ws
        l = self.l_classes
        y_trues = np.array(self.y_test)
        y_preds = []

        for i in range(n):
            x_vec = x.iloc[i]
            y_true = 0

            output_neuron_values = []

            for j in range(l):
                Wj = Ws[j]
                WjTx = Wj.T.dot(x_vec)
                output_neuron_values.append(WjTx)

            y_preds.append(self.argmax(output_neuron_values))

        y_preds = np.array(y_preds)

        print(classification_report(y_trues, y_preds))

x, y = df.loc[:, df.columns!='y'], df['y']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

p1 = Perceptron(df)
p1.display_attr()
np.exp(1)
p1.build_model()

p1.test_model()
print(p1.Ws)

x = p1.x_test
y = p1.y_test
n = x.shape[0]
Ws = p1.Ws
l = p1.l_classes
y_trues = np.array(p1.y_test)
y_preds = []
y_pred_probs = []

for i in range(n):
    x_vec = x.iloc[i]
    y_true = 0
    y_class = y.iloc[i]

    output_neuron_values = []

    for j in range(l):
        Wj = Ws[j]
        WjTx = Wj.T.dot(x_vec)
        output_neuron_values.append(WjTx)

    y_preds.append(p1.argmax(output_neuron_values))

    softmax_probabilities = []

    for j in range(l):
        y_true = 1 if j==y_class else 0
        softmax_probability = p1.softmax(j, output_neuron_values)
        softmax_probabilities.append(softmax_probability)

    y_pred_probs.append(softmax_probabilities)

y_preds = np.array(y_preds)
y_pred_probs = np.array(y_pred_probs)

print(classification_report(y_trues, y_preds))

from sklearn.metrics import roc_auc_score
# 1v1
roc_auc_score(y_trues, y_pred_probs, multi_class='ovo')

roc_auc_score(y_trues, y_pred_probs, multi_class='ovr')

# 6.softmax neuron regularized

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import random
from sklearn.metrics import classification_report

iris = load_iris()

X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.DataFrame(iris.target, columns=['y'])

df = pd.DataFrame(X)
df['y'] = y

df.head()

class Perceptron:
    def __init__(self, df):
        x, y = df.loc[:, df.columns!='y'], df['y']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)
        self.x_train = x_train
        self.x_train.insert(0,'x0',1)
        self.x_test = x_test
        self.x_test.insert(0,'x0',1)
        self.y_train = y_train
        self.y_test = y_test
        self.d_features = len(x.columns)
        self.n_datapoints = self.x_train.shape[0]
        self.l_classes = len(set(y))
        self.Ws = [np.array([0 for _ in range(self.d_features+1)]) for _ in range(self.l_classes)]

    def display_attr(self):
        # print(self.Ws)
        for i, elt in enumerate(self.Ws):
            print(i, elt)

    def softmax(self, index, output_neuron_values):
        numerator = np.exp(output_neuron_values[index])
        denominator = 0

        for val in output_neuron_values:
            denominator += np.exp(val)

        return numerator/denominator

    def argmax(self, output_neuron_values):
        maxidx = -1
        maxval = -1e12

        for i, elt in enumerate(output_neuron_values):
            if elt>maxval:
                maxval = elt
                maxidx = i

        return maxidx

    def build_model(self):
        n = self.n_datapoints
        d = self.d_features
        l = self.l_classes

        Wolds = self.Ws[:]
        Wnews = self.Ws[:]
        epochs = 0
        # eta = random.choice(np.linspace(0, 0.1, 1000))
        eta = 0.001
        lb = 0.5
        x = self.x_train
        y = self.y_train

        printer_steps = set(int(np.floor(i)) for i in np.linspace(0, n, 50))

        while True:
            if epochs!=0:
                for i, elt in enumerate(Wolds):
                    Wnews[i] = elt

            if epochs%1000==0:
                print(f"Running Epoch {epochs+1} ", end='')

            for i in range(n):
                if epochs%1000==0 and i in printer_steps:
                    print("==", end='')
                x_vec = x.iloc[i]
                y_class = y.iloc[i]
                y_true = 0

                output_neuron_values = []

                for j in range(l):
                    Wj = Wolds[j]
                    WjTx = Wj.T.dot(x_vec)
                    output_neuron_values.append(WjTx)

                for j in range(l):
                    y_true = 1 if j==y_class else 0

                    softmax_probability = self.softmax(j, output_neuron_values)
                    Wolds[j] = Wolds[j] - eta*((softmax_probability-y_true)*np.array(x_vec)+lb*Wolds[j])

            if epochs%1000==0:
                print(">")
                for i, elt in enumerate(Wolds):
                    print(f"Old vs New Weight of Hyperplane{i+1}")
                    print(f"Wold {i} {Wolds[i]}, Wnew {i} {Wnews[i]}")

            brkflg = True
            epochs += 1

            # print(Wolds, Wnews)
            for i, elt in enumerate(Wolds):
                if not np.allclose(Wolds[i],Wnews[i]):
                    brkflg = False

            if brkflg:
                break

        self.Ws = Wnews
        print(f"Model training Successful at epochs {epochs}")
        print(f"Final Weights: ")
        for i, wi in enumerate(self.Ws):
            print(i, wi)

    def test_model(self):
        x = self.x_test
        n = x.shape[0]
        Ws = self.Ws
        l = self.l_classes
        y_trues = np.array(self.y_test)
        y_preds = []

        for i in range(n):
            x_vec = x.iloc[i]
            y_true = 0

            output_neuron_values = []

            for j in range(l):
                Wj = Ws[j]
                WjTx = Wj.T.dot(x_vec)
                output_neuron_values.append(WjTx)

            y_preds.append(self.argmax(output_neuron_values))

        y_preds = np.array(y_preds)

        print(classification_report(y_trues, y_preds))

x, y = df.loc[:, df.columns!='y'], df['y']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

p1 = Perceptron(df)
p1.display_attr()
np.exp(1)

p1.build_model()

p1.test_model()

x = p1.x_test
y = p1.y_test
n = x.shape[0]
Ws = p1.Ws
l = p1.l_classes
y_trues = np.array(p1.y_test)
y_preds = []
y_pred_probs = []

for i in range(n):
    x_vec = x.iloc[i]
    y_true = 0
    y_class = y.iloc[i]

    output_neuron_values = []

    for j in range(l):
        Wj = Ws[j]
        WjTx = Wj.T.dot(x_vec)
        output_neuron_values.append(WjTx)

    y_preds.append(p1.argmax(output_neuron_values))

    softmax_probabilities = []

    for j in range(l):
        y_true = 1 if j==y_class else 0
        softmax_probability = p1.softmax(j, output_neuron_values)
        softmax_probabilities.append(softmax_probability)

    y_pred_probs.append(softmax_probabilities)

y_preds = np.array(y_preds)
y_pred_probs = np.array(y_pred_probs)

print(classification_report(y_trues, y_preds))

from sklearn.metrics import roc_auc_score
# 1v1
roc_auc_score(y_trues, y_pred_probs, multi_class='ovo')
roc_auc_score(y_trues, y_pred_probs, multi_class='ovr')

# 7. basic MLP from tensorflow

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Activation
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Cast the records into float values
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# normalize image pixel values by dividing
# by 255
gray_scale = 255
x_train /= gray_scale
x_test /= gray_scale

print("Feature matrix:", x_train.shape)
print("Target matrix:", x_test.shape)
print("Feature matrix:", y_train.shape)
print("Target matrix:", y_test.shape)

fig, ax = plt.subplots(10, 10)
plt.axis('off')
k = 0
for i in range(10):
	for j in range(10):
		ax[i][j].imshow(x_train[k].reshape(28, 28),
						aspect='auto')
		k += 1
plt.show()

model = Sequential([
	Flatten(input_shape=(28, 28)),

	# dense layer 1
	Dense(256, activation='sigmoid'),

	# dense layer 2
	Dense(128, activation='sigmoid'),

	# output layer
	Dense(10, activation='sigmoid'),
])

model.compile(optimizer='adam',
			loss='sparse_categorical_crossentropy',
			metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10,
		batch_size=2000,
		validation_split=0.2)

results = model.evaluate(x_test, y_test, verbose = 0)
print('test loss, test acc:', results)

#8. classification using multilayer perceptron using MNIST dataset

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize the data
X_train, X_test = X_train / 255.0, X_test / 255.0

# Flatten the images
X_train_flat = X_train.reshape(-1, 28*28)
X_test_flat = X_test.reshape(-1, 28*28)

model = models.Sequential()
model.add(layers.Dense(128, activation='relu', input_shape=(28*28,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))  # Output layer with 10 neurons (one for each digit)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train_flat, y_train, epochs=10, batch_size=32, validation_split=0.2)

test_loss, test_acc = model.evaluate(X_test_flat, y_test)
print(f"Test Accuracy: {test_acc}")

plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

predictions = model.predict(X_test_flat)

# Plot some test images along with their predicted and actual labels
plt.figure(figsize=(12, 12))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(X_test[i], cmap='gray')
    plt.title(f"Pred: {np.argmax(predictions[i])}, Actual: {y_test[i]}")
    plt.axis('off')
plt.show()

#9.regression using multilayer perceptron using iris dataset

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.data[:, 3]  # Let's predict petal width (the 4th feature)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

mlp = MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)

y_pred = mlp.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)  # Diagonal line
plt.title('Predicted vs. Actual Petal Width (Regression)')
plt.xlabel('Actual Petal Width')
plt.ylabel('Predicted Petal Width')
plt.grid(True)
plt.show()

#9. SIGMOID NEURON - RIDGE

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, df):
        self.n_datapoints = df.shape[0]
        self.x = df.loc[:, df.columns!='y']
        self.d_features = self.x.shape[1]
        self.y = df['y']
        self.W = np.array([1 for _ in range(self.d_features+1)])
        self.x.insert(0, 'x0', 1)
    
    def display_attr(self):
        print(self.n_datapoints, self.d_features)
        print(self.W)

    def sig_activate(self, y_hat):
        return 1/(1+np.exp(-y_hat))
    
    def build_model(self):
        Wold = W = self.W
        epochs = 0
        n_datapoints = self.n_datapoints
        d_features = self.d_features
        x = self.x
        y = self.y
        eta = 0.05
        lb = 0.01
        
        print(f"The Learning Rate is {eta}")
        print(f"The initial Weights are {Wold}")
        while True:
            if epochs!=0:
                Wold = W
            
            print(f"Running Epoch {epochs+1} ===========", end='')
            
            for i in range(n_datapoints):                
                x_vec = x.iloc[i]
                wTx = W.T.dot(x_vec)
                
                y_true = y.iloc[i]
                y_hat = self.sig_activate(wTx)
                W = W - eta*((y_true-y_hat)*y_hat*(1-y_hat)*np.array(x_vec)+lb*W)
            
            epochs += 1
            print("==>")
            print(f"Old Weights {Wold}")
            print(f"New Weights: {W}")
            
            if np.allclose(W, Wold):
                break
        
        self.W = W
        print("Model Building done., Hyperplane weights optimized.")
    
    def plot_graph(self, W):
        w0, w1, w2 = W
        x = self.x
        y = self.y
        x1 = np.linspace(-2, 2, 100)
        x2 = (-w0-w1*x1)/w2


        plt.scatter(x['x1'], x['x2'], c=y, cmap='coolwarm')
        plt.plot(x1, x2, '-r')
        plt.legend(['Datapoints', 'Hyperplane - (wTx = 0)'])
        plt.show()

or_df = pd.DataFrame({'x1':[0, 0, 1, 1], 'x2':[0, 1, 0, 1], 'y':[0, 1, 1, 1]})
p1 = Perceptron(or_df)
p1.display_attr()
print(p1.W*0.5 + 1)
p1.build_model()
p1.plot_graph(p1.W)

#10.REGULARIZATION - LINEAR REGRESSION

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# Dataset
df = pd.DataFrame({"x0": np.ones(7), "x1": np.linspace(4.0, 7.0, 7), "y": np.array([33, 42, 45, 51, 53, 61, 62])})

# Prepare features and target
X = df[['x0', 'x1']]
y = df['y']

# Linear Regression
linear_regressor = LinearRegression()
linear_regressor.fit(X, y)
linear_coefficients = linear_regressor.coef_

# Ridge Regression
ridge_regressor = Ridge(alpha=1.0)
ridge_regressor.fit(X, y)
ridge_coefficients = ridge_regressor.coef_

# Lasso Regression
lasso_regressor = Lasso(alpha=0.1)
lasso_regressor.fit(X, y)
lasso_coefficients = lasso_regressor.coef_

# Compare the coefficients
print("Linear Regression Coefficients:", linear_coefficients)
print("Ridge Regression Coefficients:", ridge_coefficients)
print("Lasso Regression Coefficients:", lasso_coefficients)

