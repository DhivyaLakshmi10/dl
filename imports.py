import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import random
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report

from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Flatten, Dense

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
'''
file names

generic_mlp
mlp_classification
mlp_mnist
mlp_regression_iris
mlp_xor
preprocess
sigmoid_neuron
softmax
'''