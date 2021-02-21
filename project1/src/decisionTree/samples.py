import numpy as np
import matplotlib.pyplot as plt
import math
from keras.datasets import mnist
from sklearn import tree, metrics
from sklearn.preprocessing import StandardScaler
from src.utils import get_project_root

root = get_project_root()

dataset = 'MNIST'

if dataset == 'MNIST':
  sample_steps = 10
  (train_x, train_y), (test_x, test_y) = mnist.load_data()
elif dataset == 'Breast Cancer':
  sample_steps = 100
  f = open(f'{root}/data/wdbc.data', "r")
  data = f.read().split('\n')

  l_train = math.floor(len(data)*.8)
  l_test = len(data)-l_train

  train_x = np.array([k.split(',')[2:] for k in data[:l_train-1]])
  train_y = np.array([k.split(',')[1] for k in data[:l_train-1]])

  test_x = np.array([k.split(',')[2:] for k in data[l_train-1:-2]])
  test_y = np.array([k.split(',')[1] for k in data[l_train-1:-2]])

# Flatten x matrices
samples = train_x.shape[0]
train_x = train_x.reshape(samples, -1)
samples = test_x.shape[0]
test_x = test_x.reshape(samples, -1)

# Scale input data
scaler = StandardScaler()
scaler.fit(train_x)
train_x = scaler.transform(train_x)
test_x = scaler.transform(test_x)

train_scores = []
test_scores = []

sample_interval = math.floor(train_x.shape[0] / sample_steps)
sample_counts = [sample_interval*(k+1) for k in range(sample_steps)]

for count in sample_counts:
  print(f'Training Set Size: {count}')
  train_x_subset = train_x[0:count-1]
  train_y_subset = train_y[0:count-1]
  
  clf = tree.DecisionTreeClassifier(random_state=1)
  clf = clf.fit(train_x_subset, train_y_subset)
  train_scores.append(clf.score(train_x_subset, train_y_subset))
  test_scores.append(clf.score(test_x, test_y))

plt.figure()

plt.subplot(211)
plt.plot(sample_counts, np.array(train_scores)*100)
plt.ylabel('Training Set Accuracy (%)')
plt.title(f'Decision Tree - {dataset} - Training Set Size')

plt.subplot(212)
plt.plot(sample_counts, np.array(test_scores)*100)
plt.ylabel('Test Set Accuracy (%)')
plt.xlabel('Training Samples')
plt.savefig(f'{root}/images/decisionTree/{dataset}_samples.png')

plt.show()