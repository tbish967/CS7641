from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from src.utils import get_project_root
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import math

root = get_project_root()

dataset = 'MNIST'
sample_steps = 5

if dataset == 'MNIST':
  algo = 'brute'
  (train_x, train_y), (test_x, test_y) = mnist.load_data()
elif dataset == 'Breast Cancer':
  algo = 'kd_tree'
  f = open(f'{root}/data/wdbc.data', "r")
  data = f.read().split('\n')

  l_train = math.floor(len(data)*.8)

  train_x = np.array([k.split(',')[2:] for k in data[:l_train-1]]).astype(np.float)
  train_y = np.array([int(k.split(',')[1] == 'B') for k in data[:l_train-1]])

  test_x = np.array([k.split(',')[2:] for k in data[l_train-1:-2]]).astype(np.float)
  test_y = np.array([int(k.split(',')[1] == 'B') for k in data[l_train-1:-2]])

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

sample_interval = math.floor(train_x.shape[0] / sample_steps)
sample_counts = [sample_interval*(k+1) for k in range(sample_steps)]

fig, sps = plt.subplots(2, sharex=True, figsize=[9, 7])

train_scores = []
test_scores = []

for count in sample_counts:
  print(f'Running for sample count: {count}')
  clf = KNeighborsClassifier(algorithm=algo)

  train_x_subset = train_x[0:count-1]
  train_y_subset = train_y[0:count-1]

  clf.fit(train_x_subset, train_y_subset)

  train_scores.append(clf.score(train_x_subset, train_y_subset))
  test_scores.append(clf.score(test_x, test_y))

sps[0].set_title(f'KNN - {dataset} - Training Set Size')
sps[0].plot(sample_counts, np.array(train_scores)*100,)
sps[0].set_ylabel('Training Set Accuracy (%)')
sps[1].plot(sample_counts, np.array(test_scores)*100)
sps[1].set_ylabel('Test Set Accuracy (%)')
sps[1].set_xlabel('Training Samples')
plt.savefig(f'{root}/images/knn/{dataset}_Samples.png')
plt.show()