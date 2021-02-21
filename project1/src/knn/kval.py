from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from src.utils import get_project_root
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import math

root = get_project_root()

dataset = 'MNIST'
kvals = np.arange(2, 17, 2)

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

fig, sps = plt.subplots(2, sharex=True, figsize=[9, 7])

train_scores = []
test_scores = []

for k in kvals:
  print(f'Running for K = {k}')
  clf = KNeighborsClassifier(algorithm=algo, n_neighbors=k)

  clf.fit(train_x, train_y)

  train_scores.append(clf.score(train_x, train_y))
  test_scores.append(clf.score(test_x, test_y))

sps[0].set_title(f'KNN - {dataset} - Nearest Neighbors')
sps[0].plot(kvals, np.array(train_scores)*100)
sps[0].set_ylabel('Training Set Accuracy (%)')
sps[1].plot(kvals, np.array(test_scores)*100)
sps[1].set_ylabel('Test Set Accuracy (%)')
sps[1].set_xlabel('Nearest Neighbors')
plt.savefig(f'{root}/images/knn/{dataset}_Kval.png')
plt.show()