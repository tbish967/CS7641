import numpy as np
import matplotlib.pyplot as plt
import math
import time
from keras.datasets import mnist
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from src.utils import get_project_root

root = get_project_root()

dataset = 'MNIST'
learn_rates = [.1, .5, 1]
estimator_counts = [20, 40, 60, 80, 100]

if dataset == 'MNIST':
  (train_x, train_y), (test_x, test_y) = mnist.load_data()
elif dataset == 'Breast Cancer':
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

fig, sps = plt.subplots(3, sharex=True, figsize=[9, 7])

for rate in learn_rates:
  print(f'Learning rate: {rate}')

  clf = GradientBoostingClassifier(n_estimators=1, learning_rate=rate, random_state=1, warm_start=True)

  train_scores = []
  test_scores = []

  last_rate = (rate == learn_rates[-1])
  if last_rate:
    times = []
    start = time.time()

  for es in estimator_counts:
    print(f'\tEstimators: {es}')
    clf.n_estimators = es
    clf.fit(train_x, train_y)
    train_scores.append(clf.score(train_x, train_y))
    test_scores.append(clf.score(test_x, test_y))
    if last_rate:
      times.append(time.time()-start)

  sps[0].plot(estimator_counts, np.array(train_scores)*100, label=f'Learning Rate: {rate}')
  sps[1].plot(estimator_counts, np.array(test_scores)*100)

sps[0].set_title(f'Boost - {dataset} - Learning Rate')
sps[0].set_ylabel('Training Set Accuracy (%)')
sps[0].legend(prop={'size': 7})
sps[1].set_ylabel('Test Set Accuracy (%)')
sps[2].plot(estimator_counts, times)
sps[2].set_ylabel('Runtime (s)')
sps[2].set_xlabel('Estimator Count')
plt.savefig(f'{root}/images/boost/{dataset}_Rates.png')
plt.show()