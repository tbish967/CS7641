from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from src.utils import get_project_root
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import math
import time

root = get_project_root()

dataset = 'MNIST'
hl_sizes = np.arange(5, 21, 5)
iterations = 50

if dataset == 'MNIST':
  (train_x, train_y), (test_x, test_y) = mnist.load_data()
elif dataset == 'Breast Cancer':
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

fig, sps = plt.subplots(3, sharex=True, figsize=[9, 7])

for s in hl_sizes:
  print(f'Running for HL size: {s}')
  clf = MLPClassifier(activation='logistic', hidden_layer_sizes=s, max_iter=1, random_state=1, warm_start=True)
  train_scores = []
  test_scores = []
  last_hl = (s == hl_sizes[-1])

  if last_hl:
    times = []
    start = time.time()

  for it in range(iterations):
    print(f'\tIteration: {it}')
    clf.fit(train_x, train_y)
    train_scores.append(clf.score(train_x, train_y))
    test_scores.append(clf.score(test_x, test_y))

    if last_hl:
      times.append(time.time()-start)

  it_list = list(range(1, len(test_scores)+1))

  sps[0].plot(it_list, np.array(train_scores)*100, label=f'{s} Hidden Layer Nodes')
  sps[1].plot(it_list, np.array(test_scores)*100, label=f'{s} Hidden Layer Nodes')

print(len(it_list))
print(len(times))

sps[0].set_title(f'Neural Net - {dataset} - Hidden Layer Nodes')
sps[0].set_ylabel('Training Set Accuracy (%)')
sps[0].legend(prop={'size': 8})
sps[1].set_ylabel('Test Set Accuracy (%)')
sps[2].plot(it_list, times)
sps[2].set_ylabel('Runtime (s)')
sps[2].set_xlabel('Backpropagation Iterations')
plt.savefig(f'{root}/images/neuralNet/{dataset}_HLs.png')
plt.show()