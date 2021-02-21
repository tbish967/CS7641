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
  alpha_vals = np.arange(0.00003, 0.00008, 0.000005)
  alpha_steps = 10
  (train_x, train_y), (test_x, test_y) = mnist.load_data()
elif dataset == 'Breast Cancer':
  alpha_vals = np.arange(0, 0.4, 0.001)
  alpha_steps = 100
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

clf = tree.DecisionTreeClassifier(random_state=1)
clf = clf.fit(train_x, train_y)

# Determine effective alpha values
path = clf.cost_complexity_pruning_path(train_x, train_y)
if alpha_vals is not None:
  ccp_alphas = alpha_vals
else:
  # Easy way to figure out alpha range
  ccp_alphas = path.ccp_alphas[0::math.ceil((len(path.ccp_alphas)-1)/alpha_steps)]

# Build classifier for each alpha
for ccp_alpha in ccp_alphas:
  print(f'Alpha: {ccp_alpha}')
  clf = tree.DecisionTreeClassifier(ccp_alpha=ccp_alpha, random_state=1)
  clf.fit(train_x, train_y)
  train_scores.append(clf.score(train_x, train_y))
  test_scores.append(clf.score(test_x, test_y))

plt.figure()

plt.subplot(211)
plt.plot(ccp_alphas, np.array(train_scores)*100)
plt.ylabel('Training Set Accuracy (%)')
plt.title(f'Decision Tree - {dataset} - Alpha Value')

plt.subplot(212)
plt.plot(ccp_alphas, np.array(test_scores)*100)
plt.ylabel('Test Set Accuracy (%)')
plt.xlabel('Alpha')
plt.savefig(f'{root}/images/decisionTree/{dataset}_alpha.png')

plt.show()