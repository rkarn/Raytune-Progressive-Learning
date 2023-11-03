import numpy as np
import pickle
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--o', default='./awid_tasks.pkl', help='output file')
parser.add_argument('--n_tasks', default=5, type=int, help='number of tasks')
parser.add_argument('--seed', default=100, type=int, help='random seed')
args = parser.parse_args()
np.random.seed(args.seed)

f = open("AWID.pickle", "rb")
X_train = pickle.load(f)
X_test = pickle.load(f) 
Y_train_all_attacks = pickle.load(f) 
Y_test_all_attacks = pickle.load(f) 
f.close()

print(X_train.shape, X_test.shape, Y_train_all_attacks.shape, Y_test_all_attacks.shape)

from keras.utils import np_utils

#task_labels = [[0,1], [2,3], [4,5], [6,7], [8,9],[1,5],[7,9],[3,8],[0,6],[4,2]]
#task_labels = [[4,2], [0,6], [3,8], [9,7], [1,5],[8,9],[6,7],[5,5],[3,2],[0,1]]
#task_labels = [[8,9], [6,7], [4,5], [2,3], [0,1]]
#task_labels = [[0,9], [7,8], [3,6], [1,4], [2,5]]
task_labels = [[0,1,2], [3,4,5], [6,7,8],[9,10,11,12], [13,14,15,16]]
#task_labels = [[0,1], [2,3,1,0],[4,5,1,2], [6,7,3,0],[8,9,4,6]]
n_tasks = len(task_labels)
nb_classes  = 17
training_datasets = []
validation_datasets = []
multihead=False

for labels in task_labels:
    idx = np.in1d(Y_train_all_attacks, labels)
    if multihead:
        label_map = np.arange(nb_classes)
        label_map[labels] = np.arange(len(labels))
        data = X_train[idx], np_utils.to_categorical(label_map[Y_train_all_attacks[idx]], len(labels))
    else:
        data = X_train[idx], np_utils.to_categorical(Y_train_all_attacks[idx], nb_classes)
        training_datasets.append(data)

for labels in task_labels:
    idx = np.in1d(Y_test_all_attacks, labels)
    if multihead:
        label_map = np.arange(nb_classes)
        label_map[labels] = np.arange(len(labels))
        data = X_test[idx], np_utils.to_categorical(label_map[Y_test_all_attacks[idx]], len(labels))
    else:
        data = X_test[idx], np_utils.to_categorical(Y_test_all_attacks[idx], nb_classes)
        validation_datasets.append(data)
        
tasks_train={}; labels_train = {}; tasks_test = {}; labels_test = {}

for i in range(len(task_labels)):
    tasks_train[str(i)] = training_datasets[i][0]
    labels_train[str(i)] = training_datasets[i][1]
    tasks_test[str(i)] = validation_datasets[i][0]
    labels_test[str(i)] = validation_datasets[i][1]
    print('Task {0} size: Trainset - {1}, {2}, Testset - {3}, {4}'.format(i,tasks_train[str(i)].shape, labels_train[str(i)].shape, tasks_test[str(i)].shape, labels_test[str(i)].shape))

Tasks_dumped = []
for i in range(len(task_labels)):
    Tasks_dumped.append((tasks_train[str(i)], labels_train[str(i)], tasks_test[str(i)], labels_test[str(i)], tasks_test[str(i)], labels_test[str(i)]))
f = open(args.o, "wb")
pickle.dump(Tasks_dumped, f)
f.close()