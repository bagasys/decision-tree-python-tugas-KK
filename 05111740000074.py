import csv
import math
from random import randrange

def loadDataset(filePath, datasetResult=[]):
    with open(filePath, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        data = []
        for x in range(len(dataset)):
            data.append([])
            try:
                for y in range(1, 11):
                    if y == 10:
                        data[x].append((dataset[x][y]))
                    else:
                        data[x].append(float(dataset[x][y]))
                datasetResult.append(data[x])
            except:
                pass

def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)

def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

def grouping_by_attribute(index, comparator, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < comparator:
            left.append(row)
        else:
            right.append(row)
    return left, right

def split_by_entropy(dataset, entropy_parent):
    class_list = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = -1, -1, -1, None
    for index in range(len(dataset[0]) - 1):
        for row in dataset:
            groups = grouping_by_attribute(index, row[index], dataset)
            gain = gainCalc(groups, entropy_parent)
            if gain > b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gain, groups
    return {'index': b_index, 'value': b_value, 'groups': b_groups}


def split_entropy(node, max_depth, min_size, depth):
    left, right = node['groups']
    del (node['groups'])
    # check for a no split
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    # check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    # process left child
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = split_by_entropy(left, entropy(left))
        split_entropy(node['left'], max_depth, min_size, depth + 1)
    # process right child
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = split_by_entropy(right, entropy(right))
        split_entropy(node['right'], max_depth, min_size, depth + 1)


def build_tree_entropy(train, entropy_parent, max_depth, min_size):
    root = split_by_entropy(train, entropy_parent)
    # print(root)
    split_entropy(root, max_depth, min_size, 1)
    return root


def gainCalc(groups, entropy_parent):
    n_instances = float(sum([len(group) for group in groups]))
    # tmp = []
    gain_tmp = 0
    for group in groups:
        size = float(len(group))
        if size == 0:
            continue
        tmp = entropy(group)
        tmp = size/n_instances * tmp
        gain_tmp += tmp
    gain = entropy_parent - gain_tmp
    return gain

def entropy(group):
    classes = list(set(row[-1] for row in group))
    ent = 0.0
    score = 0.0
    size = len(group)
    for class_val in classes:
        p = [row[-1] for row in group].count(class_val) / size
        if p == 0:
            score += 0
        else:
            score += p * math.log2(p)
    ent = score * -1
    return ent


def split_gini(node, max_depth, min_size, depth):
    left, right = node['groups']
    del (node['groups'])
    # check for a no split
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    # check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    # process left child
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = split_by_gini(left)
        split_gini(node['left'], max_depth, min_size, depth + 1)
    # process right child
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = split_by_gini(right)
        split_gini(node['right'], max_depth, min_size, depth + 1)

def gini_index(groups, classes):
    # count all samples at split point
    n_instances = float(sum([len(group) for group in groups]))
    # sum weighted Gini index for each group
    gini = 0.0
    for group in groups:
        size = float(len(group))
        # avoid divide by zero
        if size == 0:
            continue
        score = 0.0
        # score the group based on the score for each class
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        # weight the group score by its relative size
        gini += (1.0 - score) * (size / n_instances)
    return gini

def split_by_gini(dataset):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    for index in range(len(dataset[0]) - 1):
        for row in dataset:
            groups = grouping_by_attribute(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index': b_index, 'value': b_value, 'groups': b_groups}


def build_tree_gini(train, max_depth, min_size):
    root = split_by_gini(train)
    split_gini(root, max_depth, min_size, 1)
    return root


def print_tree(node, depth=0):
    if isinstance(node, dict):
        print('%s[X%d < %.3f]' % ((depth * '-', (node['index'] + 1), node['value'])))
        print_tree(node['left'], depth + 1)
        print_tree(node['right'], depth + 1)
    else:
        print('%s[%s]' % ((depth * '-', node)))

def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


def precision(actual, predicted):
    true_positive = 0
    false_positive = 0
    for i in range(len(actual)):
        if predicted[i] == '4':
            if actual[i] == predicted[i]:
                true_positive += 1
            else:
                false_positive += 1
    actual_results = true_positive + false_positive
    if actual_results == 0:
        return 0
    else:
        return true_positive / float(actual_results) * 100.0


def recall(actual, predicted):
    true_positive = 0
    false_negative = 0
    for i in range(len(actual)):
        if actual[i] == '4':
            if actual[i] == predicted[i]:
                true_positive += 1
            else:
                false_negative += 1
    predicted_results = true_positive + false_negative
    if predicted_results == 0:
        return 0
    else:
        return true_positive / float(predicted_results) * 100.0

filename = 'dataset/breast-cancer-wisconsin.data'
k = int(input('K untuk Cross-Validation: '))
max_depth = int(input ('Kedalaman maximum dari tree: '))
min_size = int(input ('Jumlah row minimum dalam satu node: '))
dataset = []
loadDataset(filename, dataset)
folds = cross_validation_split(dataset, k)
impurity = -1
while(impurity != '1' and impurity != '2'):
    impurity = input('Rumus Impurity yang digunakan:\n1.Entropy\n2.Gini\n')
print('melakukan kalkulasi...')
scores = []
scores.append([])
scores.append([])
scores.append([])
trees = []

for fold in folds:
    print('.')
    train_set = list(folds)
    train_set.remove(fold)
    train_set = sum(train_set, [])
    test_set = list()

    for row in fold:
        test_set.append(list(row))

    if impurity == '1':
        entropy_parent = entropy(train_set)
        tree = build_tree_entropy(train_set, entropy_parent, max_depth, min_size)
    else:
        tree = build_tree_gini(train_set, max_depth, min_size)


    trees.append(tree)
    predictions = list()
    for row in test_set:
        prediction = predict(tree, row)
        predictions.append(prediction)
    actual = [row[-1] for row in fold]
    accuracy = accuracy_metric(actual, predictions)
    precisionScore = precision(actual, predictions)
    recallScore = recall(actual, predictions)
    scores[0].append(accuracy)
    scores[1].append(precisionScore)
    scores[2].append(recallScore)

total = 0
best = 0
bestIndex = 0
counter=0

print('\nAkurasi: ', end = '=>')
for akurasi in scores[0]:
    print('%.3f%%' % akurasi,  end = ', ')
    total+=akurasi
    if akurasi > best :
        bestIndex = counter
    counter+=1

print('\nRata-rata: %.3f%%' %(total/len(scores[0])))
total = 0
print('\nPresisi: ', end = '=>')
for presisi in scores[1]:
    print('%.3f%%' % presisi,  end = ', ')
    total+=presisi
print('\nRata-rata: %.3f%%' %(total/len(scores[1])))
total = 0
print('\nRecall: ', end='=>')
for recall in scores[2]:
    print('%.3f%%' % recall, end=', ')
    total += recall
print('\nRata-rata: %.3f%%\n' %(total/len(scores[2])))
total = 0

isGambar = int(input('gambar tree untuk akurasi yang terbaik?(1/0)'))
if (isGambar != 0):
    print('Tree: ')
    print_tree(trees[bestIndex])