import numpy as np
import pandas as pd


class Leaf:
    """
    stores a label: the most likely guess, based on the classification rule = majority vote
    """
    def __init__(self, guess):
        self.guess = guess


class Node:
    """
    A split:
    """
    def __init__(self, feature, left, right, entropy_score):
        self.feature = feature  # The feature to split on
        self.left = left        # Subtree for feature=False
        self.right = right      # Subtree for feature=True
        self.entropy_score = entropy_score


def entropy(f):
    """
    calculate the entropy (impurity) of a single feature f
    :param f: a pandas series of binary feature values
    :return: a float, entropy value
    cf. ESL p309
    """
    # special case: if they are all the same then just return entropy zero
    if all(item == f.iloc[0] for item in f.values):
        return 0
    total_counts = f.count()
    var_counts = f.value_counts()[0]
    p = var_counts/total_counts
    return -p*np.log2(p)-(1-p)*np.log2(1-p)


def find_best_feature(features_data):
    """
    iterate through the features and find the one with the lowest entropy as the best feature to split on
    :param features_data: a pandas data frame with features and binary values and corresponding labels in first column
    :return: the name of the feature (e.g. 'Easy?') and its entropy uncertainty
    """
    min_entropy = 1
    best_feature_name = None
    # print(features_data)
    for col in features_data:
        f = features_data[col]
        f_entropy = entropy(f)
        # print("column: ", col)
        # print("feature: ", f)
        # print("value count", f.value_counts())
        # print('entropy: ', f_entropy)
        # print("\n")
        if f_entropy < min_entropy:
            min_entropy = f_entropy
            best_feature_name = f.name
    return best_feature_name, min_entropy


def decision_tree_train(data):
    """
    train a decision tree(cf. CIML p13), using entropy for feature selection at every node (cf. ESL p309)
    :param data: pandas data frame, training data, labels in the first column and the rest are features
    :return: build the whole decision tree
    """

    y = np.array(data.iloc[:, 0])
    # x = np.array(data.iloc[:, 1:])
    # labels = data.iloc[:, 0]
    features = data.iloc[:, 1:]
    guess = np.sign(np.sum(y))  # default answer for this data: the most frequent label in data
    # base case: if all labels are the same, return a Leaf node.
    if all(item == y[0] for item in y):
        return Leaf(guess)

    # base case: if there are no remaining features, return a Leaf node.
    elif features.shape[1] == 0:
        return Leaf(guess)
    else:
        best_feature_name, best_feature_entropy = find_best_feature(features_data=features)
        f = features.loc[:, best_feature_name]
        no_set = data[f == 0].drop(columns=best_feature_name)
        yes_set = data[f == 1].drop(columns=best_feature_name)
        # recursion: build subtrees for left and right branches
        left = decision_tree_train(no_set)
        right = decision_tree_train(yes_set)
        return Node(feature=best_feature_name, left=left, right=right, entropy_score=best_feature_entropy)


def decision_tree_test(tree, test_point):
    """

    :param tree: a trained decision tree
    :param test_point: a single test point, pandas series, with features and feature values
    :return: a label (guess), 1 or -1
    """
    if isinstance(tree, Leaf):
        # strange! a node doesn't have a guess attribute
        return tree.guess
    elif isinstance(tree, Node):
        if test_point[tree.feature] == 0:
            return decision_tree_test(tree.left, test_point)
        else:
            return decision_tree_test(tree.right, test_point)


def print_tree(node, spacing=" "):
    """
    print the classification tree recursively
    :return:
    """
    # spacing = " "
    # base case
    if isinstance(node, Leaf):
        print(spacing + "Leaf: predicted=" + str(node.guess))
        return

    print(spacing + "Node: " + "question=" + str(node.feature) + ". Entropy=" +
          str(np.round(node.entropy_score, decimals=3)))

    print(spacing + '>>> n:')
    print_tree(node.left, spacing=spacing + "    ")

    print(spacing + '>>> y:')
    print_tree(node.right, spacing=spacing + "    ")


# get the training data and train

datapath = "./courseRatingData.txt"
training_data = pd.read_csv(datapath, delimiter=' ', header='infer')
training_data.iloc[:, 0] = training_data.iloc[:, 0].apply(lambda x: -1 if x < 0 else 1)
training_data.replace(to_replace='y', value=1, inplace=True)
training_data.replace(to_replace='n', value=0, inplace=True)
# print(training_data)

# util
root = decision_tree_train(training_data)
print_tree(root)
