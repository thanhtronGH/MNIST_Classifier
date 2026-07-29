import numpy as np 
import matplotlib.pyplot as plt 
 
# --- Gini impurity 
def gini_index(y): 
    if len(y) == 0: 
        return 0 
    classes, counts = np.unique(y, return_counts=True) 
    probs = counts / len(y) 
    return 1 - np.sum(probs ** 2) 
 
# --- Data split 
def split_data(X, y, feature_index, threshold): 
    left_mask = X[:, feature_index] <= threshold 
    right_mask = ~left_mask 
    return X[left_mask], y[left_mask], X[right_mask], y[right_mask] 
 
# --- Best split using random subset of features 
def best_split(X, y, feature_indices): 
    best_gini = float('inf') 
    best_feature = None 
    best_threshold = None 
 
    for feature_index in feature_indices: 
        thresholds = np.unique(X[:, feature_index]) 
        for threshold in thresholds: 
            X_l, y_l, X_r, y_r = split_data(X, y, feature_index, threshold) 
            if len(y_l) == 0 or len(y_r) == 0: 
                continue 
            gini_l = gini_index(y_l) 
            gini_r = gini_index(y_r) 
            weighted = (len(y_l) / len(y)) * gini_l + (len(y_r) / len(y)) * gini_r 
            if weighted < best_gini: 
                best_gini = weighted 
                best_feature = feature_index 
                best_threshold = threshold 
 
    return best_feature, best_threshold 
 
# --- Tree node class 
class Node: 
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, 
value=None): 
        self.feature_index = feature_index 
        self.threshold = threshold 
        self.left = left 
        self.right = right 
        self.value = value 
 
# --- Build a tree recursively 
def build_tree(X, y, max_depth, min_samples_split=2, depth=0, max_features=None): 
    if len(np.unique(y)) == 1 or len(y) < min_samples_split or depth >= max_depth: 
        values, counts = np.unique(y, return_counts=True) 
        return Node(value=values[np.argmax(counts)]) 
 
    n_features = X.shape[1] 
    feature_indices = np.random.choice(n_features, max_features or n_features, 
replace=False) 
    feature_index, threshold = best_split(X, y, feature_indices) 
 
    if feature_index is None: 
        values, counts = np.unique(y, return_counts=True) 
        return Node(value=values[np.argmax(counts)]) 
 
    X_l, y_l, X_r, y_r = split_data(X, y, feature_index, threshold) 
    left = build_tree(X_l, y_l, max_depth, min_samples_split, depth + 1, 
max_features) 
    right = build_tree(X_r, y_r, max_depth, min_samples_split, depth + 1, 
max_features) 
    return Node(feature_index, threshold, left, right) 
 
# --- Predict a single point 
def predict_single(x, node): 
    if node.value is not None: 
        return node.value 
    if x[node.feature_index] <= node.threshold: 
        return predict_single(x, node.left) 
    else: 
        return predict_single(x, node.right) 
 
# --- Predict for full dataset 
def predict_tree(X, tree): 
    return np.array([predict_single(x, tree) for x in X]) 
 
# --- Build random forest 
def build_forest(X, y, n_trees, max_depth, max_features): 
    forest = [] 
    n_samples = X.shape[0] 
    for _ in range(n_trees): 
        indices = np.random.choice(n_samples, n_samples, replace=True) 
        X_sample, y_sample = X[indices], y[indices] 
        tree = build_tree(X_sample, y_sample, max_depth=max_depth, 
max_features=max_features) 
        forest.append(tree) 
    return forest 
 
# --- Predict using forest 
def predict_forest(X, forest): 
    all_preds = np.array([predict_tree(X, tree) for tree in forest]) 
    # Majority voting 
    final_preds = [] 
    for i in range(X.shape[0]): 
        votes, counts = np.unique(all_preds[:, i], return_counts=True) 
        final_preds.append(votes[np.argmax(counts)]) 
    return np.array(final_preds) 
 
def Visualize(X,y,forest):  
# -------------------- Visualize ----------------------------  
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1  
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1  
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),  
                         np.arange(y_min, y_max, 0.05))  
    grid_points = np.c_[xx.ravel(), yy.ravel()]  
    Z = predict_forest(grid_points, forest).reshape(xx.shape)  
 
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')  
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k', s=15)  
    plt.title("NumPy Random Forest Decision Boundary")  
    plt.xlabel("Feature 1")  
    plt.ylabel("Feature 2")  
    plt.grid(True)  