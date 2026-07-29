import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from myLibs_v2 import genPoints
import os
import glob
import cv2
import RF_Model as model
import os
from PIL import Image

# Init node
class Node:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx    
        self.threshold = threshold  
        self.left = left                  
        self.right = right                
        self.value = value                

    def is_leaf(self):
        return self.value is not None

# Calculate entropy
def calEntropy(y):
    if len(y) == 0:
        return 0
    # Sử dụng np.unique thay cho value_counts vì y là NumPy array
    _, counts = np.unique(y, return_counts=True)
    entropy = 0
    for count in counts:
        p = count / len(y)
        entropy -= p * math.log2(p)
    return entropy

# Calculate Informative gain
def calIG(X, y, current_entropy, max_features=None):
    listIG = []
    best_gain = -1
    best_feature_idx = None
    best_threshold = None
    
    n_features = X.shape[1]
    
    if max_features is None:
        feature_indices = np.arange(n_features) #
    elif max_features == 'sqrt':
        n_sub = int(math.sqrt(n_features))
        feature_indices = np.random.choice(n_features, n_sub, replace=False) 
    else:
        feature_indices = np.random.choice(n_features, min(max_features, n_features), replace=False)

    
    for prob in feature_indices:
        thresholds = np.unique(X[:, prob])
        # Convert analog data to true/false data
        for thresh in thresholds:
            left_mask = X[:, prob] <= thresh
            right_mask = X[:, prob] > thresh
            
            labelTrue = y[left_mask]
            labelFalse = y[right_mask]
            
            if len(labelTrue) == 0 or len(labelFalse) == 0:
                continue
                
            proDiTrue = len(labelTrue) / len(X)
            proDiFalse = len(labelFalse) / len(X)
            
            entropyTrue = calEntropy(labelTrue)
            entropyFalse = calEntropy(labelFalse)
            
            entropyDi = (proDiTrue * entropyTrue) + (proDiFalse * entropyFalse)
            IG = current_entropy - entropyDi
            
            if IG > best_gain:
                best_gain = IG
                best_feature_idx = prob
                best_threshold = thresh
                
    return best_feature_idx, best_threshold, best_gain

# Build tree
def build_tree(X, y, current_depth=0, max_depth=5, parent_label=None, max_features='sqrt'):
    unique_labels = np.unique(y)
    if len(y) == 0:
        return Node(value=parent_label)
    if len(unique_labels) == 1:
        return Node(value=unique_labels[0])
    if current_depth >= max_depth:
        vals, counts = np.unique(y, return_counts=True)
        return Node(value=vals[np.argmax(counts)])

    current_entropy = calEntropy(y)
    
    best_feature_idx, best_threshold, best_gain = calIG(X, y, current_entropy, max_features)

    if best_gain <= 1e-7 or best_feature_idx is None:
        vals, counts = np.unique(y, return_counts=True)
        return Node(value=vals[np.argmax(counts)])
        
    # Divide left and right data
    left_mask = X[:, best_feature_idx] <= best_threshold
    right_mask = X[:, best_feature_idx] > best_threshold
    
    X_left, y_left = X[left_mask], y[left_mask]
    X_right, y_right = X[right_mask], y[right_mask]

    left_child = build_tree(X_left, y_left, current_depth + 1, max_depth, max_features=max_features)
    right_child = build_tree(X_right, y_right, current_depth + 1, max_depth, max_features=max_features)

    return Node(feature_idx=best_feature_idx, threshold=best_threshold, left=left_child, right=right_child)

# Predict
def predict(node, sample):
    if node.is_leaf():
        return node.value
    
    if sample[node.feature_idx] <= node.threshold:
        return predict(node.left, sample)
    else:
        return predict(node.right, sample)

# Predict full tree 
def predict_tree(tree, X): 
    return np.array([predict(tree, x) for x in X]) 

def print_tree(node, depth=0):
    indent = "   " * depth
    if node.is_leaf():
        print(f"{indent}➔ Kết luận Lá: {node.value}")
    else:
        print(f"{indent}[Nếu {node.feature_name} == 1]:")
        print_tree(node.left, depth + 1)
        print(f"{indent}[Nếu {node.feature_name} == 0]:")
        print_tree(node.right, depth + 1)

# Build forest
def build_forest(X, y, n_trees, max_depth=5, max_features = None):    
    forest = [] 
    n_samples = X.shape[0]
    for _ in range(n_trees):
        indices = np.random.choice(n_samples, n_samples, replace=True) 
        X_sample, y_sample = X[indices], y[indices] 
        tree = build_tree(X_sample, y_sample, max_depth=max_depth, max_features=max_features) 
        forest.append(tree) 
    return forest
def predict_forest(forest, X):
    all_preds = np.array([predict_tree(tree, X) for tree in forest]) 
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

def load_mnist_test_data(test_dir):
    """
    Hàm load dữ liệu ảnh từ thư mục test và chuyển thành dạng NumPy array phẳng.
    Giả định cấu trúc thư mục: New_MNIST/test/0/, New_MNIST/test/1/, ...
    """
    X_test = []
    y_test = []
    
    # Kiểm tra thư mục có tồn tại không
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Không tìm thấy thư mục: {test_dir}")
        
    # Duyệt qua từng thư mục con (mỗi thư mục con là một nhãn lớp: '0', '1', '2'...)
    for label_dir in sorted(os.listdir(test_dir)):
        label_path = os.path.join(test_dir, label_dir)
        
        # Chỉ xử lý nếu đó là một thư mục
        if os.path.isdir(label_path):
            label = int(label_dir) # Chuyển tên thư mục thành nhãn số
            
            # Duyệt qua từng file ảnh trong thư mục nhãn
            for img_name in os.listdir(label_path):
                img_path = os.path.join(label_path, img_name)
                
                try:
                    # Mở ảnh và chuyển sang ảnh xám (L) để đảm bảo pixel từ 0-255
                    with Image.open(img_path).convert('L') as img:
                        # Chuyển ảnh thành mảng numpy phẳng (ví dụ 28x28 -> 784)
                        img_flatten = np.array(img).flatten()
                        
                        X_test.append(img_flatten)
                        y_test.append(label)
                except Exception as e:
                    # Bỏ qua nếu gặp file lỗi hoặc file hệ thống không phải ảnh
                    continue
                    
    return np.array(X_test), np.array(y_test)