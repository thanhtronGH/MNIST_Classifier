import pickle 
import RF_Lib as rf 
 
def train_random_forest(X_train, y_train, n_trees, max_depth, max_features): 
    print("Training Random Forest...") 
    forest = rf.build_forest(X_train, y_train, n_trees=n_trees, 
max_depth=max_depth, max_features=max_features) 
     
    print("Training completed.") 
    return forest 
 
def save_model(forest, filename): 
    with open(filename, "wb") as f: 
        pickle.dump(forest, f) 
    print(f"Model saved to {filename}") 
 
def load_model(filename): 
    with open(filename, "rb") as f: 
        forest = pickle.load(f) 
    print(f"Model loaded from {filename}") 
    return forest 
 
 