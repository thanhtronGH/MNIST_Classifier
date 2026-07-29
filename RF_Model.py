import pickle 
 
def save_model(forest, filename): 
    with open(filename, "wb") as f: 
        pickle.dump(forest, f) 
    print(f"Model saved to {filename}") 
 
def load_model(filename): 
    with open(filename, "rb") as f: 
        forest = pickle.load(f) 
    print(f"Model loaded from {filename}") 
    return forest 
 
 
