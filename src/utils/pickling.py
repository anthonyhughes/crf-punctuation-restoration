import pickle


# ====================
def to_pickle(object_to_pickle, pickle_path: str):
    with open(pickle_path, 'wb') as f:
        pickle.dump(object_to_pickle, f)


# ====================
def load_pickle(pickle_path: str):
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)
