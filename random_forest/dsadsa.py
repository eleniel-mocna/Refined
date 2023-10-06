
if __name__ == '__main__':
    import pickle

    with open("RFC.pckl", "rb") as f:
        RFC = pickle.load(f)

    with open("justRFC.pckl", "wb") as f:
        pickle.dump(RFC.random_forest, f)