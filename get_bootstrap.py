import numpy as np

def get_bootstrap(num_tvs, data):
    train_set = []
    track = []

    for i in range(0, num_tvs):
        index = np.random.randint(0, num_tvs)
        track.append(index)

    track = list(set(track))
    for l in track:
        train_set.append(data[l])

    return train_set