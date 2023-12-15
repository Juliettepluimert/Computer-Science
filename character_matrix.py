import numpy as np
from scipy import sparse


def character_matrix(title_words, titles):
    charac_matrix = sparse.lil_matrix(np.zeros((len(title_words), len(titles))))
    for title in titles:
        print(titles.index(title))
        for word in title_words:  # e.g. "42inch"
            if word in title:
                charac_matrix[title_words.index(word), titles.index(title)] = 1
            else:
                charac_matrix[title_words.index(word), titles.index(title)] = 0
    print("character matrix done")

    return charac_matrix