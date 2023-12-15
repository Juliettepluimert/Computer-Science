import json
from scipy import sparse
import scipy
import numpy as np
import re
import pickle
from scipy.sparse import lil_matrix
from get_bootstrap import get_bootstrap
from character_matrix import character_matrix
from signature_matrix import signature_matrix
from jaccard_similarity import jaccard_similarity
from lsh import lsh
from results import results
from matplotlib import pyplot as plt

# ------------------------------------- import data and setting values -------------------------------------
# some values
num_bootstraps = 1
jaccard_threshold = 0.5
perms = 200
bands = 8
all_brands = ["panasonic", "samsung", "sharp", "coby", "lg", "sony",
          "vizio", "dynex", "toshiba", "hp", "supersonic", "elo",
          "proscan", "westinghouse", "sunbritetv", "insignia", "haier",
          "pyle", "rca", "hisense", "hannspree", "viewsonic", "tcl",
          "contec", "nec", "naxa", "elite", "venturer", "philips",
          "open box", "seiki", "gpx", "magnavox", "hello kitty", "naxa", "sanyo",
          "sansui", "avue", "jvc", "optoma", "sceptre", "mitsubishi", "curtisyoung", "compaq",
          "upstar", "zzend", "contex", "affinity", "hiteker", "epson", "viore", "sigmac", "craig", "apple"]
evaluation = []


def get_data(file):
    with open(file) as f:
        data = json.load(f)

    data_set = []
    for key, value in data.items():
        for i in value:
            data_set.append(i)

    return data_set

data_set = get_data('TVs-all-merged.json')
tot_num_tvs = len(data_set)

for i in range(num_bootstraps):
    np.random.seed(i)
    print('bootstrap')
    print(i)
    data_info = get_bootstrap(tot_num_tvs, data_set)
    num_tvs = len(data_info)
    # print(num_tvs)
    # data_info = data_set

    # ------------------------------------- obtaining info and changing representation of data -------------------------------------
    model_ids = []
    websites = []

    for tv in range(len(data_info)):
        modelID = data_info[tv].get('modelID')
        model_ids.append(modelID)

        website = data_info[tv].get('shop')
        websites.append(website)

    # get the titles in lowercase
    titles = []
    for tv in data_info:
        title = tv['title'].lower()
        titles.append(title)

    # replace formatting of inches
    inched_titles = []
    for title in titles:
        title = title.replace('\"', 'inch')
        title = title.replace('inches', 'inch')
        title = title.replace('\'\'', 'inch')
        title = title.replace('\'', 'inch')
        inched_titles.append(title)

    # remove website as it's obtained differently
    shopped_titles = []
    for title in inched_titles:
        title = title.replace('newegg.com', '')
        title = title.replace('best buy', '')
        shopped_titles.append(title)

    # remove useless tokens
    tokened_titles = []
    for title in shopped_titles:
        title = title.replace('-', '')
        title = title.replace(')', '')
        title = title.replace('(', '')
        title = title.replace('/', '')
        title = title.replace('.0', '')
        title = title.replace("”", "")
        tokened_titles.append(title)

    # estimate brand
    est_brand = []
    for title in tokened_titles:
        brand = []
        for x in all_brands:
            if x in title:
                brand = x
        est_brand.append(brand)

    # get the specifications for all tvs
    # inches
    specs_inch = []
    for title in tokened_titles:
        split_title = title.split(" ")
        specifications = []
        for x in split_title:
            if x.__contains__('inch') and len(x) == 6:
                a = x
            else:
                a = 0
            specifications.append(a)
            specifications = list(filter(lambda num: num != 0, specifications))
            specifications = list(set(specifications))
        specs_inch.append(specifications)

    # hertz
    specs_hertz = []
    for title in tokened_titles:
        split_title = title.split(" ")
        specifications = []
        for x in split_title:
            if x.__contains__('hz'):
                b = x
            else:
                b = 0
            specifications.append(b)
            specifications = list(filter(lambda num: num != 0, specifications))
            specifications = list(set(specifications))
        specs_hertz.append(specifications)

    # resolution
    specs_res = []
    for title in tokened_titles:
        split_title = title.split(" ")
        specifications = []
        for x in split_title:
            if x.endswith('p') and x[0].isdigit():
                c = x
            else:
                c = 0
            specifications.append(c)
            specifications = list(filter(lambda num: num != 0, specifications))
            specifications = list(set(specifications))
        specs_res.append(specifications)

    # estimate the model ids
    est_modelIDs = []
    for title in tokened_titles:
        modelID = re.finditer(r'[a-zA-Z0-9]*(([0-9]+[^0-9, ]+)|([^0-9, ]+[0-9]+))[a-zA-Z0-9]*',  title)
        matches_to_append = []
        for match in modelID:
            matches = match.group()
            matches_to_append.append(matches)
            for i in matches_to_append:
                if 'inch' in i:
                    matches_to_append.pop(matches_to_append.index(i))
                if 'hz' in i:
                    matches_to_append.pop(matches_to_append.index(i))
                if i.endswith('p') and i[0].isdigit():
                    matches_to_append.pop(matches_to_append.index(i))
            for i in matches_to_append:
                if len(i) < 4:
                    # print(i)
                    matches_to_append.pop(matches_to_append.index(i))
        est_modelIDs.append(matches_to_append)

    for id in est_modelIDs:
        if len(id) > 1:
            longest = max(id, key=len)
            est_modelIDs[est_modelIDs.index(id)] = [longest]

    # get all the words that occur in the title
    title_words = []
    for title in tokened_titles:
        split_title = title.split(" ")
        for word in split_title:
            title_words.append(word)

    title_words = list(set(title_words))
    print(len(title_words))


    # ------------------------------------- constructing the similarity matrices -------------------------------------
    char_mat = character_matrix(title_words, tokened_titles)
    # print(char_mat)
    # this saves the character matrix, so we don't have to keep re-running!
    # pickle.dump(char_mat, open("save2.p", "wb"))
    # char_mat = pickle.load(open("save2.p", "rb"))

    sign_mat = signature_matrix(char_mat, perms)
    # print(sign_mat)
    # this saves the signature matrix, so we don't have to keep re-running!
    # pickle.dump(sign_mat, open("save.p", "wb"))
    # sign_mat = pickle.load(open("save.p", "rb"))

    cand_pairs = lsh(sign_mat, bands)
    # print(cand_pairs)
    # this saves the candidate pairs matrix, so we don't have to keep re-running!
    # pickle.dump(cand_pairs, open("save3.p", "wb"))
    # cand_pairs = pickle.load(open("save3.p", "rb"))

    # Manually update the candidate pairs matrix
    for i in range(num_tvs):
        print(i)
        for j in range(i + 1, num_tvs):
            # Check whether the model ids obtained from the title match
            if est_modelIDs[i] != [] and est_modelIDs[i] == est_modelIDs[j]:
                cand_pairs[i, j] = 1

    for i in range(num_tvs):
        print(i)
        for j in range(i + 1, num_tvs):
            # Check if same brand, different website, and enough similarity in title
            if (est_brand[i] != [] and est_brand[j] != []) and est_brand[i] != est_brand[j]:
                cand_pairs[i, j] = 0
            if websites[i] == websites[j]:
                cand_pairs[i, j] = 0
            if jaccard_similarity(sign_mat, i, j) < jaccard_threshold:
                cand_pairs[i, j] = 0

    for i in range(num_tvs):
        print(i)
        for j in range(i + 1, num_tvs):
            # Check whether the specifications match
            if (specs_inch[i] != [] and specs_inch[j] != []) and specs_inch[i] != specs_inch[j]:
                cand_pairs[i, j] = 0
            if (specs_hertz[i] != [] and specs_hertz[j] != []) and specs_hertz[i] != specs_hertz[j]:
                cand_pairs[i, j] = 0
            if (specs_res[i] != [] and specs_res[j] != []) and specs_res[i] != specs_res[j]:
                cand_pairs[i, j] = 0

    print("updating candidate pairs done")

    # ------------------------------------- evaluating the performance -------------------------------------
    # The actual duplicates matrix
    correct_similar = np.zeros([num_tvs, num_tvs])
    for k in range(num_tvs):
        for v in range(k + 1, num_tvs):
            if model_ids[k] == model_ids[v]:
                correct_similar[k, v] = 1
            if websites[k] == websites[v]:
                correct_similar[k, v] = 0

    res = results(correct_similar, cand_pairs)
    evaluation.append(res)
    for i in range(len(res)):
        print(res[i])

print(evaluation)

# pickle.dump(evaluation, open("save4.p", "wb"))
# evaluation = pickle.load(open("save4.p", "rb"))
mean_eval = np.mean(evaluation, axis=0)
print(mean_eval)
