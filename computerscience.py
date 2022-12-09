import json
import random
from scipy import sparse
import numpy as np
import re
from math import *
import pickle
from jarowinkler import *
from scipy.sparse import lil_matrix
from matplotlib import pyplot as plt


# Import data
f = open('TVs-all-merged.json')
data2 = json.load(f)
data_imp2 = []
for key, value in data2.items():
    for i in value:
        data_imp2.append(i)


all_brands = ["panasonic", "samsung", "sharp", "coby", "lg", "sony",
          "vizio", "dynex", "toshiba", "hp", "supersonic", "elo",
          "proscan", "westinghouse", "sunbritetv", "insignia", "haier",
          "pyle", "rca", "hisense", "hannspree", "viewsonic", "tcl",
          "contec", "nec", "naxa", "elite", "venturer", "philips",
          "open box", "seiki", "gpx", "magnavox", "hello kitty", "naxa", "sanyo",
          "sansui", "avue", "jvc", "optoma", "sceptre", "mitsubishi", "curtisyoung", "compaq",
          "upstar", "zzend", "contex", "affinity", "hiteker", "epson", "viore", "sigmac", "craig", "apple"]
inch = ['inches', '\"', '-inch', '\'\'', '\”', '\'']
# hertz = ["Hertz", "hertz", "Hz", "HZ", "-hz"]
# useless = ['-', ')', '/', '.0']
shop_list = ["newegg.com", "bestbuy.com", "amazon.com", "thenerds.net", "Newegg.com", "TheNerds.net"]

# some numbers




def get_titles(data_imp):
    titles = []
    for tv in data_imp:
        title = tv["title"]
        titles.append(title)
    return titles


def title_words(data):
    words_title = []
    for title in titles:
        split_title = title.split(" ")
        for word in split_title:
            msg2 = ''
            for w in word:
                if 65 <= ord(w) <= 90:
                    w = chr(ord(w) + 32)
                msg2 = msg2 + w
            words_title.append(msg2)

    cleaned_words = []
    for wt in words_title:
        for m in inch:
            if m in wt:
                cwt = wt.replace("m", "inch")
                cleaned_words.append(cwt)
            else:
                cleaned_words.append(wt)

    cleaned_words4 = []
    for word in cleaned_words:
        if "-" in word:
            cwt = word.split("-")
            for cw in cwt:
                cleaned_words4.append(cw)
        else:
            cleaned_words4.append(word)

    cleaned_words5 = []
    for word in cleaned_words4:
        if ".0" in word:
            cwt = word.replace(".0", "")
            cleaned_words5.append(cwt)
        else:
            cleaned_words5.append(word)

    cleaned_words6 = []
    for word in cleaned_words5:
        if '”' in word:
            cwt = word.replace("”", "")
            cleaned_words6.append(cwt)
        else:
            cleaned_words6.append(word)

    cleaned_words7 = []
    for word in cleaned_words6:
        if '\'' in word:
            cwt = word.replace("\'", "")
            cleaned_words7.append(cwt)
        else:
            cleaned_words7.append(word)

    cleaned_words8 = []
    for word in cleaned_words7:
        if ")" in word:
            cw = word.replace(")", "")
            cleaned_words8.append(cw)
        else:
            cleaned_words8.append(word)

    really_clean = []
    for word in cleaned_words8:
        if "\"" in word:
            cw = word.replace("\"", 'inch')
            really_clean.append(cw)
        else:
            really_clean.append(word)

    really_clean2 = []
    for word in really_clean:
        if "(" in word:
            cw = word.replace("(", "")
            really_clean2.append(cw)
        else:
            really_clean2.append(word)

    really_clean3 =[]
    for word in really_clean2:
        if 'diagonal' in word:
            cw = word.replace("diagonal", "")
            really_clean3.append(cw)
        else:
            really_clean3.append(word)

    cleaned_words2 = []
    for words in really_clean3:
        if re.search(r'[a-zA-Z0-9]*(([0-9]+[^0-9, ]+)|([^0-9, ]+[0-9]+))[a-zA-Z0-9]*', words):
            cleaned_words2.append(words)
    cleaned_words2 = list(set(cleaned_words2))


    return cleaned_words2


def character_matrix(title_words, titles):
    charac_matrix = sparse.lil_matrix(np.zeros((len(title_words), amount_tvs)))
    for title in titles:
        for word in title_words:  # e.g. "Newegg.com"
            if word in title:
                charac_matrix[title_words.index(word), titles.index(title)] = 1
            else:
                charac_matrix[title_words.index(word), titles.index(title)] = 0
    print("character matrix done")

    return charac_matrix


# pickle.dump(charmat, open("save2.p", "wb"))
# charmat = pickle.load(open("save2.p", "rb"))


def signature_matrix(title_words, charac_matrix, k):
    permutation = np.array([x for x in range(len(title_words))])
    large_num = 1000000
    sign_matrix = large_num * np.ones((k, amount_tvs))
    permutation_matrix = np.zeros((len(title_words), k))

    for per in range(k):  # fills the permutation matrix with permutations of 0-1331
        np.random.shuffle(permutation)
        permutation_matrix[:, per] = permutation

    for r in range(len(title_words)):
        for tv in range(amount_tvs):
            if charac_matrix[r, tv] == 1:
                for perm in range(k):
                    if permutation_matrix[r, perm] < sign_matrix[perm, tv]:
                        sign_matrix[perm, tv] = permutation_matrix[r, perm]
    print("signature matrix done")

    return sign_matrix




def neighbour_matrix(signature_matrix):
    factors = []
    for f in range(1, k + 1):
        if k % f == 0:
            factors.append(f)

    # threshold_values = []
    # for factor in factors:
    #     threshold_values.append((1 / factor) ** (1 / (k / factor)))

    # bands = int(factors[threshold_values.index(min(threshold_values, key=lambda x: abs(x - 0.7)))])
    # rows = int(k / bands)

    bands = 20
    rows = int(k / bands)

    neighb_matrix = sparse.lil_matrix(np.zeros((amount_tvs, amount_tvs)))

    buckets = dict()

    for b in range(bands):
        for p in range(amount_tvs):
            bucket = hash(int(''.join(str(int(n)) for n in signature_matrix[b * rows:b * rows + rows, p])))
            if bucket in buckets:
                buckets[bucket].append(p)
            else:
                buckets.update({bucket: [p]})

    duplicates = []

    for bucket in buckets:
        if len(buckets[bucket]) > 1:
            duplicates.append(buckets[bucket])

    for pair in duplicates:
        pair.sort()
        for i in range(len(pair) - 1):
            for j in range(i + 1, len(pair)):
                neighb_matrix[pair[i], pair[j]] = 1
    print("neighbour matrix done")

    return neighb_matrix


# pickle.dump(neighbour_matrix, open("save.p", "wb"))
# neighbour_matrix = pickle.load(open("save.p", "rb"))


def true_duplicates(data_imp):
    true_duplicates = sparse.lil_matrix(np.zeros((amount_tvs, amount_tvs)))
    for i in range(amount_tvs):
        for j in range(amount_tvs):
            if data_imp[i]["modelID"] == data_imp[j]["modelID"]:
                true_duplicates[i, j] = 1
            else:
                true_duplicates[i, j] = 0

    return true_duplicates


def model_id_title(list_of_titles):
    words_title = []
    model_id_title = []
    for title in list_of_titles:
        max = 0
        longest = ''
        for word in title.split(" "):
            if re.search(r'[a-zA-Z0-9]*(([0-9]+[^0-9, ]+)|([^0-9, ]+[0-9]+))[a-zA-Z0-9]*', word):
                if len(word) > max:
                    if word not in shop_list:
                        if word not in all_brands:
                            max = len(word)
                            longest = word
        model_id_title.append(longest)

    model_id_title2 = []
    for word in model_id_title:
        msg2 = ''
        for w in word:
            if 65 <= ord(w) <= 90:
                w = chr(ord(w) + 32)
            msg2 = msg2 + w
        model_id_title2.append(msg2)

    new_words = []
    for word in model_id_title2:
        if "-" in word:
            cwt = word.replace("-", "")
            new_words.append(cwt)
        else:
            new_words.append(word)

    return new_words


def brands(titles):
    brand_in_title = []
    for i in range(len(titles)):
        brand_tv = 'no brand'
        for word in titles[i].split(" "):
            msg2 = ''
            for w in word:
                if 65 <= ord(w) <= 90:
                    w = chr(ord(w) + 32)
                msg2 = msg2 + w
            for brand in all_brands:
                if msg2 == brand:
                    brand_tv = word
        brand_in_title.append(brand_tv)
    return brand_in_title


def hertz(titles):
    hertz_options = []
    for i in range(len(titles)):
        hz_tv = "no hertz"
        for word in titles[i].split(" "):
            if "hertz" in word:
                j = titles[i].split(" ").index(word)
                hz_tv = titles[i].split(" ")[j-1]
            if "Hz" in word:
                hz_tv = word.replace("Hz", "")
            if "hz" in word:
                hz_tv = word.replace("hz", "")
        hertz_options.append(hz_tv)
    return hertz_options


def inches(titles):
    inch_options = []
    for i in range(len(titles)):
        inch_tv = "no inch"
        for word in titles[i].split(" "):
            if "inch" in word and len("inch") < len(word):
                if "-" in word:
                    inch_tv = word.replace("-inch", "")
                else:
                    inch_tv = word.replace("inch", "")

            else:
                if "\"" in word and len("\"") < len(word):
                    inch_tv = word.replace("\"", "")
                    if "(" in inch_tv:
                        inch_tv = inch_tv.replace("(", "")

        inch_options.append(inch_tv)

    inch_options2 = []
    for io in inch_options:
        itv2 = "no inch"
        if io != "no inch":
            if "Diagonal" in io:
                io = io.replace("Diagonal", "")
            if "Class" in io:
                io = io.replace("Class", "")
            if ")" in io:
                io = io.replace(")", "")
            if "(" in io:
                io = io.replace(")", "")
            itv2 = round(float(io))
        inch_options2.append(itv2)

    return inch_options2


def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union


def tv_defenition(titles):
    def_options = []
    for i in range(len(titles)):
        deff = "no def"
        for word in titles[i].split(" "):
            if "1080p" in word:
                deff = "1080p"
            if "720p" in word:
                deff = "720p"
        def_options.append(deff)
    return def_options


def duplicate_pairs_pred(neighbour_matrix, brands, inches, hertz, model_id_title, threshold):
    duplicate_pairs_pred = []
    count = 0
    for i in range(amount_tvs - 1):
        print(i)
        for j in range(i + 1, amount_tvs):
            if neighbour_matrix[i, j] == 1:
                count += 1
                if len(model_id_title[i]) > 5 and len(model_id_title[j]) > 5:
                    if model_id_title[i] == model_id_title[j]:
                        pair = [i, j]
                        duplicate_pairs_pred.append(pair)
                else:
                    if data_imp[i]["shop"] != data_imp[j]["shop"]:
                        if brands[i] == brands[j] and brands[j] != "no brand":
                            if inches[i] == inches[j] and inches[j] != "no inch":
                                if hertz[i] == hertz[j] and hertz[j] != "no hertz":
                                    if [i, j] not in duplicate_pairs_pred:
                                        pair = [i, j]
                                        duplicate_pairs_pred.append(pair)
                    else:
                        # if inches[i] == inches[j] and inches[j] != "no inch":
                        #     if hertz[i] == hertz[j] and hertz[j] != "no hertz":
                        if jaccard_similarity(data_imp[i]["title"], data_imp[j]["title"]) > threshold:
                            if [i, j] not in duplicate_pairs_pred:
                                pair = [i, j]
                                duplicate_pairs_pred.append(pair)


    return duplicate_pairs_pred, count


def predicted_pairs_true(true_duplicates):
    duplicate_pairs_true = []
    for i in range(amount_tvs - 1):
        for j in range(i + 1, amount_tvs):
            if true_duplicates[i, j] == 1:
                dup_list2 = [i, j]
                duplicate_pairs_true.append(dup_list2)

    # print(len(duplicate_pairs_true))

    return duplicate_pairs_true


def f1_score(duplicate_pairs_pred, duplicate_pairs_true):
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    for i in range(amount_tvs - 1):
        print(i)
        for j in range(i + 1, amount_tvs):
            if not [i, j] in duplicate_pairs_true and not [i, j] in duplicate_pairs_pred:
                true_neg = true_neg + 1
            elif not [i, j] in duplicate_pairs_true and [i, j] in duplicate_pairs_pred:
                false_pos = false_pos + 1
            elif [i, j] in duplicate_pairs_true and [i, j] in duplicate_pairs_pred:
                true_pos = true_pos + 1
            elif [i, j] in duplicate_pairs_true and not [i, j] in duplicate_pairs_pred:
                false_neg = false_neg + 1


    print("true positives: ")
    print(true_pos)
    print("true negatives: ")
    print(true_neg)
    print("false negatives: ")
    print(false_neg)
    print("false positives: ")
    print(false_pos)
    pres = float(true_pos) / float(true_pos + false_pos)
    rec = float(true_pos) / float(true_pos + false_neg)

    f1 = float(2 * pres * rec / (pres + rec))
    return f1, rec


def f1_star(neighbour_matrix, duplicate_pairs_pred, duplicate_pairs_true, recall):
    # comparisons_matrix = np.triu(neighbour_matrix)
    comparisons_made = (np.count_nonzero(neighbour_matrix) - len(neighbour_matrix)) / 2

    pair_quality = float(len(duplicate_pairs_pred)/comparisons_made)
    pair_completeness = recall

    f1_star = float(2 * pair_quality * pair_completeness)/ (pair_completeness + pair_quality)
    return f1_star, comparisons_made


def bootstrap(data_imp):
    ind = range(0, len(data_imp))
    training_ind = set(list(np.random.choice(ind, len(data_imp), replace=True)))
    test_ind = set(ind).difference(training_ind)

    list_training_data =[]
    for i in training_ind:
        list_training_data.append(data_imp[i])

    list_test_data = []
    for j in test_ind:
        list_test_data.append(data_imp[j])

    return list_training_data,  list_test_data


bootstraps = 1

sum_f1 = 0
sum_f1_star = 0
sum_rec = 0
sum_comp = 0
for i in range(bootstraps):
    print(i)
    data_imp, data_test = bootstrap(data_imp2)

    amount_tvs = len(data_imp)
    # k = (amount_tvs // 3) * 2
    k = 400
    titles = get_titles(data_imp)
    title_w = title_words(data_imp)
    charmat = character_matrix(title_w, titles)
    signmat = signature_matrix(title_w, charmat, k)
    neighb_matrix = neighbour_matrix(signmat).toarray()
    true_dup = true_duplicates(data_imp)
    brands_l = brands(titles)
    hertz_l = hertz(titles)
    inches_l = inches(titles)
    defenition_l = tv_defenition(titles)
    model_id_tit = model_id_title(titles)
    duplicate_pairs_true = predicted_pairs_true(true_dup)



    # determine the jaccard similarity with the highest f1
    jac_sim = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
    f1_list = []
    f1_star_list = []

    dpp0, count0 = duplicate_pairs_pred(neighb_matrix, brands_l, inches_l, hertz_l, model_id_tit, 0.35)
    f10, rec0 = f1_score(dpp0, duplicate_pairs_true)
    f1_star0, comp0 = f1_star(neighb_matrix, dpp0, duplicate_pairs_true, rec0)
    f1_list.append(f10)
    f1_star_list.append(f1_star0)

    dpp1, count1 = duplicate_pairs_pred(neighb_matrix, brands_l, inches_l, hertz_l, model_id_tit, 0.40)
    f11, rec1 = f1_score(dpp1, duplicate_pairs_true)
    f1_star1, comp1 = f1_star(neighb_matrix, dpp1, duplicate_pairs_true, rec1)
    f1_list.append(f11)
    f1_star_list.append(f1_star1)

    dpp2, count2 = duplicate_pairs_pred(neighb_matrix, brands_l, inches_l, hertz_l, model_id_tit, 0.45)
    f12, rec2 = f1_score(dpp2, duplicate_pairs_true)
    f1_star2, comp2 = f1_star(neighb_matrix, dpp2, duplicate_pairs_true, rec2)
    f1_list.append(f12)
    f1_star_list.append(f1_star2)

    dpp3, count3 = duplicate_pairs_pred(neighb_matrix, brands_l, inches_l, hertz_l, model_id_tit, 0.50)
    f13, rec3 = f1_score(dpp3, duplicate_pairs_true)
    f1_star3, comp3 = f1_star(neighb_matrix, dpp3, duplicate_pairs_true, rec3)
    f1_list.append(f13)
    f1_star_list.append(f1_star3)

    dpp4, count4 = duplicate_pairs_pred(neighb_matrix, brands_l, inches_l, hertz_l, model_id_tit, 0.55)
    f14, rec4 = f1_score(dpp4, duplicate_pairs_true)
    f1_star4, comp4 = f1_star(neighb_matrix, dpp4, duplicate_pairs_true, rec4)
    f1_list.append(f14)
    f1_star_list.append(f1_star4)

    dpp5, count5 = duplicate_pairs_pred(neighb_matrix, brands_l, inches_l, hertz_l, model_id_tit, 0.60)
    f15, rec5 = f1_score(dpp5, duplicate_pairs_true)
    f1_star5, comp5 = f1_star(neighb_matrix, dpp5, duplicate_pairs_true, rec5)
    f1_list.append(f15)
    f1_star_list.append(f1_star5)

    dpp6, count6 = duplicate_pairs_pred(neighb_matrix, brands_l, inches_l, hertz_l, model_id_tit, 0.65)
    f16, rec6 = f1_score(dpp6, duplicate_pairs_true)
    f1_star6, comp6 = f1_star(neighb_matrix, dpp6, duplicate_pairs_true, rec6)
    f1_list.append(f16)
    f1_star_list.append(f1_star6)


    # plt.plot(jac_sim, f1_list)
    # plt.show()

    # dpp7 = duplicate_pairs_pred(neighb_matrix, brands_l, inches_l, hertz_l, model_id_tit, 0.95)
    # f17 = f1_score(dpp7, duplicate_pairs_true)
    # f1_star7 = f1_star(dpp7, duplicate_pairs_true)
    # f1_list.append(f17)
    # f1_star_list.append(f1_star7)

    print(f1_list)
    print(f1_star_list)



    max = 0
    j_sim = 0
    for k in range(len(f1_list)):
        if f1_list[k] > max:
            # max = f1_list[k]
            j_sim = jac_sim[k]

    print(j_sim)
    amount_tvs = len(data_test)
    # k = (amount_tvs // 3) * 2
    k = 400

    test_titles = get_titles(data_test)
    test_title_words = title_words(data_test)
    test_charmat = character_matrix(test_title_words, test_titles)
    test_signmat = signature_matrix(test_title_words, test_charmat, k)
    test_neighbour_matrix = neighbour_matrix(test_signmat).toarray()
    test_true_duplicates = true_duplicates(data_test)
    test_brands = brands(test_titles)
    test_hertz = hertz(test_titles)
    test_inches = inches(test_titles)
    test_defenition = tv_defenition(test_titles)
    test_model_id_title = model_id_title(test_titles)
    test_duplicate_pairs_true = predicted_pairs_true(test_true_duplicates)
    test_duplicate_pairs_pred, test_count = duplicate_pairs_pred(test_neighbour_matrix, test_brands, test_inches, test_hertz, test_model_id_title, j_sim)
    test_f1, test_rec = f1_score(test_duplicate_pairs_pred, test_duplicate_pairs_true)
    test_f1_star, test_comparisons_made = f1_star(test_neighbour_matrix, test_duplicate_pairs_pred, test_duplicate_pairs_true, test_rec)


    sum_f1 = sum_f1 + test_f1
    sum_f1_star = sum_f1_star + test_f1_star
    sum_rec = sum_rec + test_rec
    sum_comp = sum_comp + test_count

    print(test_f1)
    print(test_f1_star)
    print(test_rec)



avg_f1 = (sum_f1 / bootstraps)
print(avg_f1)
avg_f1_star = sum_f1_star / bootstraps
print(avg_f1_star)
avg_rec = sum_rec/bootstraps
print(avg_rec)
avg_comp = sum_comp/bootstraps
print(avg_comp)
