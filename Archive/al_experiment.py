import pandas as pd
import numpy as np

from tqdm import tqdm
import pickle
from joblib import Parallel, delayed

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


# Data preparation
raw_data = pd.read_csv("review.csv")
full_data = raw_data.loc[:,["(internal) id", "title", "abstract", "keywords", "label"]]
full_data["label"] = full_data[["label"]] != -1 # orginal labels are -1, 0, 1. Here, convert label 0 to 1.
full_data[["title","abstract"]] = full_data[["title","abstract"]].fillna("") # fillin missing values

# Feature extraction using tfidf
id_full = list(full_data.index)
x_full = TfidfVectorizer(max_features=500).fit_transform(full_data["title"]+full_data["abstract"])
y_full = full_data['label']


def al_experiment(x_full, y_full, workload, seed):

    rng = np.random.default_rng(seed)

    id_train = []
    id_pred = list(y_full.index)

    num_screen = int(workload * len(y_full))

#(all(y_full[id_train] == 0) or all(y_full[id_train]==1))
    while (sum(y_full[id_train]) == 0 or sum(y_full[id_train]-1) == 0) and len(id_train)< num_screen:
        id_next = rng.choice(a=id_pred)
        id_train.append(id_next)
        id_pred.remove(id_next)

    while len(id_train)< num_screen:

        clf = MultinomialNB().fit(x_full[id_train], y_full[id_train])
        predict_score = clf.predict_proba(x_full[id_pred])[:,1] 

        id_next = id_pred[np.argmax(predict_score)]

        id_train.append(id_next)
        id_pred.remove(id_next)

    return id_train, id_pred





if __name__ == '__main__':
    rand_seed_generator = np.random.default_rng(2022)
    seed_list = rand_seed_generator.integers(10000000, size=2000)
    result_al = Parallel(n_jobs=2)(delayed(al_experiment)(x_full, y_full, 1, seed) for seed in tqdm(seed_list))

    file = open('result/result_al', 'wb')
    pickle.dump(result_al, file)
    file.close()

    #al_experiment(x_full, y_full, seed=7069352)

file = open('result/result_al', 'rb')
result_loaded = pickle.load(file)
file.close()


