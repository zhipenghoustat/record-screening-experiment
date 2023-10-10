import numpy as np
import math
from tqdm import tqdm
import pickle 
import pandas as pd
#import pdb
#pdb.set_trace()

#from joblib import Parallel, delayed

rand_seed_generator = np.random.default_rng(2022)

def calculate_sample_size(R, c):
    k_s = int(math.log(1-R, c)) + 1
    k_t = round(math.log(1-R)/(c-1))
    diff = k_s-k_t
    return R, c, k_s, k_t, diff

def sequence_gen_random(N, prevalence, seed):
    rng = np.random.default_rng(seed)

    num_relevant = int(N * prevalence)
    num_irrelevant = N - num_relevant
    y_full = [1] * num_relevant + [0] * num_irrelevant

    id_ranked = np.arange(len(y_full))

    id_ranked = rng.permuted(id_ranked)

    return np.array(y_full), id_ranked

def sequence_gen_simulated(N, prevalence, seed):
    rng = np.random.default_rng(seed)

    num_relevant = int(N * prevalence)
    num_irrelevant = N - num_relevant
    y_full = [1] * num_relevant + [0] * num_irrelevant

    mean_list = np.multiply(y_full, 1.5)
    sd_list = (-np.array(y_full) + 2)/2

    priority_score = rng.normal(loc=mean_list, scale=sd_list)

    id_ranked = np.argsort(-priority_score)

    return np.array(y_full), id_ranked

def sequence_gen_real():
    # Retrieve y_full
    raw_data = pd.read_csv("review.csv")
    raw_data["label"] = raw_data[["label"]] != -1
    y_full = np.array(raw_data['label'])

    # Retrieve ranked id.
    file = open('result/result_al', 'rb')
    result_loaded = pickle.load(file)
    file.close()

    for res in result_loaded:
        yield y_full, np.array(res[0])

def record_sampling(y_full, k, seed):
    rng = np.random.default_rng(seed)

    record_id = np.arange(len(y_full))
 
    sample_list = []
    relevant_sample_list = []

    while len(relevant_sample_list) < k and len(set(sample_list)) < len(y_full):
        next = rng.choice(record_id)
        sample_list.append(next)
        if y_full[next] ==1:
            relevant_sample_list.append(next)
    return np.array(sample_list)



def result_analysis(y_full, id_ranked, sample_list, c):
    num_record = len(y_full)
    num_relevant = np.sum(y_full)
    # 1. Find the highest rank of sample relevant records

    relevant_sample_list = sample_list[y_full[sample_list]==1]
    relevant_sample_rank = np.where(np.isin(id_ranked, relevant_sample_list))
    max_rank = np.max(relevant_sample_rank)

    # 2. Find the list of screened sample id.
    screened_record_list = id_ranked[:max_rank+1]

    # 3. calculate recall
    num_identified = np.sum(y_full[screened_record_list])
    recall = num_identified/num_relevant

    # 4. sample workload
    sample_workload =  len(set(sample_list))/num_record
    # 5. screen workload
    screen_workload = len(set(screened_record_list).difference(set(sample_list)))/num_record
    # 6. total workload
    workload = sample_workload + screen_workload

    # 7. workload to achieve c recall
    relevant_record_ranked = id_ranked[y_full[id_ranked] ==1]
    L = sum(y_full)
    L_c = int(L*c) + 1
    relevant_record_rel_rank_c = relevant_record_ranked[L_c-1]
    relevant_record_rel_rank_c_index = np.where(id_ranked ==relevant_record_rel_rank_c)[0][0]
    workload_c = (relevant_record_rel_rank_c_index + 1)/num_record
    return recall, workload, sample_workload, screen_workload, workload_c


def experiment(N, prevalence, R, c, seed, round, mode):
    rng = np.random.default_rng(seed)
    _,_,k,_,_ = calculate_sample_size(R, c)

    res_list = []
    if mode in ["sim", "random"]:
        for _ in tqdm(range(round)): 
            seed_1 = rng.integers(10000000, size=1)
            if mode ==  "sim":
                y_full, id_ranked = sequence_gen_simulated(N, prevalence, seed=seed_1) 
            elif mode == "random":
                y_full, id_ranked = sequence_gen_random(N, prevalence, seed=seed_1) 
            seed_2 = rng.integers(10000000, size=1)
            sample_list = record_sampling(y_full, k=k, seed=seed_2) 

            res = result_analysis(y_full, id_ranked, sample_list, c)
            res_list.append(res)

    elif mode == "real":
        seq_generator = sequence_gen_real()

        for y_full, id_ranked in tqdm(seq_generator): 

            seed = rng.integers(10000000, size=1)
            sample_list = record_sampling(y_full, k=k, seed=seed) 

            res = result_analysis(y_full, id_ranked, sample_list, c)
            res_list.append(res)
            round = len(res_list)


    res_array = np.array(res_list)
    recall_list = res_array[:,0]

    reliability = np.sum(np.array(recall_list) > c)/round
    reliability_sd = np.sqrt(reliability*(1- reliability)/round)

    workload_list =  res_array[:,1]
    workload_mean = np.mean(workload_list)
    workload_sd = np.std(workload_list)
    sample_workload_list =  res_array[:,2]
    screen_workload_list = res_array[:,3]
    workload_c_list = res_array[:,4]

    return reliability, reliability_sd, workload_mean, workload_sd, recall_list, sample_workload_list, screen_workload_list, workload_c_list, k

rand_seed_generator = np.random.default_rng(2022)

#exp_result_sim = experiment(N=1311, prevalence=1/128, R=0.9, c=0.9, seed=67, round = 100, mode = "sim")
#exp_result_rand = experiment(N=13110, prevalence=1/128, R=0.9, c=0.9, seed=67, round = 1000, mode = "random")
#exp_result_real = experiment(N=None, prevalence=None, R=0.9, c=0.9, seed=67, round = None, mode = "real")

R_list = [0.75, 0.9]
c_list = [0.75, 0.9]
N_list = [4096, 32768,262144]
prevalence_list = [1/128, 1/32, 1/8]

res = {}

for R_val in R_list:
    for c_val in c_list:
        for N_val in N_list:
            for prevalence in prevalence_list:
                L_val = int(N_val * prevalence)
                seed = rand_seed_generator.integers(10000000, size=1)
                result_name = "R"+str(R_val)+"c"+str(c_val)+"N"+str(N_val)+"L"+str(L_val)
                exp_result = experiment(N_val, prevalence, R_val, c_val, seed=seed, round = 10000, mode = "sim")
                res[result_name] = exp_result[:4]
                print(result_name)
                print(exp_result[:4])
file = open('result/result_new', 'wb')
pickle.dump(res, file)
file.close()


rand_seed_generator = np.random.default_rng(2032)
res_real = {}
for R_val in R_list:
    for c_val in c_list:
        result_name = "R"+str(R_val)+"c"+str(c_val)
        seed = rand_seed_generator.integers(10000000, size=1)
        exp_result_real = experiment(N=None, prevalence=None, R=R_val, c=c_val, seed=seed, round = None, mode = "real")
        res_real[result_name] = exp_result_real
        print(result_name)
        print(exp_result_real[:4])
file = open('result/result_real', 'wb')
pickle.dump(res_real, file)
file.close()