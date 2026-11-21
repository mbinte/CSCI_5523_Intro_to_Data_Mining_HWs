import json
import time
import random
import argparse

import pyspark
from pyspark import SparkConf, SparkContext


def preprocessing_binary_rating_matrix(sc, input_file):

    rdd = sc.textFile(input_file).map(lambda line: json.loads(line))
    user_bus_rdd = rdd.map(lambda x: (x['user_id'], x['business_id']))

    all_businesses = sorted(user_bus_rdd.map(lambda x: x[1]).distinct().collect())
    bus_to_idx = {b: i for i, b in enumerate(all_businesses)}
    M = len(all_businesses)

    bus_to_idx_bc = sc.broadcast(bus_to_idx)
    user_sets_rdd = (
        user_bus_rdd
        .map(lambda x: (x[0], bus_to_idx_bc.value[x[1]]))
        .groupByKey()
        .mapValues(set)
    )
    user_sets_dict = dict(user_sets_rdd.collect())

    return user_sets_dict, M

def computing_minhash_signatures(user_sets_dict, M, seed, n_hashes, sc):

    P = 104729

    random.seed(int(seed))
    a1 = random.randint(1, P - 1)
    b1 = random.randint(0, P - 1)
    a2 = random.randint(1, P - 1)
    b2 = random.randint(0, P - 1)

    h1_vals = [(a1 * x + b1) % P for x in range(M)]
    h2_vals = [(a2 * x + b2) % P for x in range(M)]

    h1_bc = sc.broadcast(h1_vals)
    h2_bc = sc.broadcast(h2_vals)
    M_bc = sc.broadcast(M)
    n_hashes_bc = sc.broadcast(n_hashes)

    def computing_single_signature(pair):
        user, bus_set = pair
        h1 = h1_bc.value
        h2 = h2_bc.value
        m = M_bc.value
        n = n_hashes_bc.value
        sig = []
        for i in range(n):
            min_val = float('inf')
            for x in bus_set:
                val = (h1[x] + i * h2[x]) % m
                if val < min_val:
                    min_val = val
            sig.append(min_val)
        return (user, sig)

    user_sets_rdd = sc.parallelize(list(user_sets_dict.items()))
    signatures = dict(user_sets_rdd.map(computing_single_signature).collect())

    return signatures

def band_hash_function(band_sig, band_idx):

    P = 104729
    h = band_idx
    for val in band_sig:
        h = (h * 31 + val) % P
    return h

def LSH_candidate_pairs(signatures, n_bands, n_rows):

    users = list(signatures.keys())
    candidates = set()

    for band in range(n_bands):
        start = band * n_rows
        end = start + n_rows

        buckets = {}
        for user in users:
            band_sig = tuple(signatures[user][start:end])
            bucket_key = band_hash_function(band_sig, band)
            if bucket_key not in buckets:
                buckets[bucket_key] = []
            buckets[bucket_key].append((user, band_sig))

        for bucket_entries in buckets.values():
            if len(bucket_entries) > 1:
                sig_groups = {}
                for user, bsig in bucket_entries:
                    if bsig not in sig_groups:
                        sig_groups[bsig] = []
                    sig_groups[bsig].append(user)

                for group_users in sig_groups.values():
                    if len(group_users) > 1:
                        group_users_sorted = sorted(group_users)
                        for i in range(len(group_users_sorted)):
                            for j in range(i + 1, len(group_users_sorted)):
                                candidates.add((group_users_sorted[i], group_users_sorted[j]))

    return candidates

def writing_candidates(candidates, candidate_file):
    candidate_list = sorted(candidates)
    with open(candidate_file, 'w') as f:
        for u1, u2 in candidate_list:
            f.write(json.dumps({"u1": u1, "u2": u2}) + '\n')

def jaccard_similarity_calculation(candidates, user_sets_dict, threshold):

    results = []
    for u1, u2 in candidates:
        set1 = user_sets_dict[u1]
        set2 = user_sets_dict[u2]
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        if union == 0:
            continue
        jaccard = intersection / union
        if jaccard > threshold:
            results.append({"u1": u1, "u2": u2, "sim": jaccard})

    results.sort(key=lambda x: (x["u1"], x["u2"]))
    return results

def write_json_output(results, output_file):
    with open(output_file, 'w') as f:
        for item in results:
            f.write(json.dumps(item) + '\n')

def main(input_file, candidate_file, output_file, threshold, seed, sc):
    n_bands = 50
    n_rows = 2
    n_hashes = n_bands * n_rows  

    # preprocessing binary rating matrix
    user_sets_dict, M = preprocessing_binary_rating_matrix(sc, input_file)

    # compute minhash signatures
    signatures = computing_minhash_signatures(user_sets_dict, M, seed, n_hashes, sc)

    # apply lsh to find candidate pairs
    candidates = LSH_candidate_pairs(signatures, n_bands, n_rows)
    writing_candidates(candidates, candidate_file)

    # calculate jaccard similarity
    results = jaccard_similarity_calculation(candidates, user_sets_dict, threshold)
    write_json_output(results, output_file)

if __name__ == '__main__':
    start_time = time.time()
    sc_conf = pyspark.SparkConf() \
        .setAppName('hw3') \
        .setMaster('local[*]') \
        .set('spark.driver.memory', '4g') \
        .set('spark.executor.memory', '4g')
    sc = pyspark.SparkContext(conf=sc_conf)
    sc.setLogLevel("OFF")

    parser = argparse.ArgumentParser(description='A3')
    parser.add_argument('--input_file', type=str, default='./data/tr1.json')
    parser.add_argument('--candidate_file', type=str, default='./outputs/candidate.out')
    parser.add_argument('--output_file', type=str, default='./outputs/task.out')
    parser.add_argument('--time_file', type=str, default='./outputs/task.time')
    parser.add_argument('--threshold', type=float, default=0.3)
    parser.add_argument('--seed', type=float, default=0)
    args = parser.parse_args()

    main(args.input_file, args.candidate_file, args.output_file,
         args.threshold, args.seed, sc)
    sc.stop()

    # log time
    with open(args.time_file, 'w') as outfile:
        json.dump({'time': time.time() - start_time}, outfile)
    print('The run time is: ', (time.time() - start_time))