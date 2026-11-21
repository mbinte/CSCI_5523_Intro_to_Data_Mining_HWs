import sys
import time
import argparse
import json
import math 
from itertools import islice, combinations
import pyspark
from pyspark import SparkContext


def data_and_baskets_preprocessing(rdd, filter_threshold):

    header = rdd.first()
    df = rdd.filter(lambda x: x != header)
    df = df.map(lambda x: x.split(","))
    baskets_df = df.map(lambda x: (x[0],x[1])).groupByKey().mapValues(set)
    baskets_df = baskets_df.filter(lambda x: len(x[1]) > filter_threshold)
    totalCount = baskets_df.count()
    return baskets_df, totalCount
    

def validateCandidates(iterator, flat_candidates):

    baskets = list(iterator)
    candidate_counts = {}
    for basket in baskets:
        for candidate in flat_candidates:
            if candidate.issubset(basket):
                if candidate in candidate_counts:
                    candidate_counts[candidate] += 1
                else:
                    candidate_counts[candidate] = 1
    return candidate_counts.items()

def getCandidatesForNextSize(freq, nextSize, subThreshold, hashTable, hashSize):

    candidates = set()
    freq_list = sorted(freq)

    for i in range(len(freq_list)):
        for j in range(i+1, len(freq_list)):
            candidate = freq_list[i] | freq_list[j]
            if len(candidate) == nextSize:
                is_valid = True
                for item in candidate:
                    subset = candidate - frozenset([item])
                    if subset not in freq:
                        is_valid = False
                        break
                if is_valid and hashTable is not None:
                    idx = hash(tuple(sorted(candidate))) % hashSize
                    if hashTable[idx] < subThreshold:
                        is_valid = False
                if is_valid:
                    candidates.add(candidate)
    return candidates
def pcy(iterator, totalCount, threshold, hashSize):
    baskets = list(iterator)
    local_threshold = math.ceil(threshold * (len(baskets) / totalCount))

    single_count = {}
    for basket in baskets:
        for item in basket:
            single_count[item] = single_count.get(item, 0) + 1
    frequent_items = [item for item, count in single_count.items() if count >= local_threshold]
    
    hashTable = [0] * hashSize
    for basket in baskets:
        for pair in combinations(sorted(basket), 2):
            index = hash(pair) % hashSize
            hashTable[index] += 1
    
    bitmap = [1 if count >= local_threshold else 0 for count in hashTable]
    candidates_pairs = set()
    for pair in combinations(sorted(frequent_items), 2):
        index = hash(pair) % hashSize
        if bitmap[index]:
            candidates_pairs.add(frozenset(pair))
    
    count_pairs = {}
    for basket in baskets:
        basket_set = set(basket)
        for c in candidates_pairs:
            if c.issubset(basket_set):
                count_pairs[c] = count_pairs.get(c, 0) + 1
    freq_pairs = {c for c, count in count_pairs.items() if count >= local_threshold}
    
    all_frequent = [frozenset([item]) for item in frequent_items]
    all_frequent.extend(freq_pairs)
    current_freq = freq_pairs
    k = 3
    while current_freq:
        candidates_triplets  = getCandidatesForNextSize(current_freq, k, None, None, None)
        if not candidates_triplets: 
            break
        count_triplets = {}
        for basket in baskets:
            basket_set = set(basket)
            for c in candidates_triplets:
                if c.issubset(basket_set):
                    count_triplets[c] = count_triplets.get(c, 0) + 1
        current_freq = {c for c, count in count_triplets.items() if count >= local_threshold}
        all_frequent.extend(current_freq)
        k += 1
    return [(c, 1) for c in all_frequent]

def write_json_output(output_file, results_dict):
    with open(output_file, "w") as f:
        json.dump(results_dict, f)

def main(rdd, filter_threshold, support_threshold, outputJson, hashSize,time0):

    baskets_rdd, totalCount = data_and_baskets_preprocessing(rdd, filter_threshold)
    baskets_rdd = baskets_rdd.map(lambda x: x[1])  

    candidates_rdd = baskets_rdd.mapPartitions(
        lambda partition: pcy(partition, totalCount, support_threshold, hashSize)
    )
    all_candidates = candidates_rdd.map(lambda x: x[0]).distinct().collect()

    candidate_set = sc.broadcast(set(all_candidates))
    counts_rdd = baskets_rdd.mapPartitions(
        lambda partition: validateCandidates(partition, candidate_set.value)
    )
    global_counts = counts_rdd.reduceByKey(lambda a, b: a + b)
    frequent_itemsets = global_counts.filter(lambda x: x[1] >= support_threshold).map(lambda x: x[0]).collect()

    candidates_by_size = {}
    for c in all_candidates:
        size = len(c)
        if size not in candidates_by_size:
            candidates_by_size[size] = []
        candidates_by_size[size].append(sorted(list(c)))

    candidates_out = []
    for size in sorted(candidates_by_size.keys()):
        candidates_by_size[size].sort()
        candidates_out.append(candidates_by_size[size])

    freq_by_size = {}
    for f in frequent_itemsets:
        size = len(f)
        if size not in freq_by_size:
            freq_by_size[size] = []
        freq_by_size[size].append(sorted(list(f)))

    freq_out = []
    for size in sorted(freq_by_size.keys()):
        freq_by_size[size].sort()
        freq_out.append(freq_by_size[size])

    num_by_size = {}
    for c in all_candidates:
        size = str(len(c))
        num_by_size[size] = num_by_size.get(size, 0) + 1

    out = {
        "Candidates": candidates_out,
        "Frequent Itemsets": freq_out,
        "Num Candidates By Size": dict(sorted(num_by_size.items())),
        "Num Candidates Total": len(all_candidates),
    }

    time1 = time.time()
    out['Runtime'] = time1-time0
    with open(outputJson, 'w') as f:
        json.dump(out, f)
  
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Task 2: SON algorithm on Yelp data')
    parser.add_argument('--b', type=int, help='Filter threshold for users')
    parser.add_argument('--t', type=int, help='Support threshold for frequent itemsets')
    parser.add_argument('--input_file', type=str, help='Input file path')
    parser.add_argument('--output_file', type=str, help='Output file path')
    
    args = parser.parse_args()
    
    filter_threshold = args.b
    support_threshold = args.t
    input_file = args.input_file
    output_file = args.output_file
    
    hashSize = 30000000
    
    time0 = time.time()
    
    sc = SparkContext()
    sc.setLogLevel("ERROR")
    sc.setLogLevel("OFF")
    rdd = sc.textFile(input_file)
    main(rdd, filter_threshold, support_threshold, output_file, hashSize,time0)
    sc.stop()