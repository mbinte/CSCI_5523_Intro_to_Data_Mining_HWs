import sys
import time
import argparse
import json
import math
from itertools import islice, combinations
import pyspark
from pyspark import SparkContext


def data_and_baskets_preprocessing(rdd, year_filter,case,rating_filter):

    header = rdd.first()
    df = rdd.filter(lambda x: x != header)
    df = df.map(lambda x: x.split(","))
    df = df.filter(lambda x: x[0] == str(year_filter))
    df = df.filter(lambda x: float(x[3]) > rating_filter)
    if case == 1:
        df = df.map(lambda x: (x[1],x[2]))
        baskets_df = df.groupByKey().mapValues(set)
    elif case == 2:
        df = df.map(lambda x: (x[2],x[1]))
        baskets_df = df.groupByKey().mapValues(set)

    return baskets_df

    
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

def getCandidatesForNextSize(freq, nextSize):
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
                if is_valid:
                    candidates.add(candidate)
    return candidates
def aPriori(iterator, totalCount):

    baskets = list(iterator)
    size_of_baskets = len(baskets)
    local_threshold = math.ceil(threshold * (size_of_baskets / totalCount))

    single_count = {}
    for basket in baskets:
        for item in basket:
            if item in single_count:
                single_count[item] += 1
            else:
                single_count[item] = 1
    frequent_items = []
    for item, count in single_count.items():
        if count >= local_threshold:
            frequent_items.append(item)
    frequent_items = sorted(frequent_items)

    all_frequent = []
    for item in frequent_items:
        all_frequent.append(frozenset([item]))


    current_freq = set()
    for item in frequent_items:
        current_freq.add(frozenset([item]))

    k = 2
    while True:
        if k == 2:
            candidates = set()
            for combo in combinations(frequent_items, 2):
                candidates.add(frozenset(combo))
        else:
            candidates = getCandidatesForNextSize(current_freq, k)

        if len(candidates) == 0:
            break
        candidate_counts = {}
        for basket in baskets:
            for candidate in candidates:
                if candidate.issubset(basket):
                    if candidate in candidate_counts:
                        candidate_counts[candidate] += 1
                    else:
                        candidate_counts[candidate] = 1

        current_freq = set()
        for itemset, count in candidate_counts.items():
            if count >= local_threshold:
                current_freq.add(itemset)
                all_frequent.append(itemset)

        if len(current_freq) == 0:
            break

        k = k + 1

    return iter(all_frequent)

def write_json_output(output_file, results_dict):
    with open(output_file, "w") as f:
        json.dump(results_dict, f)

def main(rdd, case, threshold, outputJson, year_filter, rating_filter, hashSize=None):
    time0 = time.time()
    baskets_rdd = data_and_baskets_preprocessing(rdd, year_filter, case, rating_filter)
    baskets_rdd = baskets_rdd.map(lambda x: x[1])
    totalCount = baskets_rdd.count()
    candidates_rdd = baskets_rdd.mapPartitions(lambda partition: aPriori(partition, totalCount))
    all_candidates = candidates_rdd.distinct().collect()
    candidate_set = set(all_candidates) 
    counts_rdd = baskets_rdd.mapPartitions(lambda partition: validateCandidates(partition, candidate_set))
    global_counts = counts_rdd.reduceByKey(lambda a, b: a + b)
    frequent_rdd = global_counts.filter(lambda x: x[1] >= threshold)
    frequent_itemsets = frequent_rdd.collect()

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

    frequent_by_size = {}
    for itemset, count in frequent_itemsets:
        size = len(itemset)
        if size not in frequent_by_size:
            frequent_by_size[size] = []
        frequent_by_size[size].append(sorted(list(itemset)))
    frequent_out = []
    for size in sorted(frequent_by_size.keys()):
        frequent_by_size[size].sort()
        frequent_out.append(frequent_by_size[size])

    num_by_size = {}
    for c in all_candidates:
        size = str(len(c))
        if size in num_by_size:
            num_by_size[size] += 1
        else:
            num_by_size[size] = 1

    out = {}
    out["Candidates"] = candidates_out
    out["Frequent Itemsets"] = frequent_out
    out["Num Candidates By Size"] = dict(sorted(num_by_size.items()))
    out["Num Candidates Total"] = len(all_candidates)

    time1 = time.time()
    out["Runtime"] = time1 - time0
    write_json_output(outputJson, out)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HW2T1')	
    parser.add_argument('--y', type=int, default=2016, help ='Filter year')
    parser.add_argument('--c', type=int, default=1, help ='case number')
    parser.add_argument('--r', type=int, default=0, help ='filter review rate')
    parser.add_argument('--t', type=int, default=10, help ='frequent threshold')
    parser.add_argument('--input_file', type=str, default='../data/small2.csv', help ='input file')
    parser.add_argument('--output_file', type=str, default='./HW2task1.json', help ='output  file')
    
    args = parser.parse_args()
    case = args.c
    threshold = args.t
    inputJson = args.input_file
    outputJson = args.output_file
    year_filter = args.y
    rating_filter = args.r
    sc = SparkContext()
    sc.setLogLevel("ERROR")
    rdd = sc.textFile(inputJson)
    main(rdd, case, threshold, outputJson, year_filter, rating_filter)
    sc.stop()