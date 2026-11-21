import pyspark
import argparse
import json
import math
import time


def compute_jaccard(set1, set2):
    if len(set1) == 0 or len(set2) == 0:
        return 0.0
    
    intersection_count = 0
    for w in set1:
        if w in set2:
            intersection_count += 1
    
    union_count = len(set1) + len(set2) - intersection_count
    
    if union_count == 0:
        return 0.0
    
    return intersection_count / union_count


def main(test_file, model_file, output_file, sc):

    user_profiles = {}
    biz_profiles = {}
    
    with open(model_file, 'r') as f:
        for line in f:
            record = json.loads(line.strip())
            if 'user' in record:
                uid = record['user'][0]
                words = record['user'][1]
                user_profiles[uid] = set(words)
            elif 'business' in record:
                bid = record['business'][0]
                words = record['business'][1]
                biz_profiles[bid] = set(words)
    

    test_pairs = []
    with open(test_file, 'r') as f:
        for line in f:
            record = json.loads(line.strip())
            test_pairs.append((record['user_id'], record['business_id']))
    

    results = []
    for uid, bid in test_pairs:

        if uid not in user_profiles or bid not in biz_profiles:
            continue
        
        user_set = user_profiles[uid]
        biz_set = biz_profiles[bid]
        
        sim_score = compute_jaccard(user_set, biz_set)
        
        if sim_score >= 0.01:
            results.append({
                "user_id": uid,
                "business_id": bid,
                "sim": sim_score
            })
    
    with open(output_file, 'w') as f:
        for item in results:
            f.write(json.dumps(item) + '\n')


if __name__ == '__main__':
    start_time = time.time()

    sc_conf = pyspark.SparkConf() \
        .setAppName('hw4_task1') \
        .setMaster('local[*]') \
        .set('spark.driver.memory', '4g') \
        .set('spark.executor.memory', '4g')
    sc = pyspark.SparkContext(conf=sc_conf)
    sc.setLogLevel('OFF')

    parser = argparse.ArgumentParser(description='hw4-task1-predict')
    parser.add_argument('--test_file', '-test_file', type=str, default='val_review.json')
    parser.add_argument('--model_file', '-model_file', type=str, default='task1.model')
    parser.add_argument('--output_file', '-output_file', type=str, default='task1.val.out')
    parser.add_argument('--time_file', '-time_file', type=str, default='task1_predict.time')
    args = parser.parse_args()

    main(args.test_file, args.model_file, args.output_file, sc)
    sc.stop()

    with open(args.time_file, 'w') as f:
        json.dump({'time': time.time() - start_time}, f)
    print('Duration:', time.time() - start_time)
