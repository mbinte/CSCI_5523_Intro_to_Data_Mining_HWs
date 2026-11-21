import argparse
import json
import time
import pyspark
import math
import itertools


def compute_pearson(ratings1, ratings2):
  
    if len(ratings1) == 0:
        return 0.0
    

    sum1 = 0.0
    sum2 = 0.0
    for i in range(len(ratings1)):
        sum1 += ratings1[i]
        sum2 += ratings2[i]
    
    mean1 = sum1 / len(ratings1)
    mean2 = sum2 / len(ratings2)
    

    numerator = 0.0
    denom1 = 0.0
    denom2 = 0.0
    
    for i in range(len(ratings1)):
        diff1 = ratings1[i] - mean1
        diff2 = ratings2[i] - mean2
        numerator += diff1 * diff2
        denom1 += diff1 * diff1
        denom2 += diff2 * diff2
    
    denom = math.sqrt(denom1) * math.sqrt(denom2)
    
    if denom == 0:
        return 0.0
    
    return numerator / denom


def main(train_file, model_file, co_rated_thr, sc):

    raw_rdd = sc.textFile(train_file).map(lambda line: json.loads(line))
    

    reviews_rdd = raw_rdd.map(lambda r: (r['user_id'], r['business_id'], r['stars']))

    all_users = reviews_rdd.map(lambda x: x[0]).distinct()
    total_users = all_users.count()
    

    biz_user_count = reviews_rdd.map(lambda x: (x[1], x[0])) \
                                .groupByKey() \
                                .mapValues(lambda users: len(set(users))) \
                                .collectAsMap()
    
    biz_user_count_bc = sc.broadcast(biz_user_count)
    total_users_bc = sc.broadcast(total_users)

    def apply_iuf(review):
        uid, bid, stars = review
        n_j = biz_user_count_bc.value.get(bid, 1)
        iuf_weight = math.log(total_users_bc.value / n_j)
        new_rating = stars * iuf_weight
        return (uid, bid, new_rating)
    
    weighted_rdd = reviews_rdd.map(apply_iuf)
    

    biz_ratings_rdd = weighted_rdd.map(lambda x: (x[1], (x[0], x[2]))) \
                                  .groupByKey() \
                                  .mapValues(lambda pairs: dict(pairs))
    
    biz_ratings_map = biz_ratings_rdd.collectAsMap()
    biz_ratings_bc = sc.broadcast(biz_ratings_map)

    user_biz_rdd = weighted_rdd.map(lambda x: (x[0], x[1])) \
                               .groupByKey() \
                               .mapValues(lambda x: list(set(x)))

    def make_pairs(user_businesses):
        uid, biz_list = user_businesses
        biz_list = sorted(biz_list)
        pairs = []
        for i in range(len(biz_list)):
            for j in range(i + 1, len(biz_list)):
                pairs.append(((biz_list[i], biz_list[j]), 1))
        return pairs

    pair_counts_rdd = user_biz_rdd.flatMap(make_pairs) \
                                  .reduceByKey(lambda a, b: a + b) \
                                  .filter(lambda x: x[1] >= co_rated_thr)
    

    def calc_sim_for_pair(pair_info):
        (b1, b2), num_co = pair_info
        
        b1_ratings = biz_ratings_bc.value.get(b1, {})
        b2_ratings = biz_ratings_bc.value.get(b2, {})
        

        co_users = []
        for u in b1_ratings:
            if u in b2_ratings:
                co_users.append(u)
        
        if len(co_users) < co_rated_thr:
            return None
        
        ratings_b1 = []
        ratings_b2 = []
        for u in co_users:
            ratings_b1.append(b1_ratings[u])
            ratings_b2.append(b2_ratings[u])
        
        sim = compute_pearson(ratings_b1, ratings_b2)
        
        if sim <= 0:
            return None
        
        return {"b1": b1, "b2": b2, "sim": round(sim, 6), "num_co_rated": len(co_users)}
    
    results_rdd = pair_counts_rdd.map(calc_sim_for_pair) \
                                 .filter(lambda x: x is not None)
    
    all_results = results_rdd.collect()
    

    with open(model_file, 'w') as f:
        for item in all_results:
            f.write(json.dumps(item) + '\n')


if __name__ == '__main__':
    start_time = time.time()
    sc_conf = pyspark.SparkConf() \
        .setAppName('hw4_build') \
        .setMaster('local[*]') \
        .set('spark.driver.memory', '4g') \
        .set('spark.executor.memory', '4g')
    sc = pyspark.SparkContext(conf=sc_conf)
    sc.setLogLevel("OFF")

    parser = argparse.ArgumentParser(description='hw4-task2-build')
    parser.add_argument('--train_file', type=str, default='train_review.json')
    parser.add_argument('--model_file', type=str, default='task2.model')
    parser.add_argument('--time_file',  type=str, default='task2_build.time')
    parser.add_argument('--m',          type=int, default=3)
    args = parser.parse_args()

    main(args.train_file, args.model_file, args.m, sc)
    sc.stop()

    # log time
    with open(args.time_file, 'w') as outfile:
        json.dump({'time': time.time() - start_time}, outfile)
    print('The run time is: ', (time.time() - start_time))