import argparse
import json
import time
import pyspark


def main(train_file, test_file, model_file, output_file, n_weights, sc):

    train_rdd = sc.textFile(train_file).map(lambda line: json.loads(line))
    
    user_ratings_rdd = train_rdd.map(lambda r: (r['user_id'], (r['business_id'], r['stars']))) \
                                .groupByKey() \
                                .mapValues(lambda pairs: dict(pairs))
    user_ratings_map = user_ratings_rdd.collectAsMap()

    biz_ratings_rdd = train_rdd.map(lambda r: (r['business_id'], r['stars'])) \
                               .groupByKey() \
                               .mapValues(lambda ratings: sum(ratings) / len(list(ratings)))

    biz_avg_map = train_rdd.map(lambda r: (r['business_id'], r['stars'])) \
                           .groupByKey() \
                           .mapValues(lambda ratings: list(ratings)) \
                           .mapValues(lambda r: sum(r) / len(r)) \
                           .collectAsMap()

    all_ratings = train_rdd.map(lambda r: r['stars']).collect()
    overall_avg = sum(all_ratings) / len(all_ratings)

    user_avg_map = {}
    for uid, biz_dict in user_ratings_map.items():
        vals = list(biz_dict.values())
        user_avg_map[uid] = sum(vals) / len(vals)

    sim_dict = {}
    with open(model_file, 'r') as f:
        for line in f:
            record = json.loads(line.strip())
            b1 = record['b1']
            b2 = record['b2']
            sim_val = record['sim']
            num_co = record['num_co_rated']

            if b1 not in sim_dict:
                sim_dict[b1] = {}
            sim_dict[b1][b2] = (sim_val, num_co)
            
            if b2 not in sim_dict:
                sim_dict[b2] = {}
            sim_dict[b2][b1] = (sim_val, num_co)

    test_pairs = []
    with open(test_file, 'r') as f:
        for line in f:
            record = json.loads(line.strip())
            test_pairs.append((record['user_id'], record['business_id']))

    results = []
    for uid, bid in test_pairs:

        if uid not in user_ratings_map:

            if bid in biz_avg_map:
                pred = biz_avg_map[bid]
            else:
                pred = overall_avg
            results.append({"user_id": uid, "business_id": bid, "stars": round(pred, 4)})
            continue
        
        user_rated = user_ratings_map[uid]
        

        candidates = []
        if bid in sim_dict:
            neighbors = sim_dict[bid]
            for other_bid, (sim_val, num_co) in neighbors.items():
                if other_bid in user_rated:
                    # apply downweighting
                    weight_factor = min(1.0, num_co / 50.0)
                    adjusted_sim = sim_val * weight_factor
                    
                    if adjusted_sim > 0:
                        candidates.append((adjusted_sim, user_rated[other_bid]))
        
        if len(candidates) == 0:

            u_avg = user_avg_map.get(uid, overall_avg)
            b_avg = biz_avg_map.get(bid, overall_avg)
            pred = (u_avg + b_avg) / 2.0
            results.append({"user_id": uid, "business_id": bid, "stars": round(pred, 4)})
            continue
        

        candidates.sort(key=lambda x: -x[0])
        top_n = candidates[:n_weights]
        

        numerator = 0.0
        denominator = 0.0
        for sim_val, rating in top_n:
            numerator += sim_val * rating
            denominator += abs(sim_val)
        
        if denominator == 0:
            u_avg = user_avg_map.get(uid, overall_avg)
            b_avg = biz_avg_map.get(bid, overall_avg)
            pred = (u_avg + b_avg) / 2.0
        else:
            pred = numerator / denominator
        
        if pred < 1.0:
            pred = 1.0
        if pred > 5.0:
            pred = 5.0
        
        results.append({"user_id": uid, "business_id": bid, "stars": round(pred, 4)})

    with open(output_file, 'w') as f:
        for item in results:
            f.write(json.dumps(item) + '\n')


if __name__ == '__main__':
    start_time = time.time()
    sc_conf = pyspark.SparkConf() \
        .setAppName('hw4_predict') \
        .setMaster('local[*]') \
        .set('spark.driver.memory', '4g') \
        .set('spark.executor.memory', '4g')
    sc = pyspark.SparkContext(conf=sc_conf)
    sc.setLogLevel("OFF")

    parser = argparse.ArgumentParser(description='hw4-task2-predict')
    parser.add_argument('--train_file',   type=str, default='train_review.json')
    parser.add_argument('--test_file',    type=str, default='val_review.json')
    parser.add_argument('--model_file',   type=str, default='task2.model')
    parser.add_argument('--output_file',  type=str, default='task2.val.out')
    parser.add_argument('--time_file',    type=str, default='task2_predict.time')
    parser.add_argument('--n',            type=int, default=3)
    args = parser.parse_args()

    main(args.train_file, args.test_file, args.model_file, args.output_file, args.n, sc)
    sc.stop()

    # log time
    with open(args.time_file, 'w') as outfile:
        json.dump({'time': time.time() - start_time}, outfile)
    print('The run time is: ', (time.time() - start_time))