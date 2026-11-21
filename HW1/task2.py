import argparse
import pyspark
import json


def top_states_with_reviews(reviews,businesses,n):
    

    business_kv = (businesses.map(lambda b: (b.get("business_id", ""), b.get("state", ""))).filter(lambda x: x[0] != "" and x[1] != "" and x[1] is not None) )

    
    reviews_kv = reviews.map(lambda r: (r.get("business_id", ""), 1)).filter(lambda x: x[0] != "")
    

    joined_kv = reviews_kv.join(business_kv)
    state_counts = (joined_kv.map(lambda x: (x[1][1], 1)).reduceByKey(lambda a, b: a + b))


    states = state_counts.takeOrdered(n, key=lambda x: (-x[1], x[0]))
    final_ans = [[st, cnt] for (st, cnt) in states]
    return final_ans

def write_json_output(output_file, results_dict):
    with open(output_file, "w") as f:
        json.dump(results_dict, f)

def main(args):
    sc_conf = pyspark.SparkConf().setAppName('task2').setMaster('local[*]').set('spark.driver.memory','8g').set('spark.executor.memory','4g')
    sc = pyspark.SparkContext(conf=sc_conf)
    sc.setLogLevel("OFF")

    reviews = sc.textFile(args.review_file).map(lambda s: json.loads(s))
    businesses = sc.textFile(args.business_file).map(lambda s: json.loads(s))
    
    results = top_states_with_reviews(reviews,businesses,args.n)
    print(results)    
    
    write_json_output(args.output_file,{"result": results})
    sc.stop()
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='A1T2')
    parser.add_argument('--review_file', type=str, default='./data/review.json', help='review file')
    parser.add_argument('--business_file', type=str, default='./data/business.json', help='business file')
    parser.add_argument('--output_file', type=str, default='./a1t2.json', help='the output file contains your answers')
    parser.add_argument('--n', type=int, default=5, help='top n states by number of reviews')
    args = parser.parse_args()
    
    main(args)

