import pyspark
import json
import argparse
import time


def write_json_output(output_file, results_dict):
    with open(output_file, "w") as f:
        json.dump(results_dict, f)
        
def main(args):
    sc_conf = pyspark.SparkConf().setAppName('task3_c').setMaster('local[*]').set('spark.driver.memory','8g').set('spark.executor.memory','4g')
    sc = pyspark.SparkContext(conf=sc_conf)
    sc.setLogLevel("OFF")

    '''
    YOUR CODE HERE
    '''
    
    def get_year(date_str):
        return int(date_str[:4]) if date_str else 0
    start = time.time()
    reviews = sc.textFile(args.input_file).map(lambda s: json.loads(s))
    
    
    year_1s = reviews.map(lambda r: (get_year(r.get("date", "")), 1))
    def my_partitioner(year_key):
        return year_key % args.n_partitions

    year_1s_part = year_1s.partitionBy(args.n_partitions, my_partitioner)

    year_cnts = year_1s_part.reduceByKey(lambda a, b: a + b)

    year_flt = year_cnts.filter(lambda x: x[1] > args.n)
    
    num_parts = year_1s_part.getNumPartitions()
 
    items_per_part = year_1s_part.mapPartitions(lambda it: [sum(1 for _ in it)]).collect()
    
    
    pairs = year_flt.collect()
    elapsed = time.time() - start
    result = [[y, c] for (y, c) in pairs]
    out = {"n_partitions": num_parts, "n_items": items_per_part, "result": result, "time": elapsed}


    write_json_output(args.output_file, out)
    
    sc.stop()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A1T3_customized')
    parser.add_argument('--input_file', type=str, default='./data/review.json', help='review file')
    parser.add_argument('--output_file', type=str, default='./a1t3_customized.json', help='the output file contains your answers')
    parser.add_argument('--n', type=int, default=10000, help='the review count threshold per year')
    parser.add_argument('--n_partitions', type=int, default=10, help='the number of partitions ')
    args = parser.parse_args()
    main(args)


