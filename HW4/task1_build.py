import pyspark
import argparse
import json
import math
import time
import string
from collections import Counter


def load_stopwords(stopwords_file):
    stopword_list = []
    with open(stopwords_file, 'r') as f:
        for line in f:
            stopword_list.append(line.strip())
    return set(stopword_list)


def clean_the_text(text, stopwords_set):

    text = text.lower()
    

    for ch in string.punctuation:
        text = text.replace(ch, ' ')
    

    for ch in '0123456789':
        text = text.replace(ch, ' ')
    

    all_words = text.split()
    cleaned = []
    for w in all_words:
        if w not in stopwords_set and len(w) > 0:
            cleaned.append(w)
    
    return cleaned


def calculating_tfidf(biz_id, word_list, num_total_docs, doc_freq_dict):
    if len(word_list) == 0:
        return (biz_id, [])
    

    word_counter = Counter(word_list)
    max_count = max(word_counter.values())
    
    tfidf_scores = {}
    for word, count in word_counter.items():

        tf_val = count / max_count

        df = doc_freq_dict.get(word, 1)
        idf_val = math.log(num_total_docs / df)
        tfidf_scores[word] = tf_val * idf_val
    

    sorted_words = sorted(tfidf_scores.items(), key=lambda x: -x[1])
    top_100 = []
    for i in range(min(100, len(sorted_words))):
        top_100.append(sorted_words[i][0])
    
    return (biz_id, top_100)


def write_json_output(results, output_file):
    with open(output_file, 'w') as f:
        for item in results:
            f.write(json.dumps(item) + '\n')


def main(train_file, model_file, stopwords_file, sc):

    stopwords_set = load_stopwords(stopwords_file)
    raw_rdd = sc.textFile(train_file).map(lambda line: json.loads(line))
    user_business_rdd = raw_rdd.map(lambda r: (r['user_id'], r['business_id'])) \
                               .groupByKey() \
                               .mapValues(lambda x: list(set(x)))
    user_biz_map = user_business_rdd.collectAsMap()
    biz_text_rdd = raw_rdd.map(lambda r: (r['business_id'], r['text'])) \
                          .reduceByKey(lambda a, b: a + ' ' + b)
    stopwords_bc = sc.broadcast(stopwords_set)
    
    biz_words_rdd = biz_text_rdd.mapValues(
        lambda text: clean_the_text(text, stopwords_bc.value)
    )
    all_word_counts = biz_words_rdd.flatMap(lambda x: x[1]) \
                                   .map(lambda w: (w, 1)) \
                                   .reduceByKey(lambda a, b: a + b)
    
    total_num_words = biz_words_rdd.flatMap(lambda x: x[1]).count()
    rare_threshold = 0.0001 * total_num_words

    word_count_map = all_word_counts.collectAsMap()
    rare_words_set = set()
    for word, cnt in word_count_map.items():
        if cnt < rare_threshold:
            rare_words_set.add(word)
    
    rare_bc = sc.broadcast(rare_words_set)
    
    biz_words_filtered = biz_words_rdd.mapValues(
        lambda words: [w for w in words if w not in rare_bc.value]
    )

    num_docs = biz_words_filtered.count()

    doc_freq_rdd = biz_words_filtered.flatMap(
        lambda x: [(w, 1) for w in set(x[1])]
    ).reduceByKey(lambda a, b: a + b)
    
    doc_freq_map = doc_freq_rdd.collectAsMap()
    doc_freq_bc = sc.broadcast(doc_freq_map)

    biz_profiles_rdd = biz_words_filtered.map(
        lambda x: calculating_tfidf(x[0], x[1], num_docs, doc_freq_bc.value)
    )
    
    biz_profiles_map = biz_profiles_rdd.collectAsMap()
 
    user_profiles_map = {}
    for uid, biz_list in user_biz_map.items():
        user_word_set = set()
        for bid in biz_list:
            if bid in biz_profiles_map:
                biz_words = biz_profiles_map[bid]
                for w in biz_words:
                    user_word_set.add(w)
        user_profiles_map[uid] = list(user_word_set)
    
    output_lines = []
    for bid, words in biz_profiles_map.items():
        output_lines.append({"business": [bid, words]})
    for uid, words in user_profiles_map.items():
        output_lines.append({"user": [uid, words]})
    
    write_json_output(output_lines, model_file)


if __name__ == '__main__':
    start_time = time.time()

    sc_conf = pyspark.SparkConf() \
        .setAppName('hw4_task1') \
        .setMaster('local[*]') \
        .set('spark.driver.memory', '4g') \
        .set('spark.executor.memory', '4g')
    sc = pyspark.SparkContext(conf=sc_conf)
    sc.setLogLevel('OFF')

    parser = argparse.ArgumentParser(description='hw4-task2-build')
    parser.add_argument('--train_file', '-train_file', type=str, default='train_review_text_150k.json')
    parser.add_argument('--model_file', '-model_file', type=str, default='task1.model')
    parser.add_argument('--stopwords_file', '-stopwords_file', type=str, default='stopwords')
    parser.add_argument('--time_file', '-time_file', type=str, default='task1_build.time')
    args = parser.parse_args()

    main(args.train_file, args.model_file, args.stopwords_file, sc)
    sc.stop()

    with open(args.time_file, 'w') as f:
        json.dump({'time': time.time() - start_time}, f)
    print('Duration:', time.time() - start_time)
