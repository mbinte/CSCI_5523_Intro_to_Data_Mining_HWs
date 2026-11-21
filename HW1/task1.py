import argparse
import pyspark
import json

def preprocessing_of_texts(review_text, stopwords_set):
    word_tokens = (review_text or "").lower().split()
    punct_to_strip = "([,.!?:;])"
    clean_word_tokens = []
    for tok in word_tokens:
        tk = tok.strip(punct_to_strip)
        if not tk:
            continue
        if tk in stopwords_set:
            continue
        clean_word_tokens.append(tk)
    return clean_word_tokens


def All_ABCDEF_ans_func(reviews, stopwords_broadcast, t_y, m_l, n, i):

    def get_year(date_str):
        return int(date_str[:4]) if date_str else 0

    def get_month(date_str):
        return date_str[5:7] if date_str and len(date_str) >= 7 else "00"

    # A) 
    stars_rdd = reviews.map(lambda r: float(r.get("stars", 0.0)))
    cnt = stars_rdd.count()
    ansA = (stars_rdd.sum() / cnt) if cnt else 0.0

    # B) 
    years_rdd = reviews.map(lambda r: get_year(r.get("date", "")))
    ansB = years_rdd.filter(lambda y: y != t_y).count()

    filtered_reviews = reviews.filter(lambda r: get_year(r.get("date", "")) >= t_y)


    # C) 
    map_month = reviews.map(lambda r: (get_month(r.get("date", "")), 1)).filter(lambda x: x[0] != "00")
    month_counts = map_month.reduceByKey(lambda a, b: a + b)
    ansC = [
    [m, c]
    for (m, c) in month_counts.takeOrdered(n, key=lambda x: (-x[1], -int(x[0])))
    ]
    #D)
   
    month_wordcounts = filtered_reviews.map(
        lambda r: (
            get_month(r.get("date", "")),
            len(preprocessing_of_texts(r.get("text", ""), stopwords_broadcast.value))
        )
    )
    month_wordcounts = month_wordcounts.filter(lambda x: x[0] != "00")
    month_total_words = month_wordcounts.reduceByKey(lambda a, b: a + b)
    month_total_words_map = dict(month_total_words.collect())

    # all_months = [f"{k:02d}" for k in range(1, 13)]
    # ansD = [[m, int(month_total_words_map.get(m, 0))] for m in all_months]

    all_months = [f"{k:02d}" for k in range(1, 13)]
    ansD = [[m, int(month_total_words_map.get(m, 0))] for m in all_months if month_total_words_map.get(m, 0) > 0]

    ansD.sort(key=lambda x: (-x[1], -int(x[0])))
    

    review_word_counts = filtered_reviews.map(
        lambda r: len(preprocessing_of_texts(r.get("text", ""), stopwords_broadcast.value))
    )

    total_words = review_word_counts.sum()
    num_reviews = review_word_counts.count()

    avg_review_len = (total_words / num_reviews) if num_reviews else 0.0
       
    ansE = avg_review_len   


    # F)

    all_tokens = filtered_reviews.flatMap(
        lambda r: preprocessing_of_texts(r.get("text", ""), stopwords_broadcast.value)
    )

    tokens_len_filtered = all_tokens.filter(lambda w: len(w) > m_l)
    word_counts = tokens_len_filtered.map(lambda w: (w, 1)).reduceByKey(lambda a, b: a + b)
    top_word_pairs = word_counts.takeOrdered(i, key=lambda x: (-x[1], x[0]))  # freq desc, lexicographic tie-break
    ansF = [w for (w, _) in top_word_pairs]

    ansA = float(ansA)
    ansE = float(ansE)
    ansB = int(ansB)
    return {"A": ansA, "B": ansB, "C": ansC, "D": ansD, "E": ansE, "F": ansF}


def write_json_output(output_file, results_dict):
    with open(output_file, "w") as f:
        json.dump(results_dict, f)


def main(args):
    sc_conf = (
        pyspark.SparkConf()
        .setAppName("task1")
        .setMaster("local[*]")
        .set("spark.driver.memory", "8g")
        .set("spark.executor.memory", "4g")
    )
    sc = pyspark.SparkContext(conf=sc_conf)
    sc.setLogLevel("OFF")


    stopwords_rdd = (
        sc.textFile(args.stopwords)
        .map(lambda line: line.strip().lower())
        .filter(lambda line: line != "")
    )
    stopwords_set = set(stopwords_rdd.collect())
    bc_stop = sc.broadcast(stopwords_set)


    reviews = sc.textFile(args.input_file).map(lambda s: json.loads(s))

    results = All_ABCDEF_ans_func(reviews, bc_stop, args.t_y, args.m_l, args.n, args.i)
    print(results)

    write_json_output(args.output_file, results)
    sc.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HW1T1")
    parser.add_argument("--input_file", type=str, default="./data/review.json")
    parser.add_argument("--output_file", type=str, default="./a1t1.json")
    parser.add_argument("--stopwords", type=str, default="./data/stopwords")
    parser.add_argument("--t_y", type=int, default=2015)
    parser.add_argument("--m_l", type=int, default=3)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--i", type=int, default=10)
    args = parser.parse_args()
    main(args)