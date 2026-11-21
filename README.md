# CSCI 5523 · Introduction to Data Mining

Spring 2026 coursework covering Spark RDDs, exploratory analysis, frequent itemsets, approximate similarity search, and recommender systems. All five homework submissions earned full credit on Gradescope.

| Homework | Topic | Entry points |
| --- | --- | --- |
| HW0 | Average GPA from JSON Lines | [task1.py](HW0/task1.py) |
| HW1 | Yelp review statistics, state-level counts, and Spark partitioning | [task1.py](HW1/task1.py), [task2.py](HW1/task2.py), [task3_default.py](HW1/task3_default.py), [task3_customized.py](HW1/task3_customized.py) |
| HW2 | SON with Apriori or PCY for mining frequent itemsets | [task1.py](HW2/task1.py), [task2.py](HW2/task2.py), [data_preprocess.py](HW2/data_preprocess.py) |
| HW3 | MinHash and locality-sensitive hashing for user similarity | [task.py](HW3/task.py) |
| HW4 | TF-IDF profiles and item-based collaborative filtering | [task1_build.py](HW4/task1_build.py), [task1_predict.py](HW4/task1_predict.py), [task2_build.py](HW4/task2_build.py), [task2_predict.py](HW4/task2_predict.py) |

## Setup

The course specifies Python 3.9.12 and Spark 3.2.1. Use JDK 11 and set `JAVA_HOME` if Java is outside your default path. Run these commands from the repository root with Python 3.9 available.

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install pyspark==3.2.1
export PYSPARK_PYTHON="$(command -v python)"
mkdir -p outputs
```

## Local datasets

Retrieve the original course datasets through the course-provided download location and place them at the paths below. The course used sampled subsets of Yelp data, so a newer public Yelp download may produce different results.

| Homework | Expected paths |
| --- | --- |
| HW0 | `HW0/gpa.json` |
| HW1 | `HW1/data/review.json`, `HW1/data/business.json`, `HW1/data/stopwords` |
| HW2 | `HW2/data/small1.csv`, `HW2/data/small2.csv`, `HW2/data/user_business.csv`. Generate `user_business.csv` from `HW2/data/business.json` and `HW2/data/review.json`. |
| HW3 | `HW3/data/tr1.json`, `HW3/data/tr2.json`, `HW3/data/tr3.json`. Use `HW3/data/tr-small.json` and `HW3/data/tr-small-gt.out` for the small reference case. |
| HW4 | `HW4/data/train_review_text_150k.json`, `HW4/data/train_review.json`, `HW4/data/val_review.json`, `HW4/data/stopwords`. |

## Run the assignments

Run from the repository root after placing the inputs and creating `outputs/`. The scripts write their results to that directory. The HW2 preprocessing command regenerates `HW2/data/user_business.csv`.

```bash
# HW0
python HW0/task1.py --input_file HW0/gpa.json --output_file outputs/ans_hw0.json

# HW1
python HW1/task1.py --input_file HW1/data/review.json --stopwords HW1/data/stopwords --output_file outputs/hw1-task1.json
python HW1/task2.py --review_file HW1/data/review.json --business_file HW1/data/business.json --output_file outputs/hw1-task2.json --n 5
python HW1/task3_default.py --input_file HW1/data/review.json --output_file outputs/hw1-default.json --n 10000
python HW1/task3_customized.py --input_file HW1/data/review.json --output_file outputs/hw1-customized.json --n 10000 --n_partitions 10

# HW2
spark-submit --master 'local[2]' HW2/task1.py --input_file HW2/data/small2.csv --output_file outputs/hw2-task1.json --c 2 --t 9 --r 0 --y 2016
(cd HW2 && python data_preprocess.py)
spark-submit --master 'local[2]' HW2/task2.py --input_file HW2/data/user_business.csv --output_file outputs/hw2-task2.json --b 10 --t 7

# HW3
python HW3/task.py --input_file HW3/data/tr-small.json --candidate_file outputs/hw3-candidates.out --output_file outputs/hw3-results.out --time_file outputs/hw3.time --threshold 0.3 --seed 0

# HW4
python HW4/task1_build.py --train_file HW4/data/train_review_text_150k.json --stopwords_file HW4/data/stopwords --model_file outputs/task1.model --time_file outputs/hw4-task1-build.time
python HW4/task1_predict.py --test_file HW4/data/val_review.json --model_file outputs/task1.model --output_file outputs/hw4-task1.out --time_file outputs/hw4-task1-predict.time
python HW4/task2_build.py --train_file HW4/data/train_review.json --model_file outputs/task2.model --time_file outputs/hw4-task2-build.time --m 3
python HW4/task2_predict.py --train_file HW4/data/train_review.json --test_file HW4/data/val_review.json --model_file outputs/task2.model --output_file outputs/hw4-task2.out --time_file outputs/hw4-task2-predict.time --n 3
```

## Repository contents

The repository includes the homework solution scripts and instructions for running them locally. Keep course datasets, handouts, reference cases, generated outputs, and trained models in your local workspace.
