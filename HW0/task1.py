import argparse
import json
import os
from pyspark.sql import SparkSession


def argument_parser():
    parser = argparse.ArgumentParser(description="Task 1: JSON input -> JSON output")
    parser.add_argument("--input_file", required=True, help="Input the Path to input JSON file")
    parser.add_argument("--output_file", required=True, help="Input the Path to output JSON file")
    return parser.parse_args()


def read_jsonfile_line_by_line(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonfile_line_by_line(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def main():
    args = argument_parser()

    spark = SparkSession.builder.master("local[*]").appName("hw0").getOrCreate()
    sc = spark.sparkContext

    rdd = (
        sc.textFile(args.input_file)
        .map(lambda x: x.strip())
        .filter(lambda x: x != "")
        .map(json.loads)
        .map(lambda x: float(x["gpa"]))
    )

    count = rdd.count()
    avg_gpa = rdd.sum() / count if count else 0.0

    out_arg = args.output_file
    out_dir = out_arg if os.path.isdir(out_arg) else os.path.dirname(out_arg)
    if out_dir == "":
        out_dir = "."

    if out_dir != ".":
        os.makedirs(out_dir, exist_ok=True)

    output_path = os.path.join(out_dir, "ans_hw0.json")
    write_jsonfile_line_by_line(output_path, {"avg_gpa": float(avg_gpa)})

    spark.stop()


if __name__ == "__main__":
    main()
