import subprocess
from hdfs.client import Client, InsecureClient
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.functions import col, mean, avg, when
import json
import csv
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from pyspark.sql import functions as F
from pyspark.ml.regression import LinearRegression
import pandas as pd
hdfs_output_path = "/output/output.csv"
hdfs_datafile_path = "/input/dataset.csv"

spark = SparkSession.builder.appName("test").getOrCreate()
df_1 = spark.read.csv("hdfs://localhost:9000" + hdfs_output_path, header=True, inferSchema=True)
df_2 = spark.read.csv("hdfs://localhost:9000" + hdfs_datafile_path, header=True, inferSchema=True)

df_3 = pd.read_csv("dataset.csv")

df_1.printSchema()
df_2.printSchema()
print(df_3.dtypes)