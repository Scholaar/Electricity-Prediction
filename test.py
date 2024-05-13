import subprocess
from hdfs.client import Client, InsecureClient
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.functions import col, mean, avg, when, cast, translate
import json
import csv
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from pyspark.sql import functions as F
from pyspark.ml.regression import LinearRegression
import pandas as pd

def read_file(file_path):
    spark = SparkSession.builder.appName("test").getOrCreate()
    df = spark.read.csv("hdfs://localhost:9000" + file_path, header=True, inferSchema=True)
    df.show()
    spark.stop()
    
read_file("/output/output.csv")

import chardet

def method1():
    with open('yourfile.csv', 'rb') as f:
        result = chardet.detect(f.read())
        file_encoding = result['encoding']