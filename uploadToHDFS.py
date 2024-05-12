# import os

# memory = '4g'
# pyspark_submit_args = ' --conf "spark.executor.extraJavaOptions=-XX:+UseG1GC -XX:MaxGCPauseMillis=200" --conf "spark.driver.extraJavaOptions=-XX:+UseG1GC -XX:MaxGCPauseMillis=200" --conf "spark.memory.offHeap.enabled=true" --conf "spark.memory.offHeap.size=2g"'+' --driver-memory ' + memory + ' --executor-memory ' + memory + ' pyspark-shell'
# os.environ["PYSPARK_SUBMIT_ARGS"] = pyspark_submit_args
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

app = Flask(__name__)
CORS(app)

client = InsecureClient(url='http://localhost:9870', user = "zmw")

local_datafile_path = "dataset.csv"     # 本地原始数据集路径
local_localfile_path = "output.csv"     # 本地处理后的数据集路径
hdfs_datafile_path = "/input/dataset.csv"       # HDFS原始数据集路径
hdfs_output_path = "/output/output.csv"     # HDFS数据处理后的数据集路径
hdfs_path = "/output"     # HDFS文件夹路径

def exists(client, hdfs_path):
    client.status(hdfs_path)

def upload_file(client, hdfs_path, file_path):
    client.upload(hdfs_path, file_path, overwrite=True)
    
def delete_file(client, file_path):
    client.delete(file_path)
    
#  数据可视化部分
    
@app.route('/')
def show_result():
    page = request.args.get('page', default=1, type=int)
    per_page = 20  # 每页显示的行数
    csv_data, total_pages, csv_title = read_csv(page, per_page, hdfs_output_path)
    return render_template('index.html', csv_data=csv_data, total_pages=total_pages,csv_title=csv_title)

def read_csv(page, per_page, file_path):
    start_index = (page - 1) * per_page
    end_index = start_index + per_page
    # 从第二行开始读取数据
    with client.read(file_path, encoding='utf-8') as reader:
        csv_data = list(csv.reader(reader))
    csv_title = csv_data[0]
    csv_data = csv_data[1:]
    total_rows = len(csv_data) 
    total_pages = total_rows // per_page + (1 if total_rows % per_page != 0 else 0)
        
    return csv_data[start_index:end_index], total_pages, csv_title

# 数据处理部分

def read_file(file_path):
    spark = SparkSession.builder.appName("test").getOrCreate()
    df = spark.read.csv("hdfs://localhost:9000" + file_path, header=True, inferSchema=True)
    return df

def delete_df(file_path):
    df = read_file(file_path)
    # 要去除的列名
    cols_to_remove = ['MP_ID','STAT_CYCLE','MLOAD0','MLOAD1','MLOAD2','MLOAD3','MLOAD4','MLOAD5','MLOAD6','MLOAD7','MLOAD8','MLOAD9','MLOAD10','MLOAD11','MLOAD12','MLOAD13','MLOAD14','MLOAD15','MLOAD16','MLOAD17','MLOAD18','MLOAD19','MLOAD20','MLOAD21','MLOAD22','MLOAD23','WLOAD0','WLOAD1','WLOAD2','WLOAD3','WLOAD4','WLOAD5','WLOAD6','WLOAD7','WLOAD8','WLOAD9','WLOAD10','WLOAD11','WLOAD12','WLOAD13','WLOAD14','WLOAD15','WLOAD16','WLOAD17','WLOAD18','WLOAD19','WLOAD20','WLOAD21','WLOAD22','WLOAD23']
    df_without_removed_columns = df.select([col for col in df.columns if col not in cols_to_remove])
    return df_without_removed_columns

def makedata_test(file_path):
    df = delete_df(file_path)
    # 计算缺失值较多的列
    # df.agg(*[(1- (function.count(c)/ function.count('*'))).alias(c + 'missing') for c in df.columns]).show()
    # 删除缺失值较多的列
    # df = df.drop('column_name')
    cols_to_avg = ['DLOAD0','DLOAD1','DLOAD2','DLOAD3','DLOAD4','DLOAD5','DLOAD6','DLOAD7','DLOAD8','DLOAD9','DLOAD10','DLOAD11'
                   ,'DLOAD12','DLOAD13','DLOAD14','DLOAD15','DLOAD16','DLOAD17','DLOAD18','DLOAD19','DLOAD20','DLOAD21','DLOAD22','DLOAD23'
                   ,'LOAD0','LOAD1','LOAD2','LOAD3','LOAD4','LOAD5','LOAD6','LOAD7','LOAD8','LOAD9','LOAD10','LOAD11'
                   ,'LOAD12','LOAD13','LOAD14','LOAD15','LOAD16','LOAD17','LOAD18','LOAD19','LOAD20','LOAD21','LOAD22','LOAD23']
    # 一次性计算所有列的平均值
    avg_cols = {c: 'avg' for c in cols_to_avg}
    df_avg = df.groupBy('AREA_NO').agg(avg_cols)

    df = df.join(df_avg, "AREA_NO", "left")
    for col in cols_to_avg:
        avg_col_name = 'avg(' + col + ')'
        df = df.withColumn(col,\
            F.when(df[col].isNull(), df[avg_col_name]).otherwise(df[col])
        ).drop(avg_col_name)

    # 将TRADE_NO列为“31B0”的字符串转化为“3000”
    df = df.withColumn("TRADE_NO", F.when(df["TRADE_NO"] == "31B0", "3000").otherwise(df["TRADE_NO"]))
    df = df.withColumn("TRADE_NO", df["TRADE_NO"].cast("int"))
    df.coalesce(1).write.csv("output", mode='overwrite', header=True)

    delete_file(client, "/output/output.csv")
    upload_file(client, "/output", "output.csv")
   
if __name__ == "__main__":
    # mkdirs(client, hdfs_path)
    upload_file(client, hdfs_path, local_datafile_path)
    # csv = read_file(hdfs_file_path)
    # csv.show()
    # app.run(port=5000,debug=True)
    # makedata_test(hdfs_datafile_path)
    # delete_df(hdfs_datafile_path)