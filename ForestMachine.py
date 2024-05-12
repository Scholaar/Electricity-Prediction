import subprocess
from hdfs.client import Client, InsecureClient
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.functions import col, mean, avg, when, cast, translate, lit, abs, element_at
import json
import csv
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from pyspark.sql import functions as F
from pyspark.ml.regression import RandomForestRegressor, RandomForestRegressionModel
import pandas as pd
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import DataFrame
from sklearn.utils import shuffle
from pyspark.ml.evaluation import RegressionEvaluator
from time import sleep
import os
import pandas as pd

client = InsecureClient(url='http://localhost:9870', user = "zmw")
# 创建一个包含选择列名的列表
feture_cols = ["AREA_NO","HOLIDAY","ELECTRO_TYPE", "TRADE_NO",\
    "WINDSPEED","LAPSERATE","AIRPRESSURE","HUMIDITY","PRECIPITATIONRANINFALL", \
    "DLOAD0","DLOAD1","DLOAD2","DLOAD3","DLOAD4","DLOAD5","DLOAD6","DLOAD7","DLOAD8", \
    "DLOAD9","DLOAD10","DLOAD11","DLOAD12","DLOAD13","DLOAD14","DLOAD15","DLOAD16", \
    "DLOAD17","DLOAD18","DLOAD19","DLOAD20","DLOAD21","DLOAD22","DLOAD23"]

label_cols = ["LOAD0","LOAD1","LOAD2","LOAD3","LOAD4","LOAD5","LOAD6", \
    "LOAD7","LOAD8","LOAD9","LOAD10","LOAD11","LOAD12","LOAD13","LOAD14", \
    "LOAD15","LOAD16","LOAD17","LOAD18","LOAD19","LOAD20","LOAD21","LOAD22","LOAD23"]

def upload_file(client, hdfs_path, file_path):
    client.upload(hdfs_path, file_path, overwrite=True)

def train():
    spark = SparkSession.builder.appName("test").getOrCreate()
    df = spark.read.csv("hdfs://localhost:9000/output/output.csv", header=True, inferSchema=True)
    nonon_df = df.dropna(how="any")

    # 划分训练集和测试集
    train_data, _ = nonon_df.randomSplit([0.7, 0.3], seed=123)

    # 提取特征向量与结果向量
    assembler_features = VectorAssembler(inputCols=feture_cols, outputCol="features")
    data_train = assembler_features.transform(train_data)
    for label in label_cols:
        lr = RandomForestRegressor(labelCol=label, numTrees=30)
        model = lr.fit(data_train)
        model.write().overwrite().save("ForestModel/ForestModel_" + str(label) + ".model")
    spark.stop()

def predict():
    spark = SparkSession.builder.appName("test").getOrCreate()
    df = spark.read.csv("hdfs://localhost:9000/output/output.csv", header=True, inferSchema=True)
    nonon_df = df.dropna(how="any")

    # 测试集
    _, test_data = nonon_df.randomSplit([0.7, 0.3], seed=123)
    test_data_withfeature = test_data.select("AREA_NO","HOLIDAY","ELECTRO_TYPE", "TRADE_NO","WINDSPEED",\
                                        "LAPSERATE","AIRPRESSURE","HUMIDITY","PRECIPITATIONRANINFALL")
    test_data_withfeature.coalesce(1).write.csv("ForestOutput", mode='overwrite', header=True)

    # 提取特征向量与结果向量
    assembler_features = VectorAssembler(inputCols=feture_cols, outputCol="features")
    data_test = assembler_features.transform(test_data)
    precision_Forest = {}
    i = 0
    for label in label_cols:
        model = RandomForestRegressionModel.load("ForestModel/ForestModel_" + str(label) + ".model")
        predictions = model.transform(data_test)
        predictions = predictions.withColumnRenamed("prediction", f"TLOAD{i}")
        predictions = predictions.withColumn(f"TLOAD{i}_RE", 
                                            F.when(F.col(f"LOAD{i}") != 0, 
                                                    abs((F.col(f"LOAD{i}") - F.col(f"TLOAD{i}"))) / abs(F.col(f"LOAD{i}")))
                                            .otherwise(abs(F.col(f"TLOAD{i}"))))
        evaluator = RegressionEvaluator(labelCol=label, predictionCol=f"TLOAD{i}", metricName="rmse")
        rmse = evaluator.evaluate(predictions)
        predictions_with_label = predictions.select(f"TLOAD{i}", f"TLOAD{i}_RE")
        predictions_with_label.coalesce(1).write.csv("ForestOutput", mode='append', header=True)
        precision_Forest[label] = rmse
        i = i + 1

    print("随机森林")
    print(precision_Forest)
    spark.stop()

    folder_path = 'ForestOutput'
    csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
    # 根据修改日期对文件路径进行排序
    csv_files.sort(key=os.path.getmtime)
    # 读取第一个csv文件
    df = pd.read_csv(csv_files[0])
    # 读取并合并其余的csv文件
    for file in csv_files[1:]:
        df_other = pd.read_csv(file)
        for col in df_other.columns:
            if col not in df.columns:
                df[col] = df_other[col]

    df.to_csv('ForestOutput.csv', index=False)

    upload_file(client, "/output", "ForestOutput.csv")
    
if __name__ == "__main__":
    while True:
        s = input("请输入操作：1.训练 2.预测 3.退出\n")
        if s == "1":
            train()
        elif s == "2":
            predict()
        else:
            break
