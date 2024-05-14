import subprocess
from hdfs.client import Client, InsecureClient
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.functions import col, mean, avg, when, cast, translate, lit, abs, element_at
import json
import csv
from flask import Flask, request, jsonify, render_template, Blueprint
from flask_cors import CORS
from pyspark.sql import functions as F
from pyspark.ml.regression import LinearRegression, LinearRegressionModel
from pyspark.ml.regression import RandomForestRegressor, RandomForestRegressionModel
from pyspark.ml.regression import GBTRegressor, GBTRegressionModel
import pandas as pd
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import DataFrame
from sklearn.utils import shuffle
from pyspark.ml.evaluation import RegressionEvaluator
from time import sleep
import os
import pandas as pd
import glob
import time

app = Flask(__name__)
CORS(app)

client = InsecureClient(url='http://localhost:9870', user = "zmw")

feture_cols = ["AREA_NO","HOLIDAY","ELECTRO_TYPE", "TRADE_NO",\
    "WINDSPEED","LAPSERATE","AIRPRESSURE","HUMIDITY","PRECIPITATIONRANINFALL", \
    "DLOAD0","DLOAD1","DLOAD2","DLOAD3","DLOAD4","DLOAD5","DLOAD6","DLOAD7","DLOAD8", \
    "DLOAD9","DLOAD10","DLOAD11","DLOAD12","DLOAD13","DLOAD14","DLOAD15","DLOAD16", \
    "DLOAD17","DLOAD18","DLOAD19","DLOAD20","DLOAD21","DLOAD22","DLOAD23"]

label_cols = ["LOAD0","LOAD1","LOAD2","LOAD3","LOAD4","LOAD5","LOAD6", \
    "LOAD7","LOAD8","LOAD9","LOAD10","LOAD11","LOAD12","LOAD13","LOAD14", \
    "LOAD15","LOAD16","LOAD17","LOAD18","LOAD19","LOAD20","LOAD21","LOAD22","LOAD23"]

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
def index():
    return render_template('index.html')

@app.route('/pageA')
def makedata():
    return render_template('makedata.html')

@app.route('/pageB')
def MachineLearning():
    return render_template('MachineLearning.html')

@app.route('/upload', methods=['POST'])      
def upload_fromWeb():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # # 将文件保存到 HDFS
    # with client.write('/path/to/1.txt', overwrite=True) as writer:
    #     writer.write(file.read())
    # print(file.filename)
    # delete_file(client, '/input/' + file.filename)
    upload_file(client, hdfs_path='/input', file_path=file.filename)
    return jsonify({'file_url': f'http://localhost:9870/input/{file.filename}'})
    
@app.route('/output')
def show_result1():
    page = request.args.get('page', default=1, type=int)
    per_page = 20  # 每页显示的行数
    csv_data, total_pages, csv_title = read_csv(page, per_page, file_path=hdfs_output_path)
    return render_template('showoutput.html', csv_data=csv_data, total_pages=total_pages,csv_title=csv_title)

@app.route('/Line')
def show_result2():
    page = request.args.get('page', default=1, type=int)
    per_page = 20  # 每页显示的行数
    csv_data, total_pages, csv_title = read_csv(page, per_page, file_path="/output/LineOutput.csv")
    return render_template('showline.html', csv_data=csv_data, total_pages=total_pages,csv_title=csv_title)

@app.route('/Forest')
def show_result3():
    page = request.args.get('page', default=1, type=int)
    per_page = 20  # 每页显示的行数
    csv_data, total_pages, csv_title = read_csv(page, per_page, file_path="/output/ForestOutput.csv")
    return render_template('showforest.html', csv_data=csv_data, total_pages=total_pages,csv_title=csv_title)

@app.route('/GBT')
def show_result4():
    page = request.args.get('page', default=1, type=int)
    per_page = 20  # 每页显示的行数
    csv_data, total_pages, csv_title = read_csv(page, per_page, file_path="/output/GBTOutput.csv")
    return render_template('showGBT.html', csv_data=csv_data, total_pages=total_pages,csv_title=csv_title)

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

@app.route('/makedata', methods=['POST'])
def makedata_test(file_path = hdfs_datafile_path):
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
    
    # 将本地output文件夹中的csv文件另存为到output.csv
    # 获取所有匹配 'part-00000-*.csv' 模式的文件
    csv_files = glob.glob("output/part-00000-*.csv")

    # 检查是否找到了文件
    if csv_files:
        # 假设只有一个匹配的文件，重命名它
        os.rename(csv_files[0], "output/output.csv")
    else:
        print("没有找到匹配的文件。")

    upload_file(client, "/output", "output/output.csv")
    
    return '数据处理完成，已上传至/output/output.csv'

@app.route('/LineLearn', methods=['POST'])
def LineLearn():
    spark = SparkSession.builder.appName("test").getOrCreate()
    df = spark.read.csv("hdfs://localhost:9000/output/output.csv", header=True, inferSchema=True)
    nonon_df = df.dropna(how="any")

    # 划分训练集和测试集
    train_data, _ = nonon_df.randomSplit([0.7, 0.3], seed=123)

    # 提取特征向量与结果向量
    assembler_features = VectorAssembler(inputCols=feture_cols, outputCol="features")
    data_train = assembler_features.transform(train_data)
    for label in label_cols:
        lr = LinearRegression(labelCol=label, regParam=0.15, maxIter=200)
        model = lr.fit(data_train)
        model.write().overwrite().save("LineModel/LineModel_" + str(label) + ".model")
    spark.stop()
    return '线性回归模型训练完成'
    
@app.route('/ForestLearn', methods=['POST'])
def ForestLearn():
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
    return '随机森林模型训练完成' 
    
@app.route('/GBTLearn', methods=['POST'])
def GBTLearn():
    spark = SparkSession.builder.appName("test").getOrCreate()
    df = spark.read.csv("hdfs://localhost:9000/output/output.csv", header=True, inferSchema=True)
    nonon_df = df.dropna(how="any")

    # 划分训练集和测试集
    train_data, _ = nonon_df.randomSplit([0.7, 0.3], seed=123)

    # 提取特征向量与结果向量
    assembler_features = VectorAssembler(inputCols=feture_cols, outputCol="features")
    data_train = assembler_features.transform(train_data)
    for label in label_cols:
        lr = GBTRegressor(labelCol=label)
        model = lr.fit(data_train)
        model.write().overwrite().save("GBTModel/GBTModel_" + str(label) + ".model")
        print("正在保存GBT模型...")
    spark.stop()
    return '梯度提升树模型训练完成'

@app.route('/LinePredict', methods=['POST'])
def LinePredict():
    spark = SparkSession.builder.appName("test").getOrCreate()
    df = spark.read.csv("hdfs://localhost:9000/output/output.csv", header=True, inferSchema=True)
    nonon_df = df.dropna(how="any")

    # 测试集
    _, test_data = nonon_df.randomSplit([0.7, 0.3], seed=123)
    test_data_withfeature = test_data.select("AREA_NO","HOLIDAY","ELECTRO_TYPE", "TRADE_NO","WINDSPEED",\
                                        "LAPSERATE","AIRPRESSURE","HUMIDITY","PRECIPITATIONRANINFALL")
    test_data_withfeature.coalesce(1).write.csv("LineOutput", mode='overwrite', header=True)

    # 提取特征向量与结果向量
    assembler_features = VectorAssembler(inputCols=feture_cols, outputCol="features")
    data_test = assembler_features.transform(test_data)
    precision_Line = {}
    i = 0
    total_time = 0
    for label in label_cols:
        model = LinearRegressionModel.load("LineModel/LineModel_" + str(label) + ".model")
        start_time = time.time()
        predictions = model.transform(data_test)
        predictions = predictions.withColumn(f"TLOAD{i}", F.round("prediction", 4))
        predictions = predictions.withColumn(f"TLOAD{i}_RE", F.round(
                                            F.when(F.col(f"LOAD{i}") != 0, 
                                                    abs((F.col(f"LOAD{i}") - F.col(f"TLOAD{i}"))) / abs(F.col(f"LOAD{i}")))
                                            .otherwise(abs(F.col(f"TLOAD{i}"))),4))
        evaluator = RegressionEvaluator(labelCol=label, predictionCol=f"TLOAD{i}", metricName="rmse")
        rmse = evaluator.evaluate(predictions)
        predictions_with_label = predictions.select(f"TLOAD{i}", f"TLOAD{i}_RE")
        predictions_with_label.coalesce(1).write.csv("LineOutput", mode='append', header=True)
        end_time = time.time()
        total_time += end_time - start_time
        precision_Line[label] = rmse
        i = i + 1

    # print("线性回归")
    # print(precision_Line)
    spark.stop()

    folder_path = 'LineOutput'
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

    df.to_csv('LineOutput.csv', index=False)

    upload_file(client, "/output", "LineOutput.csv")
    return f'线性回归模型预测完成，总耗时:{total_time:.2f}秒，均方根误差:{precision_Line}'

@app.route('/ForestPredict', methods=['POST'])
def ForestPredict():
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
    total_time = 0
    for label in label_cols:
        model = RandomForestRegressionModel.load("ForestModel/ForestModel_" + str(label) + ".model")
        start_time = time.time()
        predictions = model.transform(data_test)
        predictions = predictions.withColumn(f"TLOAD{i}", F.round("prediction", 4))
        predictions = predictions.withColumn(f"TLOAD{i}_RE", F.round(
                                            F.when(F.col(f"LOAD{i}") != 0, 
                                                    abs((F.col(f"LOAD{i}") - F.col(f"TLOAD{i}"))) / abs(F.col(f"LOAD{i}")))
                                            .otherwise(abs(F.col(f"TLOAD{i}"))),4))
        evaluator = RegressionEvaluator(labelCol=label, predictionCol=f"TLOAD{i}", metricName="rmse")
        rmse = evaluator.evaluate(predictions)
        predictions_with_label = predictions.select(f"TLOAD{i}", f"TLOAD{i}_RE")
        predictions_with_label.coalesce(1).write.csv("ForestOutput", mode='append', header=True)
        end_time = time.time()
        total_time += end_time - start_time
        precision_Forest[label] = rmse
        i = i + 1

    # print("随机森林")
    # print(precision_Forest)
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
    return f'随机森林回归模型预测完成，总耗时:{total_time:.2f}秒，均方根误差:{precision_Forest}'

@app.route('/GBTPredict', methods=['POST'])
def GBTPredict():
    spark = SparkSession.builder.appName("test").getOrCreate()
    df = spark.read.csv("hdfs://localhost:9000/output/output.csv", header=True, inferSchema=True)
    nonon_df = df.dropna(how="any")

    # 测试集
    _, test_data = nonon_df.randomSplit([0.7, 0.3], seed=123)
    test_data_withfeature = test_data.select("AREA_NO","HOLIDAY","ELECTRO_TYPE", "TRADE_NO","WINDSPEED",\
                                        "LAPSERATE","AIRPRESSURE","HUMIDITY","PRECIPITATIONRANINFALL")
    test_data_withfeature.coalesce(1).write.csv("GBTOutput", mode='overwrite', header=True)

    # 提取特征向量与结果向量
    assembler_features = VectorAssembler(inputCols=feture_cols, outputCol="features")
    data_test = assembler_features.transform(test_data)
    precision_GBT = {}
    i = 0
    total_time = 0
    for label in label_cols:
        model = GBTRegressionModel.load("GBTModel/GBTModel_" + str(label) + ".model")
        start_time = time.time()
        predictions = model.transform(data_test)
        predictions = predictions.withColumn(f"TLOAD{i}", F.round("prediction", 4))
        predictions = predictions.withColumn(f"TLOAD{i}_RE", F.round(
                                            F.when(F.col(f"LOAD{i}") != 0, 
                                                    abs((F.col(f"LOAD{i}") - F.col(f"TLOAD{i}"))) / abs(F.col(f"LOAD{i}")))
                                            .otherwise(abs(F.col(f"TLOAD{i}"))),4))
        evaluator = RegressionEvaluator(labelCol=label, predictionCol=f"TLOAD{i}", metricName="rmse")
        rmse = evaluator.evaluate(predictions)
        predictions_with_label = predictions.select(f"TLOAD{i}", f"TLOAD{i}_RE")
        predictions_with_label.coalesce(1).write.csv("GBTOutput", mode='append', header=True)
        end_time = time.time()
        total_time += end_time - start_time
        precision_GBT[label] = rmse
        i = i + 1

    # print("梯度提升树")
    # print(precision_GBT)
    spark.stop()

    folder_path = 'GBTOutput'
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

    df.to_csv('GBTOutput.csv', index=False)

    upload_file(client, "/output", "GBTOutput.csv")
    return f'梯度提升树回归模型预测完成，总耗时:{total_time:.2f}秒，均方根误差:{precision_GBT}'
   
if __name__ == "__main__":
    # mkdirs(client, hdfs_path)
    # upload_file(client, hdfs_path, local_datafile_path)
    # csv = read_file(hdfs_file_path)
    # csv.show()
    app.run(port=5000,debug=True)
    # makedata_test(hdfs_datafile_path)
    # delete_df(hdfs_datafile_path)