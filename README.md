# Electricity-Prediction
使用Spark预测电力负荷

---

1. uploadToHDFS.py
   - upload_file：从本地上传文件到hadoop
   - delete_file：从hadoop删除文件 
   - show_result和read_csv：读取hdfs中的文件，在前端中展示
   - read_file：从hadoop读取指定文件，返回df
   - delete_file：读取文件后，去掉不想要的列，返回df
   - makedata_test：根据区域代号算出各个区域的平均值然后进行缺失值填充，将文件先保存在本地‘output’，然后再通过本地上传到hadoop，路径为/output/output.csv

2. *Machine.py：

   - train：从hadoop读取处理后的数据“output.csv”，进行训练，模型保存在*Model

   - predict：从hadoop读取处理后的数据“output.csv”，加载模型进行预测，将最终结果上传到hadoop：/output/*output.csv

     文件保存的方法：将测试集要的特征（比如天气地区等等）挑选出来，先保存，然后预测时，因为pyspark的原因一次只能预测一列，所以把每一列都保存为一个csv文件，最后，将特征和预测的所有csv文件使用pandas库组合到一起，再上传到hadoop
