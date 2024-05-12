import os
import pandas as pd
# 获取文件夹中所有csv文件的路径
folder_path = 'LineOutput'
csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]

# 根据修改日期对文件路径进行排序
csv_files.sort(key=os.path.getmtime)
print(csv_files)

# 读取第一个csv文件
df = pd.read_csv(csv_files[0])

# 读取并合并其余的csv文件
for file in csv_files[1:]:
    df_other = pd.read_csv(file)
    for col in df_other.columns:
        if col not in df.columns:
            df[col] = df_other[col]

df.to_csv('merged.csv', index=False)