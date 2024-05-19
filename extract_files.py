import os
import shutil
import random
import re

# 定义输入路径和输出路径
input_base_path = "/mnt/d/A4/A4_NII"
output_t1_path = os.path.expanduser("~/vs_code_ware/T1")
output_pet_path = os.path.expanduser("~/vs_code_ware/PET")

# 创建输出文件夹
os.makedirs(output_t1_path, exist_ok=True)
os.makedirs(output_pet_path, exist_ok=True)

# 获取所有子文件夹
all_folders = [f.path for f in os.scandir(input_base_path) if f.is_dir()]

# 随机选择50个文件夹
selected_folders = random.sample(all_folders, 50)

# 定义匹配T1和PET文件的正则表达式
t1_pattern = re.compile(r"A4_B\d+_MR_T1__GradWarp__DeFaced_Br_\d+_S\d+_I\d+\.nii\.gz")
pet_pattern = re.compile(r"A4_B\d+_MR_Florbetapir_Br_\d+_S\d+_I\d+\.nii\.gz")

# 从每个文件夹中抽取文件
for folder in selected_folders:
    for filename in os.listdir(folder):
        if t1_pattern.match(filename):
            t1_file_path = os.path.join(folder, filename)
            shutil.copy(t1_file_path, output_t1_path)
            print(f"Copied {filename} to T1 folder")
        
        if pet_pattern.match(filename):
            pet_file_path = os.path.join(folder, filename)
            shutil.copy(pet_file_path, output_pet_path)
            print(f"Copied {filename} to PET folder")

print("file extrated")
