import SimpleITK as sitk
import os
from sklearn.model_selection import train_test_split

# 读取 NIfTI 文件
def read_niifile(niifile):
    return sitk.ReadImage(niifile)

# 保存图像
def save_image(image, output_path):
    sitk.WriteImage(image, output_path)

# 分割数据集并保存图像
def process_and_save_images(input_dir, output_train_dir, output_test_dir, test_size=0.2):
    filenames = [f for f in os.listdir(input_dir) if f.endswith('.nii.gz') or f.endswith('.nii')]
    print(f"Processing {input_dir} with {len(filenames)} files")  # Debugging statement
    
    train_files, test_files = train_test_split(filenames, test_size=test_size, random_state=42)
    print(f"Train files: {len(train_files)}, Test files: {len(test_files)}")  # Debugging statement

    os.makedirs(output_train_dir, exist_ok=True)
    os.makedirs(output_test_dir, exist_ok=True)

    for filename in train_files:
        nii_path = os.path.join(input_dir, filename)
        image = read_niifile(nii_path)
        output_path = os.path.join(output_train_dir, filename)
        save_image(image, output_path)
        print(f"Saved {filename} to {output_train_dir}")  # Debugging statement

    for filename in test_files:
        nii_path = os.path.join(input_dir, filename)
        image = read_niifile(nii_path)
        output_path = os.path.join(output_test_dir, filename)
        save_image(image, output_path)
        print(f"Saved {filename} to {output_test_dir}")  # Debugging statement

# 设置文件路径
input_t1_dir = os.path.expanduser("/home/aac/T1_50")
input_pet_dir = os.path.expanduser("/home/aac/PET_50")
output_base_dir = os.path.expanduser("/home/aac/PatchBased_3DCycleGAN_CT_Synthesis/datasets/mri2pet")

# 处理 T1 文件
process_and_save_images(
    input_t1_dir, 
    os.path.join(output_base_dir, "train/A"), 
    os.path.join(output_base_dir, "test/A")
)

# 处理 PET 文件
process_and_save_images(
    input_pet_dir, 
    os.path.join(output_base_dir, "train/B"), 
    os.path.join(output_base_dir, "test/B")
)

print("Image processing and saving completed.")