import SimpleITK as sitk
import os

# 读取 NIfTI 文件
def read_niifile(niifile):
    return sitk.ReadImage(niifile)

# 检测并打印图像维度
def check_image_dimensions(input_dir):
    filenames = [f for f in os.listdir(input_dir) if f.endswith('.nii.gz') or f.endswith('.nii')]
    print(f"Found {len(filenames)} files in {input_dir}")  # Debugging statement
    
    for filename in filenames:
        nii_path = os.path.join(input_dir, filename)
        image = read_niifile(nii_path)
        
        # 获取图像维度
        dimensions = image.GetSize()
        num_dimensions = image.GetDimension()
        
        print(f"File: {filename}")
        print(f"Dimensions: {dimensions}")
        print(f"Number of Dimensions: {num_dimensions}")
        print()

# 设置文件路径
input_pet_dir = os.path.expanduser("/home/aac/PatchBased_3DCycleGAN_CT_Synthesis/datasets/mri2pet/test/A")

# 检测图像维度
check_image_dimensions(input_pet_dir)
