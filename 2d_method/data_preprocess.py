import SimpleITK as sitk
import os
from sklearn.model_selection import train_test_split
import numpy as np

# 读取 NIfTI 文件
def read_niifile(niifile):
    return sitk.ReadImage(niifile)

# 保存图像
def save_image(image, output_path):
    sitk.WriteImage(image, output_path)

# 将图像数据转换为无符号 8 位字符类型
def convert_to_uint8(image):
    image = sitk.Cast(sitk.RescaleIntensity(image, 0, 255), sitk.sitkUInt8)
    return image

# 选择并保存切片为 JPG 图像
def save_slice_as_jpg(image, output_dir, base_filename):
    if image.GetDimension() == 4:
        image = image[:,:,:,0]
    z_slice = image.GetDepth() // 2
    slice_image = image[:, :, z_slice]
    slice_image = convert_to_uint8(slice_image)
    
    # Convert to 2D array
    array = sitk.GetArrayFromImage(slice_image)
    
    # Ensure array is 2D
    if array.ndim == 2:
        image_2d = sitk.GetImageFromArray(array)
        output_path = os.path.join(output_dir, base_filename + '.jpg')
        save_image(image_2d, output_path)
    else:
        print(f"Error: Slice extraction resulted in {array.ndim}D array for file {base_filename}")

# def save_slice_as_jpg(image, output_dir, base_filename):
#     array = sitk.GetArrayFromImage(image)
    
#     # Handle 4D images
#     if array.ndim == 4:
#         array = array[:, :, :, -1]  # Take the first volume if it's a 4D image

#     # Handle 3D images
#     if array.ndim == 3:
#         z_slice = array.shape[0] // 2  # Take the middle slice
#         array = array[z_slice, :, :]   # Extract the 2D slice

#     # Ensure array is 2D
#     if array.ndim == 2:
#         # Normalize and convert to unsigned char
#         array = convert_to_uint8(array)
#         image_2d = sitk.GetImageFromArray(array)
#         output_path = os.path.join(output_dir, base_filename + '.jpg')
#         save_image(image_2d, output_path)
#     else:
#         print(f"Error: Slice extraction resulted in {array.ndim}D array with shape {array.shape} for file {base_filename}")


# 分割数据集并保存图像
def process_and_save_images(input_dir, output_train_dir, output_test_dir, test_size=0.2):
    filenames = [f for f in os.listdir(input_dir) if f.endswith('.nii.gz') or f.endswith('.nii')]
    train_files, test_files = train_test_split(filenames, test_size=test_size, random_state=42)
    
    os.makedirs(output_train_dir, exist_ok=True)
    os.makedirs(output_test_dir, exist_ok=True)

    for filename in train_files:
        nii_path = os.path.join(input_dir, filename)
        image = read_niifile(nii_path)
        save_slice_as_jpg(image, output_train_dir, os.path.splitext(filename)[0])

    for filename in test_files:
        nii_path = os.path.join(input_dir, filename)
        image = read_niifile(nii_path)
        save_slice_as_jpg(image, output_test_dir, os.path.splitext(filename)[0])

# 设置文件路径
input_t1_dir = os.path.expanduser("~/vs_code_ware/T1")
input_pet_dir = os.path.expanduser("~/vs_code_ware/PET")
output_base_dir = os.path.expanduser("~/vs_code_ware/projects/info/pytorch-CycleGAN-and-pix2pix/datasets/mri2pet")

# 处理 T1 文件
process_and_save_images(
    input_t1_dir, 
    os.path.join(output_base_dir, "trainA"), 
    os.path.join(output_base_dir, "testA")
)

# 处理 PET 文件
process_and_save_images(
    input_pet_dir, 
    os.path.join(output_base_dir, "trainB"), 
    os.path.join(output_base_dir, "testB")
)

print("图像处理和保存完成。")
