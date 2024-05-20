# PET synthesis from MRI

## What we want to try:
- [x] CycleGAN
- [ ] diffusion model
  
## CycleGAN

### 1.Get the data
首先我们拥有了大量的初始数据，每个初始数据文件夹都代表一个病人，比如说：
```bash
(cv) ➜  B10081264 tree
.
├── A4_B10081264_MR_DWI_Br_20201121023309054_S851227_I1375270.nii.gz
├── A4_B10081264_MR_FLAIR__DeFaced_Br_20201111192440421_S851236_I1365759.nii.gz
├── A4_B10081264_MR_Florbetapir_Br_20210214125609391_S989842_I1410362.nii.gz
├── A4_B10081264_MR_T1__GradWarp__DeFaced_Br_20201111104015448_S851230_I1364445.nii.gz
├── A4_B10081264_MR_T2_SE__DeFaced_Br_20201112103907914_S851228_I1367106.nii.gz
├── A4_B10081264_MR_T2_star_Br_20201116190317999_S851229_I1369972.nii.gz
├── A4_B10081264_MR_fMRI_rest_Br_20201124164344053_S851232_I1377521.nii.gz
├── CR4A4_B10081264_MR_Florbetapir_Br_20210214125609391_S989842_I1410362.nii.gz
├── CRA4_B10081264_MR_FLAIR__DeFaced_Br_20201111192440421_S851236_I1365759.nii.gz
├── CRA4_B10081264_MR_Florbetapir_Br_20210214125609391_S989842_I1410362.nii.gz
├── CRA4_B10081264_MR_T1__GradWarp__DeFaced_Br_20201111104015448_S851230_I1364445.nii.gz
├── CRA4_B10081264_MR_T2_SE__DeFaced_Br_20201112103907914_S851228_I1367106.nii.gz
├── CRA4_B10081264_MR_T2_star_Br_20201116190317999_S851229_I1369972.nii.gz
├── R4A4_B10081264_MR_Florbetapir_Br_20210214125609391_S989842_I1410362.nii.gz
├── RA4_B10081264_MR_FLAIR__DeFaced_Br_20201111192440421_S851236_I1365759.nii.gz
├── RA4_B10081264_MR_Florbetapir_Br_20210214125609391_S989842_I1410362.nii.gz
├── RA4_B10081264_MR_T1__GradWarp__DeFaced_Br_20201111104015448_S851230_I1364445.nii.gz
├── RA4_B10081264_MR_T2_SE__DeFaced_Br_20201112103907914_S851228_I1367106.nii.gz
├── RA4_B10081264_MR_T2_star_Br_20201116190317999_S851229_I1369972.nii.gz
├── aseg.nii.gz
├── brainmask.nii.gz
└── wmparc.nii.gz

0 directories, 22 files
```
这里面我们选取一个T1作为MRI，一个Florbetapir_Br作为PET，从原始数据集中抽取50个病人，形成新的数据集（小数据集先用来训练玩具模型）
```python
python extract_files.py
```
### 2.Data preprocess
这里我们神经网络需要图片输入，但是我们这里数据集是nii类型，所以需要转换成jpg图片
```python
python data_preprocess.py
```
### 3.Train
```python
python train.py --dataroot ./datasets/mri2pet --name mri2pet --model cycle_gan --display_id -1
```
### 4.Test
```python
python test.py --dataroot ./datasets/mri2pet --name mri2pet --model cycle_gan
```
or
```python
python test.py --dataroot ./datasets/mri2pet/testA --name mri2pet --model test --no_dropout --model_suffix _A --num_test 10
```
第一个是双向的，第二个是单向的