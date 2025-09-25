'''
医学图像处理
'''
import os
import h5py
import numpy as np
from tqdm import tqdm
from skimage.io import imread

def get_img_file(file_name):
    """
    获取指定目录下所有图片文件的路径
    参数:
        file_name: 目录路径
    返回:
        imagelist: 包含所有图片文件完整路径的列表
    """
    imagelist = []
    for parent, dirnames, filenames in os.walk(file_name):
        for filename in filenames:
            # 检查文件是否为支持的图片格式
            if filename.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff', '.npy')):
                imagelist.append(os.path.join(parent, filename))
    return imagelist
    
def rgb2y(img):
    """
    将RGB图像转换为YCbCr中的Y通道（亮度）
    参数:
        img: RGB图像数组 [3, H, W]
    返回:
        y: 亮度通道 [1, H, W]
    """
    # 彩色图to亮度的计算公式
    y = img[0:1, :, :] * 0.299000 + img[1:2, :, :] * 0.587000 + img[2:3, :, :] * 0.114000
    return y

def Im2Patch(img, win, stride=1):
    """
    将图像分割成多个小块（patches）
    输入c,h,w 输出 c,win,win,num
    参数:
        img: 输入图像 [C, H, W]
        win: patch的大小
        stride: 滑动步长
    返回:
        patches数组 [C, win, win, num_patches]
    """
    c, h, w = img.shape
    num_h = (h - win) // stride + 1
    num_w = (w - win) // stride + 1
    
    # 预分配输出数组
    patches = np.zeros((c, win, win, num_h * num_w), dtype=img.dtype)
    
    k = 0
    for i in range(num_h):
        for j in range(num_w):
            h_start = i * stride
            h_end = h_start + win
            w_start = j * stride
            w_end = w_start + win
            
            patches[:, :, :, k] = img[:, h_start:h_end, w_start:w_end]
            k += 1
            
    return patches

def is_low_contrast(image, fraction_threshold=0.1, lower_percentile=10,
                    upper_percentile=90):
    """
    判断图像是否为低对比度
    参数:
        image: 输入图像
        fraction_threshold: 对比度阈值
        lower_percentile: 低百分位数
        upper_percentile: 高百分位数
    返回:
        布尔值，True表示低对比度
    """
    limits = np.percentile(image, [lower_percentile, upper_percentile])
    ratio = (limits[1] - limits[0]) / limits[1]
    return ratio < fraction_threshold


# 设置数据处理参数
data_name="MIF_train"  # 数据集名称
img_size=128   # patch大小
stride=128     # patch提取的步长

# 创建保存处理后数据的目录
os.makedirs('data', exist_ok=True)

# 获取MRI和PET图像文件列表
# IR_files是一个路径列表 sorted()函数是排序函数
MRI_files = sorted(get_img_file(r"MSRS_train/mri"))
PET_files = sorted(get_img_file(r"MSRS_train/pet"))

# 确保红外和可见光图像数量相同
assert len(MRI_files) == len(PET_files)

# 创建HDF5文件用于存储处理后的数据
h5f = h5py.File(os.path.join('.\\data',
                                 data_name+'_imgsize_'+str(img_size)+"_stride_"+str(stride)+'.h5'), 
                    'w')
h5_ir = h5f.create_group('mri_patchs')    # MRI图像patches组
h5_vis = h5f.create_group('pet_patchs')  # PET图像patches组
train_num=0  # 用于记录处理后的patch数量

'''
(3, 256, 256)
(1, 256, 256)
'''

# 遍历所有图像对
# tqdm是用来显示程序进度条的函数
for i in tqdm(range(len(PET_files))):
        # 读取并预处理PET图像
        I_PET = imread(PET_files[i]).astype(np.float32).transpose(2, 0, 1)/255.  # [3, H, W] Uint8->float32
        I_PET = rgb2y(I_PET)  # 转换为亮度通道 [1, H, W] Float32
        # 读取并预处理红外图像
        I_MRI = imread(MRI_files[i]).astype(np.float32)[None, :, :]/255.  # [1, H, W] Float32
        # 将图像分割成patches    
        I_MRI_Patch_Group = Im2Patch(I_MRI, img_size, stride)
        I_PET_Patch_Group = Im2Patch(I_PET, img_size, stride)
        
        # 处理每个patch
        for ii in range(I_MRI_Patch_Group.shape[-1]):
            # 检查patch的对比度
            bad_IR = is_low_contrast(I_MRI_Patch_Group[0,:,:,ii])
            bad_VIS = is_low_contrast(I_PET_Patch_Group[0,:,:,ii])
            
            # 如果两个patch都不是低对比度，则保存
            if not (bad_IR or bad_VIS):
                avl_IR= I_MRI_Patch_Group[0,:,:,ii]  # 可用的红外patch
                avl_VIS= I_PET_Patch_Group[0,:,:,ii]  # 可用的可见光patch
                avl_IR=avl_IR[None,...]  # 添加通道维度
                avl_VIS=avl_VIS[None,...]  # 添加通道维度

                # 将patches保存到HDF5文件
                # 在h5_ir组中，创建数据集(即每个小patch都保存)
                h5_ir.create_dataset(str(train_num), data=avl_IR, 
                                   dtype=avl_IR.dtype, shape=avl_IR.shape)
                h5_vis.create_dataset(str(train_num), data=avl_VIS, 
                                    dtype=avl_VIS.dtype, shape=avl_VIS.shape)
                train_num += 1        


# 关闭HDF5文件
h5f.close()

# 打开HDF5文件并显示其结构
with h5py.File(os.path.join('data',
                                 data_name+'_imgsize_'+str(img_size)+"_stride_"+str(stride)+'.h5'),"r") as f:
    for key in f.keys():
        print(f[key], key, f[key].name)
