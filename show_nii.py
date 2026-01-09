import numpy as np
from nnunet.network_architecture.generic_UNet import Generic_UNet
import cv2
import torch
import scipy.ndimage as ndimage
from skimage.transform import resize
from matplotlib import pyplot as plt
import nibabel
from torch import nn
from nnunet.network_architecture.initialization import InitWeights_He
import nibabel as nib
 
###---导入模型----###
MRI_path = '/home/zwz/nnUNet/data/nnUNet_raw_data_base/nnUNet_raw_data/Task09_Hippocampus/imagesTr/hippocampus_001.nii.gz'
# model_path = '/home/zwz1/nnUNet/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task009_Hippocampus/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/model_best.model'
# 读取数据
MRI = nibabel.load(MRI_path)
MRI_array = MRI.get_fdata()
MRI_array = MRI_array.astype('float32')

# 设置文件保存路径
normalized_data = MRI_array[:, 35, :]  # 选择切片的注意力图
normalized_data = (normalized_data - np.min(normalized_data)) / (np.max(normalized_data) - np.min(normalized_data))
save_path = '/home/zwz/nnUNet/image/sagittal_MRI_img1.png'

# 保存图像
plt.imshow(normalized_data, cmap="gray", vmin=0, vmax=1) 
#plt.imshow(normalized_data)
plt.axis('off')  # 关闭坐标轴
plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
plt.clf()  # 清除图像，以便绘制下一个图像
normalized_tensor = torch.from_numpy(normalized_data)
normalized_tensor = normalized_tensor.unsqueeze(0)
normalized_tensor = np.repeat(normalized_tensor, 3, axis=0)
normalized_tensor = np.transpose(normalized_tensor.numpy(), (1, 2, 0))
result_path = '/home/zwz/nnUNet/data/nnUNet_raw_data_base/nnUNet_raw_data/Task09_Hippocampus/labelsTr/hippocampus_001.nii.gz'
# 读取 nii.gz 文件
nii_image = nib.load(result_path)

# 获取数据并转换为 numpy 数组
data = nii_image.get_fdata()
# 将注意力图进行归一化处理，以便展示
data = data[26, :, :]  # 选择切片的注意力图
#cam = conv_out.features  # gain the ith output
# 设置文件保存路径
save_path = '/home/zwz/nnUNet/image/ground_truth2.png'

# 保存图像
plt.imshow(data, cmap="gray")
plt.axis('off')  # 关闭坐标轴
plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
plt.clf()  # 清除图像，以便绘制下一个图像
# 将注意力图映射为 RGB（使用 jet 色图）


# 归一化原始图像（如果它的值范围是 [0, 255] 或 [0, 1]，你可以跳过此步骤）



#attention_map_rgb = data.astype(np.float32)
#attention_map = (attention_map_rgb - attention_map_rgb.min()) #/ (attention_map.max() - attention_map.min())
attention_map_rgb = plt.cm.jet(data)[:, :, :3]  # 选择 jet 色图，去掉 alpha 通道

attention_overlay = attention_map_rgb + normalized_tensor
# 叠加注意力图和原始图像
# 加权叠加原始图像和注意力图

#overlay = cv2.addWeighted(sagittal_MRI_img_rgb, 0.5, attention_map_rgb, 0.5, 0)
#overlay = sagittal_MRI_img_rgb + attention_map_rgb
# 可视化叠加图像
plt.imshow(attention_overlay[:,:,0], cmap="jet") #[:,:,1]
#plt.imshow(overlay)
plt.axis('off')
plt.show()

# 保存叠加图像
plt.savefig('/home/zwz/nnUNet/image/CAM_demo_test1.png', bbox_inches='tight', pad_inches=0)
#output = np.squeeze(output[sagittal_slice_count, :,:])
 
# axial_MRI_img = np.squeeze(MRI_array[:, :, axial_slice_count])
# axial_grad_cmap_img = np.squeeze(heatmap[:, :, axial_slice_count])
 
# coronal_MRI_img = np.squeeze(MRI_array[:, coronal_slice_count, :])
# coronal_grad_cmap_img = np.squeeze(heatmap[:, coronal_slice_count, :])


#img_tensor_rgb = np.repeat(sagittal_MRI_img, 3, axis=1)

# 可视化原始图像

# plt.imshow(sagittal_MRI_img, cmap='gray')
# plt.axis('off')
# plt.savefig('/home/zwz1/nnUNet/image/sagittal_MRI_img.png', bbox_inches='tight', pad_inches=0)
# plt.clf()
# print("原始图像已保存original_image.png")
# for i in range(1, 2):
#     # 获取注意力图（注意力分布）
#     attention_map = torch.softmax(output, dim=1)# F.softmax(output, dim=1)
#     # print(attention_map.shape)

#     attention_map = attention_map[:, i, :, :, :]
#     #print(attention_map.size())
#     attention_map = attention_map.unsqueeze(0)
#     #attention_map = F.interpolate(attention_map, size=(224, 224), mode='bilinear', align_corners=False)
#     attention_map = attention_map.squeeze().detach().cpu().numpy()


#     cmap = plt.cm.jet
#     # 将注意力图进行归一化处理 attention_map.min()
#     attention_map = (attention_map - attention_map.min()) #/ (attention_map.max() - attention_map.min())
#     slice_attention_map = attention_map[:, 27, :]  # 选择该切片
#     plt.imshow(slice_attention_map, cmap='gray')
#     plt.axis('off')
#     plt.savefig('/home/zwz1/nnUNet/image/slice_attention_map.png', bbox_inches='tight', pad_inches=0)

#     # 归一化切片数据，确保它在 [0, 1] 范围内
#     slice_attention_map = (slice_attention_map - slice_attention_map.min()) / (slice_attention_map.max() - slice_attention_map.min())

#     # 使用 colormap 将切片数据转换为 RGB 图像
#     cmap = plt.cm.jet
#     #rgb_slice = cmap(slice_attention_map)[:, :, :3]
#     #rgb_slice = cmap(sagittal_MRI_img)[:, :, :3]
#     # 叠加注意力图到原始图像上
#     #attention_overlay = (cmap(attention_map)[:, :, :3]) + np.transpose(sagittal_MRI_img.squeeze(), (1, 2, 0))
#     attention_overlay = cmap(slice_attention_map)[:, :, :3] + cmap(sagittal_MRI_img)[:, :, :3]


#     # 可视化注意力叠加图像
#     plt.imshow(attention_overlay[:,:,1], cmap="jet") #[:,:,1]
#     plt.axis('off')
#     plt.savefig('/home/zwz1/nnUNet/image/CAM_demo_test1.png')

#     # 显示保存成功的信息
#     print("注意力叠加图像已保存至attention_overlay_" + str(i) + ".png")
 
# # Sagittal view
# img_plot = axarr[0, 0].imshow(np.rot90(sagittal_MRI_img, 1), cmap='gray')
# axarr[0, 0].axis('off')
# axarr[0, 0].set_title('Sagittal MRI', fontsize=25)
 
# img_plot = axarr[0, 1].imshow(np.rot90(sagittal_grad_cmap_img, 1), cmap='jet')
# axarr[0, 1].axis('off')
# axarr[0, 1].set_title('Weight-CAM', fontsize=25)
 
# # Zoom in ten times to make the weight map smoother
# sagittal_MRI_img = ndimage.zoom(sagittal_MRI_img, (1, 1), order=3)
# # Overlay the weight map with the original image
# sagittal_overlay = cv2.addWeighted(sagittal_MRI_img, 0.3, sagittal_grad_cmap_img, 0.6, 0)
 
# img_plot = axarr[0, 2].imshow(np.rot90(sagittal_overlay, 1), cmap='jet')
# axarr[0, 2].axis('off')
# axarr[0, 2].set_title('Overlay', fontsize=25)
 
# # Axial view
# img_plot = axarr[1, 0].imshow(np.rot90(axial_MRI_img, 1), cmap='gray')
# axarr[1, 0].axis('off')
# axarr[1, 0].set_title('Axial MRI', fontsize=25)
 
# img_plot = axarr[1, 1].imshow(np.rot90(axial_grad_cmap_img, 1), cmap='jet')
# axarr[1, 1].axis('off')
# axarr[1, 1].set_title('Weight-CAM', fontsize=25)
 
# axial_MRI_img = ndimage.zoom(axial_MRI_img, (1, 1), order=3)
# axial_overlay = cv2.addWeighted(axial_MRI_img, 0.3, axial_grad_cmap_img, 0.6, 0)
 
# img_plot = axarr[1, 2].imshow(np.rot90(axial_overlay, 1), cmap='jet')
# axarr[1, 2].axis('off')
# axarr[1, 2].set_title('Overlay', fontsize=25)
 
# # coronal view
# img_plot = axarr[2, 0].imshow(np.rot90(coronal_MRI_img, 1), cmap='gray')
# axarr[2, 0].axis('off')
# axarr[2, 0].set_title('Coronal MRI', fontsize=25)
 
# img_plot = axarr[2, 1].imshow(np.rot90(coronal_grad_cmap_img, 1), cmap='jet')
# axarr[2, 1].axis('off')
# axarr[2, 1].set_title('Weight-CAM', fontsize=25)
 
# coronal_ct_img = ndimage.zoom(coronal_MRI_img, (1, 1), order=3)
# Coronal_overlay = cv2.addWeighted(coronal_ct_img, 0.3, coronal_grad_cmap_img, 0.6, 0)
 
# img_plot = axarr[2, 2].imshow(np.rot90(Coronal_overlay, 1), cmap='jet')
# axarr[2, 2].axis('off')
# axarr[2, 2].set_title('Overlay', fontsize=25)
 
# plt.colorbar(img_plot,shrink=0.5) # color bar if need
# # plt.show()
# plt.savefig('/home/zwz1/nnUNet/image/CAM_demo_test1.png')
    # model = Generic_UNet(1, 32, 4,
    #                                 4,
    #                                 2, 2, nn.Conv2d, nn.BatchNorm2d, {'eps': 1e-5, 'affine': True}, nn.Dropout2d,
    #                                 {'p': 0, 'inplace': True},
    #                                 nn.LeakyReLU, None, True, False, lambda x: x, InitWeights_He(1e-2),
    #                                 None, None, False, True, True)