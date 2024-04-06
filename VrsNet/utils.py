import numpy as np
import torch.nn.functional as F
import math
from torchvision import transforms
import torch
import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
matplotlib.use('agg')
from mpl_toolkits.mplot3d import Axes3D

MAPS = ['map3','map4']
Scales = [0.9, 1.1]
MIN_HW = 384
MAX_HW = 1584
IM_NORM_MEAN = [0.485, 0.456, 0.406]
IM_NORM_STD = [0.229, 0.224, 0.225]



def select_exemplar_rois(image):
    all_rois = []

    print("Press 'q' or Esc to quit. Press 'n' and then use mouse drag to draw a new examplar, 'space' to save.")
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break
        elif key == ord('n') or key == '\r':
            rect = cv2.selectROI("image", image, False, False)
            x1 = rect[0]
            y1 = rect[1]
            x2 = x1 + rect[2] - 1
            y2 = y1 + rect[3] - 1

            all_rois.append([y1, x1, y2, x2])
            for rect in all_rois:
                y1, x1, y2, x2 = rect
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            print("Press q or Esc to quit. Press 'n' and then use mouse drag to draw a new examplar")

    return all_rois

def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def sigmoid(x):
    return torch.sigmoid(x)



def dice_loss(pred_density, gt_density, threshold=0.50001):

    pred_density,gt_density = resize_tensors_bilinear(pred_density,gt_density)
    pred_density = sigmoid(pred_density)
    gt_density = sigmoid(gt_density)

    pred_mask = (pred_density > threshold).to(torch.float32)
    gt_mask = (gt_density > threshold).to(torch.float32)

    # 计算交集和并集
    intersection = torch.sum(pred_mask * gt_mask)
    union = torch.sum(pred_mask) + torch.sum(gt_mask)

    # 计算Dice系数
    dice = (2.0 * intersection) / (union + 1e-8)  # 加上一个很小的常数，避免分母为零

    # 计算Dice损失
    dice_loss = 1 - dice
    dice_loss = dice_loss

    return dice_loss

def PerturbationLoss(output,boxes,sigma=8, use_gpu=True):
    Loss = 0.
    if boxes.shape[1] > 1:
        boxes = boxes.squeeze()
        for tempBoxes in boxes.squeeze():
            y1 = int(tempBoxes[1])
            y2 = int(tempBoxes[3])
            x1 = int(tempBoxes[2])
            x2 = int(tempBoxes[4])
            out = output[:,:,y1:y2,x1:x2]
            GaussKernel = matlab_style_gauss2D(shape=(out.shape[2],out.shape[3]),sigma=sigma)
            GaussKernel = torch.from_numpy(GaussKernel).float()
            if use_gpu: GaussKernel = GaussKernel.cuda()
            Loss += F.mse_loss(out.squeeze(),GaussKernel)
    else:
        boxes = boxes.squeeze()
        y1 = int(boxes[1])
        y2 = int(boxes[3])
        x1 = int(boxes[2])
        x2 = int(boxes[4])
        out = output[:,:,y1:y2,x1:x2]
        Gauss = matlab_style_gauss2D(shape=(out.shape[2],out.shape[3]),sigma=sigma)
        GaussKernel = torch.from_numpy(Gauss).float()
        if use_gpu: GaussKernel = GaussKernel.cuda()
        Loss += F.mse_loss(out.squeeze(),GaussKernel) 
    return Loss


def MincountLoss(output,boxes, use_gpu=True):
    Loss = 0.
    if boxes.shape[1] > 1:
        boxes = boxes.squeeze()
        for tempBoxes in boxes.squeeze():
            y1 = int(tempBoxes[1])
            y2 = int(tempBoxes[3])
            x1 = int(tempBoxes[2])
            x2 = int(tempBoxes[4])
            X = output[:,:,y1:y2,x1:x2].sum()
            X = (X-1) * (X-1) * 2
            Loss += min(X,2)
    else:
        boxes = boxes.squeeze()
        y1 = int(boxes[1])
        y2 = int(boxes[3])
        x1 = int(boxes[2])
        x2 = int(boxes[4])
        X = output[:,:,y1:y2,x1:x2].sum()
        X = (X - 1) * (X - 1) * 2
        Loss += min(X, 2)
    return Loss


def pad_to_size(feat, desire_h, desire_w):
    """ zero-padding a four dim feature matrix: N*C*H*W so that the new Height and Width are the desired ones
        desire_h and desire_w should be largers than the current height and weight
    """

    cur_h = feat.shape[-2]
    cur_w = feat.shape[-1]

    left_pad = (desire_w - cur_w + 1) // 2
    right_pad = (desire_w - cur_w) - left_pad
    top_pad = (desire_h - cur_h + 1) // 2
    bottom_pad =(desire_h - cur_h) - top_pad

    return F.pad(feat, (left_pad, right_pad, top_pad, bottom_pad))


def extract_features(feature_model, image, boxes,feat_map_keys=['map3','map4'], exemplar_scales=[0.8,0.9,1.1,1.2]):
    N, M = image.shape[0], boxes.shape[2]
    """
    Getting features for the image N * C * H * W
    """
    Image_features = feature_model(image)
    """
    Getting features for the examples (N*M) * C * h * w
    """
    # print(boxes)
    for ix in range(0,N):
        # boxes = boxes.squeeze(0)
        boxes = boxes[ix][0]
        cnter = 0
        Cnter1 = 0
        for keys in feat_map_keys:
            image_features = Image_features[keys][ix].unsqueeze(0)
            if keys == 'map1' or keys == 'map2':
                Scaling = 4.0
            elif keys == 'map3':
                Scaling = 8.0
            elif keys == 'map4':
                Scaling =  16.0
            else:
                Scaling = 32.0
            boxes_scaled = boxes / Scaling
            #print('before---------')
            #print(boxes_scaled)

            boxes_scaled[:, 1:3] = torch.floor(boxes_scaled[:, 1:3])
            boxes_scaled[:, 3:5] = torch.ceil(boxes_scaled[:, 3:5])
            boxes_scaled[:, 3:5] = boxes_scaled[:, 3:5] + 1 # make the end indices exclusive 
            feat_h, feat_w = image_features.shape[-2], image_features.shape[-1]
            # make sure exemplars don't go out of bound
            boxes_scaled[:, 1:3] = torch.clamp_min(boxes_scaled[:, 1:3], 0)
            boxes_scaled[:, 3] = torch.clamp_max(boxes_scaled[:, 3], feat_h)
            boxes_scaled[:, 4] = torch.clamp_max(boxes_scaled[:, 4], feat_w)
            #print('after---------')
            #print(boxes_scaled)
            box_hs = abs(boxes_scaled[:, 3] - boxes_scaled[:, 1])
            box_ws = abs(boxes_scaled[:, 4] - boxes_scaled[:, 2])
            max_h = math.ceil(max(box_hs))
            max_w = math.ceil(max(box_ws))            
            for j in range(0,M):
                y1, x1 = int(boxes_scaled[j,1]), int(boxes_scaled[j,2])  
                y2, x2 = int(boxes_scaled[j,3]), int(boxes_scaled[j,4])

                #print(y1,y2,x1,x2,max_h,max_w)
                if j == 0:
                    examples_features = image_features[:,:,y1:y2, x1:x2]
                   # print(examples_features.shape)
                    if examples_features.shape[2] != max_h or examples_features.shape[3] != max_w:
                        #examples_features = pad_to_size(examples_features, max_h, max_w)

                        examples_features = F.interpolate(examples_features, size=(max_h,max_w),mode='bilinear')                    
                else:
                    feat = image_features[:,:,y1:y2, x1:x2]
                    if feat.shape[2] != max_h or feat.shape[3] != max_w:
                        feat = F.interpolate(feat, size=(max_h,max_w),mode='bilinear')
                        #feat = pad_to_size(feat, max_h, max_w)
                    examples_features = torch.cat((examples_features,feat),dim=0)
            """
            Convolving example features over image features
            """
            h, w = examples_features.shape[2], examples_features.shape[3]
            features =    F.conv2d(
                    F.pad(image_features, ((int(w/2)), int((w-1)/2), int(h/2), int((h-1)/2))),
                    examples_features
                )
            combined = features.permute([1,0,2,3])
            # computing features for scales 0.9 and 1.1 
            for scale in exemplar_scales:
                    h1 = math.ceil(h * scale)
                    w1 = math.ceil(w * scale)
                    if h1 < 1: # use original size if scaled size is too small
                        h1 = h
                    if w1 < 1:
                        w1 = w
                    examples_features_scaled = F.interpolate(examples_features, size=(h1,w1),mode='bilinear')  
                    features_scaled =    F.conv2d(F.pad(image_features, ((int(w1/2)), int((w1-1)/2), int(h1/2), int((h1-1)/2))),
                    examples_features_scaled)
                    features_scaled = features_scaled.permute([1,0,2,3])
                    combined = torch.cat((combined,features_scaled),dim=1)
            if cnter == 0:
                Combined = 1.0 * combined
            else:
                if Combined.shape[2] != combined.shape[2] or Combined.shape[3] != combined.shape[3]:
                    combined = F.interpolate(combined, size=(Combined.shape[2],Combined.shape[3]),mode='bilinear')
                Combined = torch.cat((Combined,combined),dim=1)
            cnter += 1
        if ix == 0:
            All_feat = 1.0 * Combined.unsqueeze(0)
        else:
            All_feat = torch.cat((All_feat,Combined.unsqueeze(0)),dim=0)
    return All_feat


class resizeImage(object):
    """
    If either the width or height of an image exceed a specified value, resize the image so that:
        1. The maximum of the new height and new width does not exceed a specified value
        2. The new height and new width are divisible by 8
        3. The aspect ratio is preserved
    No resizing is done if both height and width are smaller than the specified value
    By: Minh Hoai Nguyen (minhhoai@gmail.com)
    """
    
    def __init__(self, MAX_HW=1504):
        self.max_hw = MAX_HW

    def __call__(self, sample):
        image,lines_boxes = sample['image'], sample['lines_boxes']
        
        W, H = image.size
        if W > self.max_hw or H > self.max_hw:
            scale_factor = float(self.max_hw)/ max(H, W)
            new_H = 8*int(H*scale_factor/8)
            new_W = 8*int(W*scale_factor/8)
            resized_image = transforms.Resize((new_H, new_W))(image)
        else:
            scale_factor = 1
            resized_image = image

        boxes = list()
        for box in lines_boxes:
            box2 = [int(k*scale_factor) for k in box]
            y1, x1, y2, x2 = box2[0], box2[1], box2[2], box2[3]
            boxes.append([0, y1,x1,y2,x2])

        boxes = torch.Tensor(boxes).unsqueeze(0)
        resized_image = Normalize(resized_image)
        sample = {'image':resized_image,'boxes':boxes}
        return sample


class resizeImageWithGT(object):
    """
    If either the width or height of an image exceed a specified value, resize the image so that:
        1. The maximum of the new height and new width does not exceed a specified value
        2. The new height and new width are divisible by 8
        3. The aspect ratio is preserved
    No resizing is done if both height and width are smaller than the specified value
    """
    
    def __init__(self, MAX_HW=1504):
        self.max_hw = MAX_HW

    def __call__(self, sample):
        image,lines_boxes,density = sample['image'], sample['lines_boxes'],sample['gt_density']
        
        W, H = image.size
        if W > self.max_hw or H > self.max_hw:
            scale_factor = float(self.max_hw)/ max(H, W)
            new_H = 8*int(H*scale_factor/8)
            new_W = 8*int(W*scale_factor/8)
            resized_image = transforms.Resize((new_H, new_W))(image)
            resized_density = cv2.resize(density, (new_W, new_H))
            orig_count = np.sum(density)
            new_count = np.sum(resized_density)

            if new_count > 0: resized_density = resized_density * (orig_count / new_count)
            
        else:
            scale_factor = 1
            resized_image = image
            resized_density = density
        boxes = list()
        for box in lines_boxes:
            box2 = [int(k*scale_factor) for k in box]
            y1, x1, y2, x2 = box2[0], box2[1], box2[2], box2[3]
            boxes.append([0, y1,x1,y2,x2])

        boxes = torch.Tensor(boxes).unsqueeze(0)
        resized_image = Normalize(resized_image)
        resized_density = torch.from_numpy(resized_density).unsqueeze(0).unsqueeze(0)
        sample = {'image':resized_image,'boxes':boxes,'gt_density':resized_density}
        return sample


Normalize = transforms.Compose([transforms.ToTensor(),
    transforms.Normalize(mean=IM_NORM_MEAN, std=IM_NORM_STD)])
Transform = transforms.Compose([resizeImage( MAX_HW)])
TransformTrain = transforms.Compose([resizeImageWithGT(MAX_HW)])


def denormalize(tensor, means=IM_NORM_MEAN, stds=IM_NORM_STD):
    denormalized = tensor.clone()
    for channel, mean, std in zip(denormalized, means, stds):
        channel.mul_(std).add_(mean)
    return denormalized


def scale_and_clip(val, scale_factor, min_val, max_val):
    "Helper function to scale a value and clip it within range"
    new_val = int(round(val*scale_factor))
    new_val = max(new_val, min_val)
    new_val = min(new_val, max_val)
    return new_val




def plot_density_contour(np_pre, save_path=None):
    plt.contour(np_pre, cmap='Spectral_r')
    #plt.title('Density Contour Plot')
    # plt.colorbar()
    plt.axis('off')  # 关闭坐标轴显示



from matplotlib.path import Path

def crop_and_display_by_contour_levels(original_image, density_map, save_path=None):
    # 显示原图
    plt.figure()
    plt.imshow(original_image, cmap='viridis')
    plt.title('Original Image')
    plt.axis('off')

    # 生成等高线图
    contours = plt.contour(density_map, cmap='Spectral_r')

    # 获取等高线的级别
    levels = contours.levels

    # 遍历每个级别
    for i, level in enumerate(levels):
        # 获取当前级别的所有路径
        paths = contours.collections[i].get_paths()

        # 创建一个空白的掩码
        mask = np.zeros_like(density_map, dtype=bool)

        # 将每条路径添加到掩码中
        for path in paths:
            vertices = path.vertices.astype(int)
            # 使用contains_points判断像素是否在路径内
            rr, cc = np.meshgrid(np.arange(mask.shape[0]), np.arange(mask.shape[1]), indexing='ij')
            points = np.column_stack((rr.flatten(), cc.flatten()))
            mask |= path.contains_points(points).reshape(mask.shape)

        # 使用掩码裁剪原图
        cropped_image = np.copy(original_image)
        cropped_image[~mask] = 255  # 将掩码外的区域置零

        # 显示裁剪后的图像
        plt.figure()
        plt.imshow(cropped_image, cmap='viridis')
        plt.title(f'Cropped Image - Contour Level {level}')
        plt.axis('off')

        # 保存裁剪后的图像
        if save_path:
            plt.savefig(f'{save_path}_contour_leveld_{i + 1}.png', bbox_inches='tight', pad_inches=0)

def plot_density_contour_filled(np_pre, save_path=None,img_name = None,figsize=(20,20)):
    plt.figure(figsize=figsize)
    # 使用 plt.contourf() 生成带有填充颜色的等高线图
    contour = plt.contourf(np_pre,levels= 50 ,cmap='Spectral_r',origin='upper')

    # 添加颜色条
    #plt.colorbar(contour, label='Density')
    # 关闭坐标轴显示
    plt.axis('off')
    if save_path:
        plt.savefig(f'{save_path}/{img_name}_filled.png', bbox_inches='tight', pad_inches=0)
    plt.close()

def equal_dimension(x, y):
    target_size = (y.size(0), y.size(1))
    resized_tensor = F.interpolate(x.unsqueeze(0).unsqueeze(0), size=target_size, mode='bilinear', align_corners=False)
    resized_tensor = resized_tensor.squeeze(0).squeeze(0)
    return resized_tensor


def calculate_and_scale_area(original_image, density_map, scaling_factor, save_path=None,image_name = None):

    density_map = equal_dimension(density_map,original_image)
    # 生成等高线图
    contours = plt.contour(density_map,levels=10,cmap='Spectral_r',origin='upper')
    # 获取等高线的级别
    levels = contours.levels[:5]
    # 遍历每个级别
    for i, level in enumerate(levels):
        # 获取当前级别的所有路径
        paths = contours.collections[i].get_paths()
        # 计算并缩放每条等高线的面积
        scaled_areas = []
        for path in paths:
            area = 0.5 * np.abs(np.dot(path.vertices[:, 0], np.roll(path.vertices[:, 1], 1)) -
                                np.dot(np.roll(path.vertices[:, 0], 1), path.vertices[:, 1]))
            scaled_area = scaling_factor * area
            scaled_areas.append(scaled_area)
        # 创建一个空白的掩码
        mask = np.zeros_like(density_map, dtype=float)  # 将掩码的数据类型改为整数
        # 将每条路径添加到掩码中，并进行面积缩放
        for j, path in enumerate(paths):
            vertices = path.vertices.astype(int)
            rr, cc = np.meshgrid(np.arange(mask.shape[0]), np.arange(mask.shape[1]), indexing='ij')
            points = np.column_stack((rr.flatten(), cc.flatten()))
            mask_for_path = path.contains_points(points).reshape(mask.shape)

            # 将缩放后的面积应用到掩码上
            mask[mask_for_path] += scaled_areas[j]  # 使用加法而不是按位或运算

        # 使用掩码裁剪原图
        cropped_image = np.copy(original_image)
        cropped_image[mask == 0] = 255  # 将掩码值为0的区域置零

        # 显示裁剪后的图像
        plt.figure()
        plt.imshow(cropped_image, cmap='viridis')
        # plt.title(f'Scaled Area - Contour Level {level}')
        plt.axis('off')

        # 保存裁剪后的图像
        if save_path:
            plt.savefig(f'{save_path}/{image_name}_{level}.png', bbox_inches='tight', pad_inches=0)

def visualize_output_and_savesubplot(input_, output,box,save_path, image_name=None,figsize = (20,12),dots=None):



    img1 = format_for_plotting(denormalize(input_))
    output = format_for_plotting(output)
    # sv = './group/'
    # calculate_and_scale_area(img1, output, 3,sv,image_name = image_name)
    fig = plt.figure(figsize=figsize)

    # Display the input image
    ax = fig.add_subplot(2, 2, 1)
    ax.set_axis_off()
    ax.imshow(img1)



    if dots is not None:
        ax.scatter(dots[:, 0], dots[:, 1], c='red', edgecolors='blue')
        # ax.set_title("Input image, gt count: {}".format(dots.shape[0]))
    else:
        print('1')
        # ax.set_title("Input image")

    # Save the first subplot

    # Create a new figure for the remaining subplots
    fig = plt.figure(figsize=figsize)

    ax = fig.add_subplot(2, 2, 2)
    ax.set_axis_off()
    # ax.set_title("Overlaid result, predicted count: {:.2f}".format(pred_cnt))

    # img2 = 0.2989 * img1[:, :, 0] + 0.5870 * img1[:, :, 1] + 0.1140 * img1[:, :, 2]
    ax.imshow(img1)

    min_pixel_value = torch.min(output)
    max_pixel_value = torch.max(output)
    ax.imshow(output, cmap='Spectral_r',vmin=min_pixel_value, vmax=max_pixel_value,alpha= 0.5)
    plot_density_contour(output)
    overlaid_result_save_path = save_path.replace('.jpg', '_overlaid_result.png')
    fig.savefig(overlaid_result_save_path, bbox_inches="tight",pad_inches=0)
    plt.close()

    # Create a new figure for the next subplot
    fig = plt.figure()

    ax = fig.add_subplot(2, 2, 3)
    ax.set_axis_off()
    # ax.set_title("Density map, predicted count: {:.2f}".format(pred_cnt))
    ax.imshow(output)

    # Save the third subplot
    density_map_save_path = save_path.replace('.png', '_density_map.png')
    fig.savefig(density_map_save_path, bbox_inches="tight")
    plt.close()

    # Create a new figure for the last subplot
    fig = plt.figure(figsize=figsize)

    ax = fig.add_subplot(2, 2, 4)
    ax.set_axis_off()
    # ax.set_title("Density map, predicted count: {:.2f}".format(pred_cnt))
    ret_fig = ax.imshow(output)

    # Save the fourth subplot
    density_map_with_boxes_save_path = save_path.replace('.png', '_density_map_with_boxes.png')
    fig.savefig(density_map_with_boxes_save_path, bbox_inches="tight")
    plt.close()







def format_for_plotting(tensor):
    """Formats the shape of tensor for plotting.
    Tensors typically have a shape of :math:`(N, C, H, W)` or :math:`(C, H, W)`
    which is not suitable for plotting as images. This function formats an
    input tensor :math:`(H, W, C)` for RGB and :math:`(H, W)` for mono-channel
    data.
    Args:
        tensor (torch.Tensor, torch.float32): Image tensor
    Shape:
        Input: :math:`(N, C, H, W)` or :math:`(C, H, W)`
        Output: :math:`(H, W, C)` or :math:`(H, W)`, respectively
    Return:
        torch.Tensor (torch.float32): Formatted image tensor (detached)
    Note:
        Symbols used to describe dimensions:
            - N: number of images in a batch
            - C: number of channels
            - H: height of the image
            - W: width of the image
    """

    has_batch_dimension = len(tensor.shape) == 4
    formatted = tensor.clone()

    if has_batch_dimension:
        formatted = tensor.squeeze(0)

    if formatted.shape[0] == 1:
        return formatted.squeeze(0).detach()
    else:
        return formatted.permute(1, 2, 0).detach()
import numpy as np
import cv2

def find_all_bounding_boxes(density_map, adaptive_threshold=True, threshold_multiplier=1.5):
    # Ensure density_map has two dimensions
    if len(density_map.shape) != 2:
        raise ValueError("Density map should have two dimensions.")

    # Use adaptive threshold method to determine target pixels
    if adaptive_threshold:
        threshold = np.mean(density_map) + threshold_multiplier * np.std(density_map)
        _, binary_map = cv2.threshold(density_map, threshold, 255, cv2.THRESH_BINARY)
    else:
        _, binary_map = cv2.threshold(density_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours of the target bounding boxes
    contours, _ = cv2.findContours(binary_map.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the bounding boxes for all contours
    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bbox_x1 = x
        bbox_y1 = y
        bbox_x2 = x + w
        bbox_y2 = y + h
        bounding_boxes.append((bbox_x1, bbox_y1, bbox_x2, bbox_y2))

    return bounding_boxes
def plot_counters(bboxes, image):
    plt.switch_backend('TkAgg')
    if bboxes:
        # 在原始图像上绘制所有目标框
        image = np.transpose(image, (1, 2, 0))
        image_with_bboxes = image.copy()
        for bbox in bboxes:
            bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox
            image_with_bboxes = cv2.rectangle(image_with_bboxes, (bbox_x1, bbox_y1), (bbox_x2, bbox_y2), (255, 0, 0), 2)

        # 可视化结果
        plt.figure(figsize=(8, 8))
        plt.imshow(image_with_bboxes)
        plt.title('Image with Bounding Boxes')
        plt.axis('off')
        plt.show()
    else:
        print("未找到目标框")




def resize_tensors_bilinear(tensor1, tensor2):
    # 获取两个张量的形状
    shape1 = tensor1.shape
    shape2 = tensor2.shape

    # 计算两个张量各个维度的最大值
    max_shape = [max(dim1, dim2) for dim1, dim2 in zip(shape1, shape2)]

    # 使用双线性插值将两个张量调整为相同的尺寸
    resized_tensor1 = F.interpolate(tensor1, size=max_shape[2:], mode='bilinear', align_corners=False)
    resized_tensor2 = F.interpolate(tensor2, size=max_shape[2:], mode='bilinear', align_corners=False)

    return resized_tensor1, resized_tensor2

def visual_features(x):
    feature_tensor = x.detach().cpu().numpy()
     # 选择一个通道的索引
    for i in range(6):
        channel_index = i
        plt.figure(figsize=(10, 6))
        plt.imshow(feature_tensor[0, 0, channel_index, :, :], cmap='viridis')
        #plt.title(f'Channel {channel_index} of the Feature Tensor')
        #plt.colorbar()
        plt.axis('off')
        plt.savefig(f'features{i}.png',bbox_inches='tight', pad_inches=0)


