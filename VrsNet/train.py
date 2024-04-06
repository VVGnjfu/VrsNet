"""
Training Code for VrsNet
"""
import torch.nn as nn
from model import CountRegressor, weights_normal_init, Resnet101FPNWithAttention
from utils import MAPS, Scales, Transform, TransformTrain, extract_features,dice_loss
from PIL import Image
import os
import torch
import argparse
import json
import numpy as np
from tqdm import tqdm
from os.path import exists, join
import random
import torch.optim as optim
import torch.nn.functional as F

parser = argparse.ArgumentParser(description="Trian for VrsNet")
parser.add_argument("-bg", "--backbone", type=str, default='Resnet101FPN', help="backbone you chosed")
parser.add_argument("-dp", "--data_path", type=str, default='./data/', help="Path to the TreeFsc dataset")
parser.add_argument("-o", "--output_dir", type=str, default="./Modelargs", help="/Path/to/output/logs/")
parser.add_argument("-ts", "--test-split", type=str, default='val', choices=["train", "test", "val"],
                    help="what data split to evaluate on on")
parser.add_argument("-ep", "--epochs", type=int, default=400, help="number of training epochs")
parser.add_argument("-g", "--gpu", type=int, default=0, help="GPU id")
parser.add_argument("-lr", "--learning-rate", type=float, default=1e-6, help="learning rate")
args = parser.parse_args()

data_path = args.data_path
anno_file = data_path + 'annotation_tree.json'
data_split_file = data_path + 'train_test_val.json'
im_dir = data_path + 'tree_jpg'
gt_dir = data_path + 'tree_density'

if not exists(args.output_dir):
    os.mkdir(args.output_dir)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

criterion = nn.MSELoss().cuda()

if args.backbone == 'ResNet50FPN':
    bgmodel = Resnet101FPNWithAttention()
    print(1)
else:
    bgmodel = Resnet101FPNWithAttention()

bgmodel.cuda()
regressor = CountRegressor(6, pool='mean')
weights_normal_init(regressor, dev=0.001)
regressor.cuda()

# updata
params_to_optimize = list(bgmodel.ca_map3.parameters()) + list(bgmodel.ca_map4.parameters()) + list(
    regressor.parameters())
optimizer = optim.Adam(params_to_optimize, lr=args.learning_rate)

with open(anno_file) as f:
    annotations = json.load(f)

with open(data_split_file) as f:
    data_split = json.load(f)


def train():
    print("Training on Tree train set data")
    im_ids = data_split['train']
    random.shuffle(im_ids)
    train_mae = 0
    train_rmse = 0
    train_loss = 0
    pbar = tqdm(im_ids)
    cnt = 0

    for im_id in pbar:
        try:
            cnt += 1
            anno = annotations[im_id]
            bboxes = anno['box_examples_coordinates']
            dots = np.array(anno['points'])

            rects = list()
            for bbox in bboxes:
                x1 = bbox[0][0]
                y1 = bbox[0][1]
                x2 = bbox[2][0]
                y2 = bbox[2][1]
                if y1 > y2:
                    t2 = x1
                    x1 = x2
                    x2 = t2
                    t = y1
                    y1 = y2
                    y2 = t
                rects.append([y1, x1, y2, x2])

            image = Image.open('{}/{}'.format(im_dir, im_id))
            image.load()
            density_path = gt_dir + '/' + im_id.split(".jpg")[0] + ".npy"
            # print(density_path)

            density = np.load(density_path).astype('float32')
            sample = {'image': image, 'lines_boxes': rects, 'gt_density': density}
            sample = TransformTrain(sample)
            image, boxes, gt_density = sample['image'].cuda(), sample['boxes'].cuda(), sample['gt_density'].cuda()
            optimizer.zero_grad()

            features = extract_features(bgmodel, image.unsqueeze(0), boxes.unsqueeze(0), MAPS, Scales)

            output = regressor(features)

            # if image size isn't divisible by 8, gt size is slightly different from output size
            if output.shape[2] != gt_density.shape[2] or output.shape[3] != gt_density.shape[3]:
                orig_count = gt_density.sum().detach().item()
                gt_density = F.interpolate(gt_density, size=(output.shape[2], output.shape[3]), mode='bilinear')
                new_count = gt_density.sum().detach().item()
                if new_count > 0: gt_density = gt_density * (orig_count / new_count)
            loss1 = criterion(output, gt_density)
            loss2 = dice_loss(output, gt_density)
            loss2_tr = loss2 * 0.0000005

            loss = loss1 + loss2_tr
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pred_cnt = torch.sum(output).item()
            gt_cnt = torch.sum(gt_density).item()
            cnt_err = abs(pred_cnt - gt_cnt)
            train_mae += cnt_err
            train_rmse += cnt_err ** 2
            pbar.set_description(
                'actual-predicted: {:6.1f}, {:6.1f}, error: {:6.1f}. Current MAE: {:5.2f}, RMSE: {:5.2f} Best VAL MAE: {:5.2f}, RMSE: {:5.2f}'.format(
                    gt_cnt, pred_cnt, abs(pred_cnt - gt_cnt), train_mae / cnt, (train_rmse / cnt) ** 0.5, best_mae,
                    best_rmse))
            print("")
        except Exception as e:
            print("CUDA out of memory:{}".format(im_id))
            continue


    train_loss = train_loss / len(im_ids)
    train_mae = (train_mae / len(im_ids))
    train_rmse = (train_rmse / len(im_ids)) ** 0.5
    return train_loss, train_mae, train_rmse


def eval():
    cnt = 0
    SAE = 0  # sum of absolute errors
    SSE = 0  # sum of square errors

    print("Evaluation on {} data".format(args.test_split))
    im_ids = data_split[args.test_split]
    pbar = tqdm(im_ids)
    for im_id in pbar:
        try:
            anno = annotations[im_id]
            bboxes = anno['box_examples_coordinates']
            dots = np.array(anno['points'])
            rects = list()
            for bbox in bboxes:
                x1 = bbox[0][0]
                y1 = bbox[0][1]
                x2 = bbox[2][0]
                y2 = bbox[2][1]
                if y1 > y2:
                    t2 = x1
                    x1 = x2
                    x2 = t2
                    t = y1
                    y1 = y2
                    y2 = t
                rects.append([y1, x1, y2, x2])
            image = Image.open('{}/{}'.format(im_dir, im_id))
            image.load()
            sample = {'image': image, 'lines_boxes': rects}
            sample = Transform(sample)
            image, boxes = sample['image'].cuda(), sample['boxes'].cuda()
            with torch.no_grad():
                output = regressor(extract_features(bgmodel, image.unsqueeze(0), boxes.unsqueeze(0), MAPS, Scales))
            gt_cnt = dots.shape[0]
            pred_cnt = output.sum().item()
            cnt = cnt + 1
            err = abs(gt_cnt - pred_cnt)
            SAE += err
            SSE += err ** 2
            pbar.set_description(
                '{:<8}: actual-predicted: {:6d}, {:6.1f}, error: {:6.1f}. Current MAE: {:5.2f}, RMSE: {:5.2f}'.format(im_id,
                                                                                                                      gt_cnt,
                                                                                                                      pred_cnt,
                                                                                                                      abs(pred_cnt - gt_cnt),
                                                                                                                      SAE / cnt,
                                                                                                                      (
                                                                                                                                  SSE / cnt) ** 0.5))
            print("")
        except Exception as e:
            continue
    print('On {} data, MAE: {:6.2f}, RMSE: {:6.2f}'.format(args.test_split, SAE / cnt, (SSE / cnt) ** 0.5))
    return SAE / cnt, (SSE / cnt) ** 0.5


best_mae, best_rmse = 1e7, 1e7
stats = list()
for epoch in range(0, args.epochs):
    regressor.train()
    bgmodel.train()
    train_loss, train_mae, train_rmse = train()
    regressor.eval()
    bgmodel.eval()
    val_mae, val_rmse = eval()
    stats.append((train_loss, train_mae, train_rmse, val_mae, val_rmse))
    stats_file = join(args.output_dir, "stats_ca" + ".txt")
    with open(stats_file, 'w') as f:
        for s in stats:
            f.write("%s\n" % ','.join([str(x) for x in s]))
    if best_mae >= val_mae:
        best_mae = val_mae
        best_rmse = val_rmse
        if args.backbone == 'ResNet50FPN':
            bgmodel_name = "Modelargs" + "/TreeFsc/{}MLCA{}.pth".format(50, epoch)
            regressor_model_name = "Modelargs" + "/TreeFsc/{}Regreesor{}.pth".format(50, epoch)
        else:
            bgmodel_name = "Modelargs" + "/TreeFsc/{}MLCA{}.pth".format(101, epoch)
            regressor_model_name = "Modelargs" + "/TreeFsc/{}Regreesor{}.pth".format(101, epoch)
        torch.save(regressor.state_dict(), regressor_model_name)
        torch.save(bgmodel.state_dict(), bgmodel_name)
    print(
        "Epoch {}, Avg. Epoch Loss: {} Train MAE: {} Train RMSE: {} Val MAE: {} Val RMSE: {} Best Val MAE: {} Best Val RMSE: {} ".format(
            epoch + 1, stats[-1][0], stats[-1][1], stats[-1][2], stats[-1][3], stats[-1][4], best_mae, best_rmse))





