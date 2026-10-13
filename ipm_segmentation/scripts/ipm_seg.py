#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '.')
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
import time

import rospkg

# 获取包路径并添加到 Python 路径
# rospack = rospkg.RosPack()
# pkg_path = rospack.get_path('ipm_segmentation')
# sys.path.insert(0, f"{pkg_path}/lib")
# sys.path.insert(0, f"{pkg_path}/config")

import lib.data.transform_cv2 as T
from lib.models import model_factory
from config import set_cfg_from_file
# from ipm_segmentation.lib.data import transform_cv2 as T
# from ipm_segmentation.lib.models import model_factory
# from ipm_segmentation.config import set_cfg_from_file


# ROS
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

torch.set_grad_enabled(False)
np.random.seed(123)

# 定义可视化窗口名称
VIS_WINDOW_NAME = "IPM Segmentation Result"

# args
parse = argparse.ArgumentParser()
parse.add_argument('--config', dest='config', type=str, default='/home/chaoszzss/avp_ws/src/ipm_segmentation/scripts/config/bisenetv1_SUPS.py',)
parse.add_argument('--weight-path', type=str, default='/home/chaoszzss/avp_ws/src/ipm_segmentation/models/model_final_v1_4000_IPMdata.pth',)
# args = parse.parse_args()
args, unknown = parse.parse_known_args() 
cfg = set_cfg_from_file(args.config)

palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)
# palette = np.zeros((256, 3), dtype=np.uint8)
# palette[1] = [130, 63, 122]     # 类别1 BGR
# palette[2] = [0, 254, 254]      # 类别2 BGR
# palette[3] = [45, 192, 112]     # 类别3 BGR
# palette[4] = [0, 0, 0]          # 类别4 BGR
# palette[5] = [136, 133, 33]     # 类别5 BGR
# palette[6] = [254, 254, 254]    # 类别6 BGR
# palette[7] = [0, 191, 192]      # 类别7 BGR
# palette[8] = [254, 128, 76]     # 类别8 BGR
# palette = np.array([
#     [130, 63, 122],    # 类别0 → 实际类别1
#     [0, 254, 254],     # 类别1 → 实际类别2
#     [45, 192, 112],    # 类别2 → 实际类别3
#     [0, 0, 0],         # 类别3 → 实际类别4
#     [136, 133, 33],    # 类别4 → 实际类别5
#     [254, 254, 254],   # 类别5 → 实际类别6
#     [0, 191, 192],     # 类别6 → 实际类别7
#     [254, 128, 76]     # 类别7 → 实际类别8
# ], dtype=np.uint8)
# palette = np.vstack([palette, np.zeros((256-8,3), dtype=np.uint8)])


class IPMSegmentation:
    def __init__(self):
        
        # 初始化ROS节点
        rospy.init_node('ipm_segmentation')
        self.bridge = CvBridge()
        
        # 创建窗口
        # cv2.namedWindow(VIS_WINDOW_NAME, cv2.WINDOW_NORMAL)
        # cv2.resizeWindow(VIS_WINDOW_NAME, 1000, 1000)

        # define model
        self.net = model_factory[cfg.model_type](cfg.n_cats, aux_mode='eval')
        self.net.load_state_dict(torch.load(args.weight_path, map_location='cpu'), strict=False)
        self.net.eval()
        self.net.cuda()
        
        # 订阅和发布
        self.sub = rospy.Subscriber("/ipm_raw", Image, self.image_callback)
        self.pub = rospy.Publisher("/ipm_seg", Image, queue_size=10)  # 减小队列长度

    def image_callback(self, msg):
        try:
            # 检查回调函数是否被调用
            # if msg is None:
            #     rospy.logwarn("Received empty message!")
            #     return
            # else:
            #     rospy.loginfo("Received IPM image!")

            # prepare data
            im = self.bridge.imgmsg_to_cv2(msg, "rgb8")
            to_tensor = T.ToTensor(
                mean=(0.3257, 0.3690, 0.3223), # city, rgb
                std=(0.2112, 0.2148, 0.2115),
            )
            im = to_tensor(dict(im=im, lb=None))['im'].unsqueeze(0).cuda()

            # 转换图像并检查数据有效性
            # if im is None or im.size == 0:
            #     rospy.logwarn("Converted faulty image!")
            #     return
            # else:
            #     rospy.loginfo("Converted image to Tensor format!")
            
            # 形状调整（确保输入尺寸为32的倍数）
            # shape divisor
            org_size = im.size()[2:]
            new_size = [math.ceil(el / 32) * 32 for el in im.size()[2:]]

            # 检查输入数据类型和形状
            # rospy.loginfo(f"Input shape: {im.shape}, dtype: {im.dtype}")

            # 推理
            # 添加推理时间计算
            start_time = time.time() 
            im = F.interpolate(im, size=new_size, align_corners=False, mode='bilinear')
            out = self.net(im)[0]
            out = F.interpolate(out, size=org_size, align_corners=False, mode='bilinear')
            out = out.argmax(dim=1)
            # 计算时间差
            inference_time = time.time() - start_time
            print(f"Inference time: {inference_time:.4f} seconds")

            # 应用颜色映射
            out = out.squeeze().detach().cpu().numpy()
            pred = palette[out]
  

            # 检查输出数据类型和形状
            # rospy.loginfo(f"Output shape: {pred.shape}, dtype: {pred.dtype}")

            # 保存中间结果
            # cv2.imwrite('/home/chaoszzss/avp_ws/src/ipm_segmentation/scripts/res/debug_pred.png', pred)
            # rospy.loginfo("Saved debug images!")

            # 发布和显示
            seg_msg = self.bridge.cv2_to_imgmsg(pred, "bgr8")
            seg_msg.header = msg.header
            self.pub.publish(seg_msg)
            self._display_result(pred)    #可视化
            # cv2.imshow(VIS_WINDOW_NAME, pred)
            # cv2.waitKey(1)

            # 显存清理
            del im, out
            torch.cuda.empty_cache()

        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge Error: {e}")
        except Exception as e:
            rospy.logerr(f"Error: {e}")
    
    def _display_result(self, pred):
        try:
            # 确保 pred 是 uint8 类型且值在 0-255 范围内
            if pred.dtype != np.uint8:
                pred = pred.astype(np.uint8)

            # 保存调试图像
            # cv2.imwrite('./debug_seg.jpg', pred)

            cv2.namedWindow(VIS_WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(VIS_WINDOW_NAME, 800, 600)

            # 转换颜色空间 RGB->BGR
            display_img = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
            # display_img = pred  # 直接使用原始图像
            
            # 显示图像
            cv2.imshow(VIS_WINDOW_NAME, display_img)
            
            # 保持窗口响应（1ms延迟）
            cv2.waitKey(1)
        
        except Exception as e:
            rospy.logerr(f"Display error: {e}")

    # def __del__(self):
    #     """析构时关闭窗口"""
    #     cv2.destroyAllWindows()

if __name__ == "__main__":
    ipm_segmentation = IPMSegmentation()
    rospy.spin()
    # try:
    #     rospy.spin()
    # finally:
    #     cv2.destroyAllWindows()


