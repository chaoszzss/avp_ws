# import sys
# sys.path.insert(0, '.')
# import argparse
# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from PIL import Image
# import numpy as np
# import cv2

# import lib.data.transform_cv2 as T
# from lib.models import model_factory
# from config import set_cfg_from_file

# # ROS
# import rospy
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge, CvBridgeError

# torch.set_grad_enabled(False)
# np.random.seed(123)

# # args
# parse = argparse.ArgumentParser()
# parse.add_argument('--config', dest='config', type=str, default='configs/bisenetv2.py',)
# parse.add_argument('--weight-path', type=str, default='./res/model_final_v1_2000.pth',)
# args = parse.parse_args()
# cfg = set_cfg_from_file(args.config)

# palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)


# class IPMSegmentation:
#     def __init__(self):
#         # 初始化ROS节点
#         rospy.init_node('ipm_segmentation_node')
#         self.bridge = CvBridge()

#         # 加载模型
#         self.net = model_factory[cfg.model_type](cfg.n_cats, aux_mode='eval')
#         weight_path = rospy.get_param('~weight_path', args.weight_path)
#         self.net.load_state_dict(torch.load(weight_path, map_location='cpu'), strict=False)
#         self.net.eval()
#         self.net.cuda()

#         # 订阅和发布话题
#         self.sub = rospy.Subscriber("/ipm_raw", Image, self.image_callback)
#         self.pub = rospy.Publisher("/ipm_seg", Image, queue_size=10)

#     def image_callback(self, msg):
#         try:
#             # 将ROS图像消息转换为Opencv格式
#             cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
#             im = cv_image[:, :, ::-1]  # BGR -> RGB
            
#             # 调整尺寸并归一化
#             im = cv2.resize(im, (1024, 512))  # 模型输入尺寸
#             im = im.astype(np.float32) / 255.0
#             im = torch.from_numpy(im).permute(2, 0, 1).unsqueeze(0).cuda()

#             # 进行推理
#             out = self.net(im)[0]
#             out = F.interpolate(out, size=im.size()[2:], mode='bilinear', align_corners=False)
#             out = out.argmax(dim=1)

#             # 将输出转换为numpy数组
#             out = out.squeeze().detach().cpu().numpy()
#             pred = palette[out]

#             # 发布分割的结果
#             seg_msg = self.bridge.cv2_to_imgmsg(pred, encoding="rgb8") # rgb8
#             self.pub.publish(seg_msg)

#         except CvBridgeError as e:
#             rospy.logerr(f"CV Bridge Error: {e}")
#         except Exception as e:
#             rospy.logerr(f"Error: {e}")

# if __name__ == "__main__":
#     ipm_segmentation = IPMSegmentation()
#     rospy.spin()


