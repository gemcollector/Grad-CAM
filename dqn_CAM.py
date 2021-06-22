import os
import torch
import cv2
import numpy as np
import argparse
from gradcam import GradCam
from dqn_pytorch.model import DQN
from dqn_pytorch.atari_wrappers import *


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


if __name__ == "__main__":
    # 模型加载过程
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # if gpu is to be used
    model = DQN(84, 84, 18, device).to(device)
    model.load_state_dict(torch.load('./dqn_pytorch/model_in_21400000.pth'))
    grad_cam = GradCam(model=model, feature_module=model.conv3,
                       target_layer_names=["2"], use_cuda=True)
    env_name = 'SpaceInvaders'
    env_raw = make_atari('{}NoFrameskip-v4'.format(env_name))
    env = wrap_deepmind(env_raw, frame_stack=False, episode_life=True, clip_rewards=True)
    lib1 = "./seaquest_pic/"
    lib2 = "./trans/grad-CAM/"

    for i in range(127, 128):
        img0 = cv2.imread(lib1 + "pic-" + str(i) + ".jpg", 1)
        img1 = cv2.imread(lib1 + "pic-" + str(i) + ".jpg", 0)
        img1 = cv2.resize(img1, (84, 84), interpolation=cv2.INTER_AREA).reshape(84, 84, 1)
        img2 = cv2.imread(lib1 + "pic-" + str(i + 1) + ".jpg", 0)
        img2 = cv2.resize(img2, (84, 84), interpolation=cv2.INTER_AREA).reshape(84, 84, 1)
        img3 = cv2.imread(lib1 + "pic-" + str(i + 2) + ".jpg", 0)
        img3 = cv2.resize(img3, (84, 84), interpolation=cv2.INTER_AREA).reshape(84, 84, 1)
        img4 = cv2.imread(lib1 + "pic-" + str(i + 3) + ".jpg", 0)
        img4 = cv2.resize(img4, (84, 84), interpolation=cv2.INTER_AREA).reshape(84, 84, 1)
        img = np.concatenate((img1, img2, img3, img4), axis=2)
        img_cam = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float()
        # print(img_cam)
        target_category = None

        output = model(img_cam)
        print(output)

        grayscale_cam = grad_cam(img_cam, target_category)

        # print(grayscale_cam.dtype)
        # 处理GRAD-CAM
        cam = show_cam_on_image(None, grayscale_cam)

        # 将CAM 放大到原图大小
        new_cam = cv2.resize(cam, (160, 210))
        # CAM和原图片融合
        final_pic = cv2.addWeighted(img0, 0.4, new_cam, 0.6, dtype=cv2.CV_32F, gamma=0)
        # 保存
        import matplotlib.pyplot as plt

        cv2.imwrite(lib2 + "pic-" + str(i) + ".jpg", final_pic)
        # cv2.imwrite(lib2 + "pic-" + str(i) + ".jpg", new_cam)
