import os
import torch
import cv2
import numpy as np
import argparse
from ppo_gradcam import GradCam
from ppo import PPO
from ppo_model import Policy
from a2c_ppo_acktr.envs import make_vec_envs
import matplotlib.pyplot as plt

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # if gpu is to be used
    param_path = './PPO_SI.pth'

    envs = make_vec_envs('SpaceInvadersNoFrameskip-v4', 1, 1,
                         0.9, None, device, False)

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': False})
    a = actor_critic


    actor_critic.load_state_dict(torch.load(param_path)[0].state_dict())
    actor_critic.to(device)
    lib1 = "./seaquest_pic/"

    # for name, module in actor_critic._modules.items():
    #     if module == actor_critic.base:
    #         print(module.main)


    grad_cam = GradCam(model=actor_critic, feature_module=actor_critic.base,
                       target_layer_names=["2"], use_cuda=True)

    def show_cam_on_image(img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap
        cam = cam / np.max(cam)
        return np.uint8(255 * cam)

    # def show_cam_on_image(img, mask):
    #     heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    #     heatmap = np.float32(heatmap) / 255
    #     cam = heatmap + np.float32(img)
    #     cam = cam / np.max(cam)
    #     return np.uint8(255 * cam)


    # print(actor_critic)
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
        img_cam = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float().to(device)
        # 用这样的方式将图像接入模型
        actor_features = actor_critic.base(img_cam)
        # print('actor_features:', actor_features)
        dist = actor_critic.dist(actor_features)
        print(dist.prob())

        target_category = None
        grayscale_cam = grad_cam(img_cam, target_category)
        cam = show_cam_on_image(None, grayscale_cam)

        # 将CAM 放大到原图大小
        new_cam = cv2.resize(cam, (160, 210))
        # CAM和原图片融合
        final_pic = cv2.addWeighted(img0, 0.4, new_cam, 0.6, dtype=cv2.CV_32F, gamma=0)
        cv2.imwrite("pic-" + str(i) + ".jpg", final_pic)
        # for name, module in actor_critic._modules.items():
        #     if name == 'base':
        #         print(module.main(img_cam))


if __name__ == "__main__":
    main()


