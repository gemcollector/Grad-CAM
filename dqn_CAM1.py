import os
import torch
import argparse
from gradcam import GradCam
from dqn_pytorch.model import DQN
from dqn_pytorch.atari_wrappers import *
import torch.nn.functional as F

def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    # cam = heatmap + np.float32(img)
    cam = heatmap
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


if __name__ == "__main__":
    # 模型加载过程
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # if gpu is to be used
    model = DQN(84, 84, 18, device).to(device)
    model.load_state_dict(torch.load("./dqn_pytorch/SI_to_SE_best.pth"))
    grad_cam = GradCam(model=model, feature_module=model.conv3,
                       target_layer_names=["2"], use_cuda=True)
    env_name = 'SpaceInvaders'
    env_raw = make_atari('{}NoFrameskip-v4'.format(env_name))
    env = wrap_deepmind(env_raw, frame_stack=False, episode_life=True, clip_rewards=True)

    lib1 = "D:/seaquest/./"
    lib2 = "D:/trans/seaquest/fromSI/./"

    # 以非灰度图读出来原图片
    picture1 = cv2.imread(lib1 + "pic-16.jpg", 1)
    # 以灰度图读出来原图片
    picture2 = cv2.imread(lib1 + "pic-16.jpg", 0)
    # 将灰度图压缩
    img = cv2.resize(picture2, (84, 84), interpolation=cv2.INTER_AREA).reshape(84, 84, 1)
    # 叠起来，连续或者是同一张
    a = np.concatenate((img, img, img, img), axis=2)
    # 计算GRAD-CAM
    img2 = torch.FloatTensor(a).unsqueeze(0).reshape(1, 4, 84, 84) / 255
    print('img2.max()', img2.max())

    img2 = img2.cuda()
    output = model(img2)
    print(output)
    target_category = None
    grayscale_cam = grad_cam(img2, target_category)
    # 处理GRAD-CAM
    cam = show_cam_on_image(None, grayscale_cam)
    # 将CAM 放大到原图大小
    new_cam = cv2.resize(cam, (160, 210))
    # CAM和原图片融合
    final_pic = cv2.addWeighted(picture1, 0.4, new_cam, 0.6, dtype=cv2.CV_32F, gamma=0)
    # 保存
    cv2.imwrite(lib2 + "cam-pic-" + str(16) + ".jpg", new_cam)