import torch
import torch.nn.functional as F
import argparse
import cv2
import numpy as np
import torch
from torch.autograd import Function
from torchvision import models, transforms


# 特征梯度提取
class FeatureExtractor():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        # resnet.layer4
        self.model = model
        # [0 1 2]中的2
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    # magic method
    def __call__(self, x):
        outputs = []
        self.gradients = []
        count = 0
        for layer in self.model:
            x = layer(x)
            if count == 5:
                x.register_hook(self.save_gradient)
                outputs += [x]
            count += 1
        return outputs, x


# 特征提取器
class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        # res_net_layer4
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module.main, target_layers)

    # 获取最后一层向前传播的梯度
    def get_gradients(self):
        return self.feature_extractor.gradients

    # magic method
    def __call__(self, x):
        target_activations = []
        x = x.float() / 255.
        for name, module in self.model._modules.items():
            # 判断是否是需要的resnet50.layer4,取出最后一层所有的特征图
            if module == self.feature_module:
                # 将梯度取出
                target_activations, x = self.feature_extractor(x)

            else:
                x = module(x)
                print(x)


        return target_activations, x


def preprocess_image(img):
    normalize = transforms.Normalize(mean=[0.485], std=[0.229])
    preprocessing = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    return preprocessing(img.copy()).unsqueeze(0)


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


class GradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        # CNN NET
        self.model = model
        # FEATURE LAYER
        self.feature_module = feature_module
        # 开启eval模式
        # self.model.eval()
        # 是否使用cuda
        self.cuda = True
        if self.cuda:
            self.model = model.cuda()
        # 特征提取类
        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    # Res_net前向过程
    def forward(self, input_img):
        return self.model(input_img)

    def __call__(self, input_img, target_category=None):
        if self.cuda:
            input_img = input_img.cuda()

        # 取出最后一层所有通道的特征和输出

        features, output = self.extractor(input_img)

        # output = self.model.dist(output)
        print('PPO output is:', output.prob())

        # 把output转成概率
        output = output.prob()


        if target_category == None:
            target_category = np.argmax(output.cpu().data.numpy())
        # 1*1000的全0向量
        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        # 目标category的类别对应的像素为1
        one_hot[0][target_category] = 1
        #output[0][target_category] = 100
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = one_hot.cuda()
        # 和输出相乘求和
        one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)
        # 取出最后一层的梯度（之前已经被钩子函数记录了）
        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()
        print('grads_val.shape:', grads_val.shape)
        target = features[-1]
        print('target.shape:', target.shape)
        target = target.cpu().data.numpy()[0, :]
        # 把yc对特征图每个像素的偏导数求出来之后，取一次宽高维度上的全局平均
        # gradcam求梯度
        print(np.mean(grads_val, axis=(2, 3)).shape)
        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)
        # x-grad求权重
        x_weights = np.sum(grads_val[0, :] * target, axis=(1, 2))
        x_weights = x_weights / (np.sum(target, axis=(1, 2)) + 1e-6)
        x_cam = np.zeros(target.shape[1:], dtype=np.float32)


        # 加权求和
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
            x_cam += x_weights[i] * target[i, :, :]


        # 得出CAM
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, input_img.shape[2:])
        x_cam = np.maximum(x_cam, 0)
        x_cam = cv2.resize(x_cam, input_img.shape[2:])
        # 归一化CAM图
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        x_cam = x_cam - np.min(x_cam)
        x_cam = x_cam / np.max(x_cam)

        return cam


class GuidedBackpropReLU(Function):
    @staticmethod
    def forward(self, input_img):
        positive_mask = (input_img > 0).type_as(input_img)
        output = torch.addcmul(torch.zeros(input_img.size()).type_as(input_img), input_img, positive_mask)
        self.save_for_backward(input_img, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input_img, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input_img > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input_img.size()).type_as(input_img),
                                   torch.addcmul(torch.zeros(input_img.size()).type_as(input_img), grad_output,
                                                 positive_mask_1), positive_mask_2)
        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        def recursive_relu_apply(module_top):
            for idx, module in module_top._modules.items():
                recursive_relu_apply(module)
                if module.__class__.__name__ == 'ReLU':
                    module_top._modules[idx] = GuidedBackpropReLU.apply

        # replace ReLU with GuidedBackpropReLU
        recursive_relu_apply(self.model)

    def forward(self, input_img):
        return self.model(input_img)

    def __call__(self, input_img, target_category=None):
        if self.cuda:
            input_img = input_img.cuda()

        input_img = input_img.requires_grad_(True)

        output = self.forward(input_img)

        if target_category == None:
            target_category = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][target_category] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = one_hot.cuda()

        one_hot = torch.sum(one_hot * output)
        one_hot.backward(retain_graph=True)

        output = input_img.grad.cpu().data.numpy()
        output = output[0, :, :, :]

        return output


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='./examples/plane.png',
                        help='Input image path')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args


def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img * 255)
