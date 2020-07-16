# Author:电子科技大学刘俊凯、陈昂
# https://github.com/JKLinUESTC/License-Plate-Recognization-Pytorch
import sys
import os

sys.path.append(os.getcwd())
from STN.model.STN import STNet
import numpy as np
import argparse
import torch
import time
import cv2


def convert_image(inp):
    # convert a Tensor to numpy image
    inp = inp.squeeze(0).cpu()
    inp = inp.detach().numpy().transpose((1, 2, 0))
    inp = 127.5 + inp / 0.0078125
    inp = inp.astype('uint8')

    return inp

if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## 加载网络
    STN = STNet()
    STN.to(device)
    STN.load_state_dict(torch.load('STN/weights/STN_Model_LJK_CA_XZH.pth', map_location=lambda storage, loc: storage))
    STN.eval()

    print("空间变换网络搭建完成")
    for i in range(1,21):    # 调整测试数量
        i = str(i)
        num = "test_dataset/"+i+".PNG"
        image = cv2.imread(num)
        im = cv2.resize(image, (94, 24), interpolation=cv2.INTER_CUBIC)
        im = (np.transpose(np.float32(im), (2, 0, 1)) - 127.5) * 0.0078125
        data = torch.from_numpy(im).float().unsqueeze(0).to(device)  # torch.Size([1, 3, 24, 94])
        transfer = STN(data)
        transformed_img = convert_image(transfer)
        Img_name = "test_reults\\" + i + ".PNG"
        cv2.imwrite(Img_name, transformed_img)

    # cv2.waitKey()
    # cv2.destroyAllWindows()
