# STN_LPRNet_Pytorch

该代码由电子科技大学(UESTC)刘俊凯、陈昂主写，我们创新性地在LPRNet网络里加入了空间变换网络STN，提高了识别准确率
该网络完全适用于中国车牌识别（Chinese License Plate Recognition）及国外车牌识别！  
目前仅支持同时识别蓝牌和绿牌即新能源车牌等中国车牌，但可通过扩展训练数据或微调支持其他类型车牌及提高识别准确率！

# 所需环境配置

- pytorch >= 1.0.0
- opencv-python 3.x
- python 3.x
- imutils
- Pillow
- numpy

# 训练
我们使用了10万张车牌图片进行训练，已经给出生成的权重文件，读者不必再次训练。
若要再次训练，如有问题，可联系我们。
https://github.com/JKLinUESTC/License-Plate-Recognization-Pytorch
https://github.com/ca-jj/License-Plate-Recognization-Pytorch

#测试

1. LPRNet需要测试的车牌可以放进test文件夹内，size为94×24，图片名为车牌
2. LPRNet只需要运行LPRNet test函数即可得到结果
3. 其中我们已经添加了STN网络，如果您想去掉STN，只需要对images = STN（images）进行注释即可

# 表现

能够实时识别车牌图片，足够轻量级，在复杂环境下识别准确率依旧很高
1000张车牌图片的识别准确率在94%以上


# 参考文献

1. [LPRNet: License Plate Recognition via Deep Neural Networks](https://arxiv.org/abs/1806.10447v1)
2. [PyTorch中文文档](https://pytorch-cn.readthedocs.io/zh/latest/)
3. Jaderberg M, Simonyan K, Zisserman A. Spatial transformer networks[C]//Advances in neural information processing systems. 2015: 2017-2025

# postscript

如果您认可我们的工作，麻烦给我们一个星星，谢谢您
