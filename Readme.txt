U-Net文件：

dataset文件---包括训练集数据，训练集label，测试集数据，格式为.tif
Results文件---包括不同模型参数下得到的测试集对应的label的图片文件和label的npy文件
test_label文件---测试集的label

acc.py---准确率评估代码
datasets.py---获取图片数据代码
generate.py---将npy文件转换成二值灰度图代码
main.py---主程序
models.py---U-Net网络结构代码
models_xxx.py---U-Net改进网络结构代码，可替换原有网络运行得出改进后正确率
trainers.py---模型训练代码

How to work:
运行main.py可以获得模型输出的测试集对应的npy文件并保存在目录中，再运行generate.py可以将npy文件转化成对应的二值化灰度图即分割结果图片png格式并保存在目录中，最后运行acc.py获得测试集上的准确率。模型输出的测试集的分割结果图片已保存在Results文件中。

