# 第三届计图人工智能挑战赛——赛道二：大规模无监督语义分割

![](https://user-images.githubusercontent.com/20515144/196449430-5ac6a88c-24ea-4a82-8a45-cd244aeb0b3b.png)

## 简介

本项目包含了第三届计图挑战赛计图 - 大规模无监督语义分割的Jittor代码实现。

大型无监督语义分割目前面临着如下挑战：

1. 模型需要在没有图像层级的监督信息的条件下学到与类别有关的表征
2. 提取语义分割掩模需要模型学到形状表征
3. 形状与类别的表征应该在冲突尽量少的条件下共存
4. 模型应该能够一亿个较高的效率将自监督学习到的标签分配给每一个像素
5. 大规模的数据集能够帮助模型学习到丰富的表征信息，但是不可避免的也会带来大量的训练开销，因此训练的高效性也是必需的。

因此赛方提供的baseline方法包含如下四个步骤：

1. 自监督：用一个随机初始化的模型进行代理任务的自监督学习训练，从而得到了所有训练图像的特征图
2. 聚类：应用一个基于像素注意力的聚类方案来获得每个图像像素的伪类别
3. 微调：用生成的伪类别来微调已预训练的模型来改进分割质量
4. 推理：在推理阶段，模型给每张图像的每个像素打标签

更详细的方法描述见致谢中的论文

## 安装

本项目使用曙光智算提供的dcu算力运行，在单dcu核心上完成训练，预训练时间约为26小时，此后的像素语义对齐等阶段训练时间共约为7小时

#### 运行环境

使用曙光智算提供的运行环境，详见[曙光智算官网](https://ac.sugon.com/home/index.html)

#### 安装依赖

执行以下命令安装jittor以及相关环境依赖

```
python -m pip install scikit-learn
python -m pip install pandas
python -m pip install munkres
python -m pip install tqdm
python -m pip install pillow
python -m pip install opencv-python
python -m pip install faiss-gpu
```

## 数据集准备

本项目使用的ImageNet-S数据集建立在ImageNet2012数据集上。为了运行本项目，请首先从[ImageNet官网](https://www.image-net.org/)下载ILSVRC2012数据集，随后按照[ImageNet-S数据集项目页面](https://github.com/LUSSeg/ImageNet-S)的指引准备ImageNet-S数据集。由于我下载的数据集的文件目录结构与ImageNet-S数据集中所用到的有所不同，因此需要在ImageNet-S/data目录下的data_preparation_val.py作一些修改，在其中的两句`src, dst = item.split(' ')`后各添加一句`src = src.split('/')[1]`。此外，若希望不修改`./scripts/luss50_pass_jt.sh`脚本中的路径，请将ImageNet-S目录置于本项目目录的父目录下。

## 训练

执行以下命令运行原始设定的训练：

```shell
bash ./scripts/luss50_pass_jt.sh
```

执行以下命令运行单卡训练

```shell
bash ./train.sh
```

## 推理

在shell命令行运行

```shell
bash ./test.sh
```

或

```shell
python test.py
```

即可

## 致谢

此项目基于论文 *Large-scale unsupervised semantic segmentation* 实现，部分代码参考了 [jittor-pass](https://github.com/LUSSeg/PASS/tree/jittor)。
