# 第三届计图人工智能挑战赛赛道二——大规模无监督语义分割

<!-- | 标题名称包含赛题、方法 -->

<!-- ![主要结果](https://s3.bmp.ovh/imgs/2022/04/19/440f015864695c92.png) -->

<!-- ｜展示方法的流程特点或者主要结果等 -->

## 简介

<!-- | 简单介绍项目背景、项目特点 -->

本项目包含了第三届计图挑战赛计图 - 大规模无监督语义分割的Jittor代码实现。主要方法如下：

## 安装 

<!-- | 介绍基本的硬件需求、运行环境、依赖安装方法 -->

本项目使用曙光智算提供的dcu算力运行，在单dcu核心上完成训练，预训练时间约为26小时，此后的像素语义对齐等阶段训练时间共约为7小时

#### 运行环境

使用曙光智算提供的运行环境，详见[曙光智算官网](https://ac.sugon.com/home/index.html)

#### 安装依赖

执行以下命令安装jittor以及相关环境依赖

```
pip install -r requirements.txt
```

<!-- #### 预训练模型 -->



## 训练

<!-- ｜ 介绍模型训练的方法 -->

运行以下命令执行原始设定的训练：

```shell
bash ./scripts/luss50_pass_jt.sh
```

运行以下命令执行单卡训练

```shell
bash ./train.sh
```


## 推理

<!-- ｜ 介绍模型推理、测试、或者评估的方法 -->

```shell
bash ./test.sh
```

或

```shell
python test.py
```

## 致谢

<!-- | 对参考的论文、开源库予以致谢，可选 -->

此项目基于论文 *Large-scale unsupervised semantic segmentation* 实现，部分代码参考了 [jittor-pass](https://github.com/LUSSeg/PASS/tree/jittor)。

<!-- ## 注意事项

点击项目的“设置”，在Description一栏中添加项目描述，需要包含“jittor”字样。同时在Topics中需要添加jittor。

![image-20220419164035639](https://s3.bmp.ovh/imgs/2022/04/19/6a3aa627eab5f159.png) -->
