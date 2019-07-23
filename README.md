多分辨率特征融合的行人再识别方法

1. 安装依赖环境:
    - [pytorch 1.0](https://pytorch.org/)
    - torchvision
    - [ignite](https://github.com/pytorch/ignite)
    - [yacs](https://github.com/rbgirshick/yacs)
2. 准备数据集；下载Market1501、CUHK03以及DukeMTMC-ReID数据集；数据集的组织形式如下：

    ```bash
    dataset
        market1501
            bounding_box_test/
            bounding_box_train/
    ```

* 训练

  1. 下载HRNet在ImageNet上的预训练权重，https://pan.baidu.com/s/1DD3WKxgLM1jawR87WdAtsw   密	码：itc1。

  2. 修改reid_stage_submit/data/datasets中的对应数据集的路径;

  3. 配置文件位于reid_stage_submit/configs/softmax_triplet_with_center.yml文件中。将1中下载的预训练权重位置填入PRETRAINED一栏。在该配置文件中还可以修改使用的loss，训练的epochs，以及日志文件和训练权重的保存位置。

  4. 运行python tools/train.py即可开始训练。

* 测试

  1. 下载已经训练好的权重文件：链接：https://pan.baidu.com/s/1uzmPVQaapgdq0K-J4vnfqg 
     提取码：juo1 

  2. 将该权重文件放置于result文件夹下的最深层路径。

  3. 运行python tools/test.py即可得到结果。



​	