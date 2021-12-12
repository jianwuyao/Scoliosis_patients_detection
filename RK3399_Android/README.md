# 基于 MNN 与 RK3399 的检测模型部署

​		为了便于实际使用，需将深度学习模型进行移动端部署，本项目基于 MNN 与 RK3399 实现了简单的 Android APP 开发。

1. 下载安装 Android studio，学习 IDE 中的编译、调试等基本操作；
2. 在 IDE 中配置好开发所需的 SDK、NDK 等 API、Tool（可能遇到各种问题 ... 问度娘）；
3. 准备一个带显示屏的开发板，项目用的是荣品电子 RK3399 4+16 主板、10.1 寸 mipi 显示屏、Android 8.1系统，若无开发板，可用开发者模式下的安卓手机代替（需在代码中修改分辨率）；
4. 根据需求分析，修改或更新代码；
5. 将训练好的 Pytorch 模型转换为移动端支持的模型格式：Pytorch -> ONNX -> MNN，相关教程网上很多，需注意算子支持问题；

```python
模型结构可视化：https://netron.app/
```

6. body_ssd_300.mnn 是人体目标检测模型，用于从图像中选取一个人，将其裁剪出来输入关键点检测模型；pose_hrnet_256.mnn 是人体目标检测模型，输出16张heatmap，从而得到每个关键点的位置；


```python
body_ssd用的是SSD算法：https://blog.csdn.net/qianqing13579/article/details/82106664
pose_hrnet用的是HRNet网络：https://blog.csdn.net/xiaolouhan/article/details/90142937
```

7. 利用 adb 工具将两个 mnn 模型文件放在手机内部存储空间的 MNN/ 文件夹，将配置文件 config.json 放在内部存储的根目录；在根目录创建 Results 文件夹，用于保存结果文件；将需检测的图片放在手机内部存储空间的 DCIM/Camera 文件夹（均可在代码中自定义路径）；

```python
adb 使用前需要先使用命令 adb root 和 adb remount
eg: adb push ${PATH}/config.json /sdcard/
```

8. 连接好所有设备（若无摄像头则无法使用实时检测功能），运行使用！

