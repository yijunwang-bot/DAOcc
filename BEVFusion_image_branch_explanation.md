# BEVFusion 图像分支详解

这份文档专门解释 `BEVFusion` 中图像分支的主流程，重点回答下面这些问题：

- 每一步在做什么
- 为什么要这样做
- 为什么不用别的方法
- 每个 shape 的含义是什么
- shape 变化代表着什么
- 为什么必须发生这样的变化

这份文档只聚焦 **图像主线**，也就是：

```text
数据前处理
-> 进入模型
-> extract_camera_features
-> camera BEV feature
```

---

## 1. 图像主线总览

当前已经确认的主线可以先概括成：

```text
图片路径
-> PIL.Image
-> 统一尺寸后的 PIL.Image
-> torch.Tensor
-> 单个 sample 的多相机张量
-> batch 形式的多相机张量
-> 拉平为 2D backbone 可处理的图像 batch
-> backbone 多尺度特征
-> neck 融合后的多相机特征
-> vtransform 映射后的 BEV 特征
```

---

## 2. 数据前处理阶段

### 2.1 `LoadMultiViewImageFromFiles`

#### 做什么

- 从磁盘读取 6 路相机图像
- 形成 `list[PIL.Image]`

#### 已观察到的结果

```python
len(results["img"]) == 6
results["img"][0].size == (1600, 900)
```

#### shape / size 含义

这里还是 `PIL.Image`，所以没有 tensor 的 `.shape`，而是：

```python
size = (W, H)
```

所以：

```python
(1600, 900)
```

表示：

- 宽 `1600`
- 高 `900`

#### 为什么这样做

模型不能直接使用字符串形式的图片路径，必须先把图像实际内容读到内存里。

#### 为什么不用别的方法

也可以用：

- OpenCV
- mmcv
- 其他图像库

但本质目的都是一样的：

```text
把“路径”变成“图像对象”
```

#### 它的意义

这是从：

```text
文件路径
```

变成：

```text
真实图像数据
```

的第一步。

---

### 2.2 `ImageAug3D`

#### 做什么

- 对图像做几何增强
- 典型包括：
  - resize
  - crop
  - flip
  - rotate
- 同时生成 `img_aug_matrix`

#### 已观察到的结果

增强前：

```python
results["img"][0].size == (1600, 900)
```

增强后：

```python
results["img"][0].size == (704, 256)
```

#### shape / size 含义

这里仍然是：

```python
PIL.Image.size = (W, H)
```

所以：

```python
(704, 256)
```

表示：

- 宽 `704`
- 高 `256`

它对应配置里的：

```yaml
image_size: [256, 704]
```

#### 为什么这样做

有两个主要原因。

##### 原因 1：统一输入尺寸

如果每张图大小都不同：

- batch 很难拼接
- backbone 输入不统一
- 后面特征对齐困难

所以必须把图像变成统一大小。

##### 原因 2：做数据增强

训练时如果始终看完全一致的图像分布，模型容易过拟合。  
通过随机增强，可以让模型对：

- 裁剪变化
- 缩放变化
- 翻转变化
- 小角度旋转

更鲁棒。

#### 为什么不用原图直接输入

如果直接用原图：

- 显存更高
- 计算更慢
- batch 更小
- 多相机场景代价更大

而且训练时泛化往往更差。

#### 为什么还要记录 `img_aug_matrix`

因为这一步不仅改了像素，还改变了几何关系。

例如：

- 原始图像里的某个像素坐标
- 经过 resize / crop / flip 后
- 已经不再对应原来的位置

如果后面做图像到 3D 空间的投影时不修正这个变化，就会导致空间对齐错误。

#### 它的意义

这是从：

```text
原始图像
```

变成：

```text
统一尺寸、带几何增强的训练图像
```

的步骤。

---

### 2.3 `ImageNormalize`

#### 做什么

- 把 `PIL.Image` 转成 `torch.Tensor`
- 做归一化

#### 已观察到的结果

```python
results["img"][0].shape == (3, 256, 704)
```

#### shape 含义

现在已经变成 tensor，shape 顺序是：

```python
(C, H, W)
```

所以：

```python
(3, 256, 704)
```

表示：

- `3`：RGB 通道
- `256`：高
- `704`：宽

#### 为什么要转成 Tensor

后面的卷积网络、BatchNorm、激活函数等都工作在 tensor 上，不处理 `PIL.Image`。

#### 为什么要做 Normalize

原始像素值分布通常不稳定，归一化之后可以：

- 提高训练稳定性
- 更匹配预训练 backbone 的输入分布
- 让梯度更平稳

#### 为什么不用原始像素直接训

可以，但通常会：

- 收敛慢
- 数值不稳定
- 预训练权重利用效果差

#### 它的意义

这是从：

```text
图像对象
```

变成：

```text
模型真正可计算的标准输入张量
```

的关键一步。

---

### 2.4 `GridMask`

#### 做什么

- 在训练时给图像加网格遮挡

#### 已知结论

这一步通常：

- 改图像内容
- 不改图像 shape

所以可以先记成：

```python
输入单张图 shape = (3, 256, 704)
输出单张图 shape = (3, 256, 704)
```

#### 为什么这样做

防止模型过度依赖局部纹理，提高鲁棒性。

#### 为什么不用其他增强

并不是不用其他增强，而是这里作者额外选择了 `GridMask` 作为一种遮挡增强方式。

---

### 2.5 `DefaultFormatBundle3D`

#### 做什么

- 把 6 路相机的单张 tensor 堆叠起来
- 再包成 `DataContainer`

#### 已观察到的结果

```python
results["img"].data.shape == (6, 3, 256, 704)
```

#### shape 含义

这里表示的是：

```python
(N, C, H, W)
```

所以：

- `6`：相机数
- `3`：RGB 通道
- `256`：高
- `704`：宽

#### 为什么要 stack

前面图像是：

```python
list[tensor(3,256,704), tensor(3,256,704), ...]
```

后面 dataloader 和模型更适合处理整块张量，所以要变成：

```python
tensor(6,3,256,704)
```

#### 为什么不用一直保留 list

因为：

- batch 拼接不方便
- tensor 运算更统一
- 模型 forward 更清晰

#### 它的意义

这是从：

```text
6 张彼此独立的图
```

变成：

```text
单个 sample 的统一多相机输入块
```

---

### 2.6 `Collect3D`

#### 做什么

- 把 `img`
- `points`
- `gt_bboxes_3d`
- `gt_labels_3d`
- `voxel_semantics`
- `mask_camera`
- 以及各种几何 meta

统一打包成交给模型的输入结构。

#### 为什么这样做

模型需要的不只是图像，还需要：

- 点云
- 标注
- occupancy 标签
- 几何矩阵

所以必须在进入模型前完成统一装箱。

#### 它的意义

这是从：

```text
零散字段
```

变成：

```text
模型 forward 标准输入字典
```

的最后一步。

---

## 3. 从单个 sample 到 batch

### 3.1 单个 sample 的图片张量

在前处理结束时，单个 sample 的图片是：

```python
(6, 3, 256, 704)
```

表示：

- 6 个相机
- 每个相机 1 张 RGB 图

### 3.2 batch 是怎么来的

通过：

- `DataLoader`
- `collate`
- `DataContainer` 的 batch 拼接

把多个 sample 堆成一个 batch。

如果：

```yaml
samples_per_gpu: 4
```

那么 batch 形式就是：

```python
(4, 6, 3, 256, 704)
```

如果：

```yaml
samples_per_gpu: 1
```

那么就是：

```python
(1, 6, 3, 256, 704)
```

### 3.3 shape 含义

batch 后的图片张量是：

```python
(B, N, C, H, W)
```

所以：

- `B`：batch size
- `N`：相机数
- `C`：通道
- `H,W`：图像尺寸

### 为什么要做 batch

因为训练通常不是一次只看一个样本，而是多个样本一起算：

- GPU 利用率更高
- 训练更快
- 梯度更稳定

#### 它的意义

这是从：

```text
单个样本
```

变成：

```text
一批样本
```

的步骤。

---

## 4. 进入模型：`BEVFusion.forward`

### 4.1 `forward` 的作用

它是模型总入口。

主要作用：

- 接收 dataloader 传进来的 batch
- 把主要逻辑交给 `forward_single`

### 为什么要单独有这层

这样做的好处是：

- 入口统一
- 便于兼容训练/推理
- 便于插入自动混精等包装逻辑

### 它的意义

这一步不是核心计算展开点，而是：

```text
模型总入口
```

---

## 5. `forward_single`：单次前向总调度

### 5.1 它在做什么

它负责：

1. 提取 camera 特征
2. 提取 lidar 特征
3. 把它们放入 `features`
4. 做 `fuser`
5. 走 `decoder`
6. 交给各个 `head`
7. 训练返回 loss，测试返回预测

### 为什么需要这一层

因为模型不是单一路径，而是多模块协作：

- camera branch
- lidar branch
- fuser
- decoder
- object / occ head

所以必须有一个“总调度器”。

### 它的意义

可以简单理解成：

```text
单个 batch 前向主流程的总编排函数
```

---

## 6. `extract_camera_features`：图像分支核心

这个函数是当前图像主线最关键的部分。

---

### 6.1 输入 `x.shape = (B, N, C, H, W)`

已观察到：

```python
x.shape = (4, 6, 3, 256, 704)
```

#### 含义

- `4`：batch size
- `6`：相机数
- `3`：RGB 通道
- `256,704`：图像尺寸

#### 为什么这样组织

因为每个样本有 6 个相机视角，必须保留“相机维度”。

#### 它代表什么

这时的 `x` 还是：

```text
原始多相机图像张量
```

---

### 6.2 `x = x.view(B * N, C, H, W)`

已观察到：

```python
(4, 6, 3, 256, 704)
-> (24, 3, 256, 704)
```

#### 为什么这样做

因为 2D backbone（ResNet、Swin、ConvNet）通常只处理：

```python
(batch, C, H, W)
```

它不直接处理：

```python
(B, N, C, H, W)
```

所以必须把：

- batch 维
- 相机维

合并。

#### 为什么不用 backbone 直接处理 5D

因为普通 2D 卷积定义就是 4D 输入。  
如果想直接处理 5D，就不是普通 2D backbone 了，需要另一套设计。

#### 变化意味着什么

这是从：

```text
多相机组织形式
```

变成：

```text
适配 2D CNN 的图像 batch
```

---

### 6.3 `x = self.encoders["camera"]["backbone"](x)`

#### 它在做什么

从图像中提取深层视觉特征。

#### 已观察到的现象

backbone 输出不是单个 tensor，而是多尺度特征：

```python
type(x) = tuple / list
len(x) > 1
```

示例之一：

```python
x[0].shape = (24, 512, 32, 88)
```

#### 为什么会有多个 shape

因为 backbone 不只输出一个尺度，而是多个 stage 的 feature map。

这就是“多尺度特征”。

你应该继续关注：

```python
x[0].shape
x[1].shape
x[2].shape
```

#### 为什么空间变小

从：

```python
(24, 3, 256, 704)
```

到：

```python
(24, 512, 32, 88)
```

空间分辨率下降，是因为 backbone 中存在：

- stride conv
- pooling
- stage 下采样

这样可以：

- 降低计算量
- 增大感受野
- 获取更抽象的语义特征

#### 为什么通道变大

从 `3` 变到 `512`，是因为网络需要用更多特征通道来表达高层语义信息。

#### 它的意义

这是从：

```text
原始图像
```

变成：

```text
多尺度视觉特征
```

的步骤。

---

### 6.4 `x = self.encoders["camera"]["neck"](x)`

#### 它在做什么

对 backbone 输出的多尺度特征进一步融合和整理。

#### 为什么要 neck

因为 backbone 的多尺度特征：

- 分辨率不同
- 语义强度不同

直接只拿某一层通常不够好。  
neck 的作用就是：

- 融合不同层级的信息
- 统一通道
- 为后续几何投影做准备

#### 为什么不用 backbone 直接输出去做投影

可以，但通常效果较差，因为：

- 单尺度信息不全面
- 低层语义弱
- 高层空间粗

neck 可以综合这些信息。

---

### 6.5 重新 reshape 回多相机结构

已观察到在 `vtransform` 前：

```python
x.shape = (4, 6, 256, 32, 88)
```

#### 为什么要变回去

前面拉平成 `(B*N, C, H, W)` 只是为了适配 2D backbone。

但后面做几何投影时，必须重新知道：

- 哪个 feature 属于哪个样本
- 哪个 feature 属于哪个相机

所以必须恢复：

```python
(B, N, C, h, w)
```

#### 变化意味着什么

这是从：

```text
普通图像 batch 特征
```

回到：

```text
按样本、按相机组织的多相机特征
```

---

### 6.6 `x = self.encoders["camera"]["vtransform"](...)`

#### 它在做什么

这是最关键的步骤：

> 把多相机图像特征，结合几何信息，映射到统一的 BEV 空间。

#### 已观察到的 shape 变化

输入：

```python
(4, 6, 256, 32, 88)
```

输出：

```python
(4, 1280, 180, 180)
```

#### 每一维含义

输入：

```python
(B, N, C, h, w)
```

输出：

```python
(B, C_bev, H_bev, W_bev)
```

所以：

- `4`：batch size
- `1280`：BEV 特征通道
- `180,180`：BEV 网格大小

#### 为什么必须这样做

因为 6 个相机本来在不同视角上，不能直接和点云特征融合。

如果想做：

- 多相机融合
- 与 lidar 对齐
- occupancy / detection

就必须把图像特征投影到统一空间。

#### 为什么不用“直接拼接 6 张图”

直接拼接只能得到视觉拼接，不能保证真实几何对齐。  
而这里任务目标是：

- 3D
- occupancy
- BEV

所以必须利用：

- `camera_intrinsics`
- `camera2ego`
- `lidar2camera`
- `lidar2image`
- `img_aug_matrix`
- `lidar_aug_matrix`

这些几何矩阵来做投影。

#### 为什么 `6` 这个维度没了

因为投影后已经不再按“相机张数”组织，而是：

```text
所有相机共同构成统一 BEV 平面特征
```

#### 为什么空间从 `32x88` 变成 `180x180`

因为这里已经从图像特征平面，切换到了：

```text
鸟瞰视角下的离散 BEV 网格
```

这个网格大小由配置中的 BEV 范围和划分方式决定。

#### 为什么通道是 `1280`

这和：

- neck 输出通道
- 深度/高度结构
- 内部实现方式

有关。

当前阶段先把它理解成：

```text
用于表达统一 BEV 空间信息的高维特征
```

#### 它的意义

这是从：

```text
多相机 2D 图像特征
```

变成：

```text
统一 BEV 特征
```

的关键拐点。

---

## 7. 回到 `forward_single`

### 7.1 `feature = self.extract_camera_features(...)`

#### 它在做什么

把 `extract_camera_features` 的返回值保存到 `feature`。

也就是说：

```text
extract_camera_features 的输出
=
forward_single 里的 camera feature
```

当前可理解为：

```python
feature.shape = (4, 1280, 180, 180)
```

### 7.2 `features.append(feature)`

#### 它在做什么

把 camera 分支输出放进 `features` 列表。

后面还会再加入：

- lidar 分支输出

然后一起交给 `fuser`。

#### 为什么用 list

因为这里需要先收集多路特征，再统一融合。

---

## 8. 当前图像分支主线一句话总结

```text
多相机原始图像
-> 统一尺寸与标准化
-> 单个 sample 的多相机图像张量
-> batch 形式图像张量
-> 拉平成 2D CNN 输入
-> backbone 提取多尺度视觉特征
-> neck 融合多尺度特征
-> 恢复多相机组织
-> vtransform 映射到统一 BEV 特征空间
-> 作为 camera feature 返回给 forward_single
```

---

## 9. 当前最关键的理解结论

1. 图像一开始是 2D、多相机、按视角组织的数据
2. backbone 必须先把它当普通 2D 图像 batch 处理
3. neck 用来融合多尺度特征
4. `vtransform` 才是真正把图像特征变成 BEV 特征的关键步骤
5. `extract_camera_features` 的输出，就是 `forward_single` 中 camera 分支的 `feature`
