# DAOcc 调试入门：从 `LoadPointsFromFile` 开始看数据流

这份文档是给刚开始看 `DAOcc` / `MMDetection3D` 代码的人准备的。

你的问题很典型：

> 我想从 `LoadPointsFromFile` 开始 debug 数据流，看看数据经过每个 class 后 shape 怎么变，但我找不到它的 `forward()`，也不知道该在哪里打断点。

先说结论：

- `LoadPointsFromFile` 不是模型层的类
- 它通常没有 `forward()`
- 它一般通过 `__call__()` 被执行
- 你调试它，重点是看 `results` 这个字典怎么变化

---

## 1. 先建立最基础的概念

在这类项目里，代码大致分成两层：

### 1.1 数据处理层

负责：

- 读取点云文件
- 读取图像
- 做数据增强
- 整理成模型输入

这一层的 class 常见位置：

- `mmdet3d/datasets/`
- `mmdet3d/datasets/pipelines/`

这类 class 最常见的执行入口是：

```python
__call__(self, results)
```

### 1.2 模型计算层

负责：

- backbone
- neck
- head
- loss
- 推理输出

这一层的 class 常见位置：

- `mmdet3d/models/`

这类 class 最常见的执行入口是：

```python
forward(self, ...)
```

---

## 2. 为什么 `LoadPointsFromFile` 没有 `forward()`

因为它通常不是 `nn.Module` 风格的模型层，而是一个“数据变换器”。

你可以把它理解成一个可以被当作函数调用的对象。

比如 Python 里：

```python
class A:
    def __call__(self, x):
        return x + 1

a = A()
print(a(3))
```

这里看起来是在调用 `a(3)`，本质上调用的是：

```python
a.__call__(3)
```

`LoadPointsFromFile` 这类 pipeline class 也是一样。

所以不要去找：

```python
LoadPointsFromFile.forward(...)
```

而应该去找：

```python
LoadPointsFromFile.__call__(results)
```

---

## 3. 数据流到底是怎么走的

你可以把整个训练流程想成两大阶段。

### 阶段 A：数据准备

这一步一般是：

```text
磁盘文件 -> dataset -> pipeline -> 整理成模型输入
```

这里流动的通常不是单个 tensor，而是一个字典，名字常常叫：

```python
results
```

例如一开始可能长这样：

```python
results = {
    "pts_filename": "/path/to/xxx.bin",
    "img_filename": [...],
    "ann_info": ...,
}
```

然后每个 pipeline class 会依次处理它：

```python
results = LoadPointsFromFile(results)
results = LoadPointsFromMultiSweeps(results)
results = GlobalRotScaleTrans(results)
results = RandomFlip3D(results)
...
```

### 阶段 B：模型前向

当数据准备好了，才会进入模型：

```python
model.forward(img, points, ...)
```

所以：

- `LoadPointsFromFile` 属于阶段 A
- `BEVFusion.forward()` 属于阶段 B

---

## 4. 什么是 pipeline

可以把 pipeline 想成“流水线”。

比如工厂里的流程：

1. 原材料进来
2. 第一站切割
3. 第二站打磨
4. 第三站喷漆
5. 最后打包出厂

代码里也一样：

1. 文件路径进来
2. `LoadPointsFromFile` 读取点云
3. `RandomFlip3D` 做翻转
4. `GlobalRotScaleTrans` 做旋转缩放
5. `Collect3D` 整理输出

每个 class 就像流水线中的一个工位。

你 debug 的核心就是：

- 这一站输入是什么
- 这一站输出是什么
- 和上一站相比改了什么

---

## 5. `results` 是什么，为什么它这么重要

在数据 pipeline 里，最重要的不是某一个单独 tensor，而是 `results` 这个字典。

因为很多 class 做的事情不是单纯“改 shape”，而是：

- 新增一个字段
- 修改一个字段
- 把 numpy 数组包装成点云对象
- 删除临时字段

例如 `LoadPointsFromFile` 很可能会把：

```python
results["pts_filename"]
```

变成：

```python
results["points"]
```

所以你调试时最该关心的是：

- `results.keys()`
- `results["pts_filename"]`
- `results["points"]`
- `results["points"]` 的类型
- `results["points"]` 的 shape

---

## 6. `LoadPointsFromFile` 大概在做什么

不同仓库版本会有一点差别，但通常流程类似：

1. 从 `results["pts_filename"]` 取出点云文件路径
2. 从磁盘读取原始点云
3. reshape 成 `[N, C]`
4. 根据配置选择需要的维度
5. 包装成 `BasePoints` 或类似对象
6. 存回 `results["points"]`

所以它最常见的效果是：

- 新增 `points`
- `points` 变成一个点云对象
- 内部真实数据 shape 类似 `[N, 4]`、`[N, 5]` 之类

---

## 7. 在哪里找 `LoadPointsFromFile`

你可以先全局搜索：

```bash
grep -R "class LoadPointsFromFile" -n .
```

通常会在类似下面的位置：

```text
mmdet3d/datasets/pipelines/loading.py
```

然后重点看这个 class 里的：

- `__init__`
- `__call__`
- 可能还有辅助函数，比如 `_load_points`

---

## 8. 为什么你现在应该打 `__call__` 的断点

因为这类 pipeline class 在执行时，常见形式是：

```python
data = t(data)
```

而不是：

```python
data = t.forward(data)
```

也就是说，真正会被执行的是：

```python
t.__call__(data)
```

所以 `LoadPointsFromFile` 的第一断点，应该下在：

```python
class LoadPointsFromFile:
    def __call__(self, results):
        ...
```

---

## 9. 最推荐的入门断点顺序

如果你刚开始 debug，推荐按这个顺序来。

### 9.1 先看单个 transform

先在：

- `LoadPointsFromFile.__call__`

打断点。

目的：

- 先看 `results` 长什么样
- 先理解这个 class 做了哪些事

### 9.2 再看整条 pipeline

然后去找 `Compose`，常见代码逻辑大概像这样：

```python
for t in self.transforms:
    data = t(data)
```

这里可以打断点，观察：

- 当前执行的是哪个 transform
- 每个 transform 执行前后 `results` 怎么变

### 9.3 再看 dataset 出口

然后再去看：

- `__getitem__`
- `prepare_train_data`

目的：

- 看 pipeline 最终返回给 dataloader 的到底是什么

### 9.4 最后看模型入口

等你把数据层看明白，再去看：

- `BEVFusion.forward()`
- `BEVFusion.forward_single()`

这样你就能把“文件 -> pipeline -> 模型输入”这条线完整串起来。

---

## 10. 第一次断点时应该看什么

如果你已经停在 `LoadPointsFromFile.__call__(results)` 里，建议优先看这些：

```python
results.keys()
```

看当前有哪些字段。

```python
results["pts_filename"]
```

看点云文件路径。

如果还没加载完点云，再单步执行几行，然后看：

```python
type(results["points"])
```

如果它是 OpenMMLab 的点云对象，通常还可以继续看：

```python
results["points"].tensor.shape
```

有时中间变量还没放回 `results`，那就看局部变量，比如：

```python
type(points)
points.shape
```

---

## 11. 为什么很多时候不能只看 `shape`

因为在 pipeline 里，变化不一定只是 shape。

可能发生的是：

- key 变了
- 类型变了
- 坐标值变了，但 shape 没变
- 点数变了
- 通道数变了

例如：

- `RandomFlip3D` 可能 shape 不变，但坐标值变了
- `GlobalRotScaleTrans` 可能 shape 不变，但点的位置变了
- `LoadPointsFromMultiSweeps` 可能点数变多
- `Collect3D` 可能主要是整理字段，不是改点云内容

所以你要养成两个同时看的习惯：

- 看字段
- 看类型和 shape

---

## 12. 最常见的一条误区

很多人刚开始会默认：

> 只要是个 class，就应该从 `forward()` 开始看。

这在 PyTorch 模型层里没问题，但在 OpenMMLab 的数据 pipeline 里经常不成立。

更准确的判断方式是：

- 如果 class 在 `datasets/pipelines/`，优先找 `__call__`
- 如果 class 在 `models/`，优先找 `forward`

---

## 13. 一个很实用的调试策略

如果你想知道“每个 class 后 shape 怎么变”，最稳的方法通常不是只盯一个类，而是在 pipeline 外层统一观察。

也就是说，不只是断在 `LoadPointsFromFile` 里，还要去看 pipeline 的总调度位置。

典型逻辑是：

```python
for t in self.transforms:
    data = t(data)
```

如果你在这里临时加打印，就能看到每一步经过了哪个 class。

例如你可以临时打印：

```python
print(t.__class__.__name__)
print(data.keys())
```

如果 `points` 已经存在，还可以继续看：

```python
pts = data["points"]
print(type(pts))
if hasattr(pts, "tensor"):
    print(pts.tensor.shape)
```

这样会比只在单个 class 里盯着看更容易建立整体感觉。

---

## 14. 你现在可以先记住的最小结论

如果只记住最关键的几句话，可以记这几条：

1. `LoadPointsFromFile` 属于数据 pipeline，不是模型层
2. 它通常没有 `forward()`，而是走 `__call__()`
3. pipeline 里流动的核心对象通常是 `results` 字典
4. debug 时要看 `results` 的字段、类型和 shape
5. 真正到模型阶段，才重点看 `forward()`

---

## 15. 推荐你接下来的实际操作顺序

建议你按这个顺序动手：

1. 搜索 `class LoadPointsFromFile`
2. 打开它所在文件
3. 找 `__call__(self, results)`
4. 在 `__call__` 第一行打断点
5. 观察 `results.keys()` 和 `results["pts_filename"]`
6. 单步执行后观察 `results["points"]`
7. 再去找 `Compose`，看这一类 transform 是怎么被依次调用的
8. 最后再去看模型 `forward()`

---

## 16. 一句话理解你当前的位置

你现在不是在 debug 神经网络“前向传播”，而是在 debug “数据进入神经网络之前，怎么被一站一站加工”的过程。

这个思路一旦转过来，后面就会清楚很多。

---

## 17. 按 `configs/nuscenes/default.yaml` 只看图片处理流程

如果你现在的目标是：

> 不管点云、不管 bbox、不管 occupancy，只想看图片数据在 pipeline 里怎么流动

那就只需要盯住 `train_pipeline` / `test_pipeline` 里和图片有关的 class。

### 17.1 训练阶段图片主线

按 `configs/nuscenes/default.yaml`，训练时图片相关步骤按顺序是：

1. `LoadMultiViewImageFromFiles`
2. `ImageAug3D`
3. `GlobalRotScaleTrans`
4. `ImageNormalize`
5. `GridMask`
6. `DefaultFormatBundle3D`
7. `Collect3D`

其中：

- `GlobalRotScaleTrans` 不是“图片像素处理主角”，但会影响图像和点云之间的几何关系
- `GridMask` 只在训练里有，测试里没有

### 17.2 测试阶段图片主线

测试时顺序是：

1. `LoadMultiViewImageFromFiles`
2. `ImageAug3D`
3. `GlobalRotScaleTrans`
4. `ImageNormalize`
5. `DefaultFormatBundle3D`
6. `Collect3D`

---

## 18. 每一步最该关注什么

### 18.1 `LoadMultiViewImageFromFiles`

作用：

- 从 `results["image_paths"]` 里取多视角图片路径
- 把图片读进来
- 存进 `results["img"]`
- 初始化一些尺寸相关字段

你最该关注：

- `results["image_paths"]`
- `results["img"]`
- `len(results["img"])`
- `type(results["img"][0])`
- `results["img"][0].size`
- `results["img_shape"]`
- `results["ori_shape"]`
- `results["pad_shape"]`

这一层的典型状态是：

- `results["img"]` 是 `list[PIL.Image]`
- 每张图还不是 tensor
- 当前尺寸可能是原始尺寸，比如 `(1600, 900)`

### 18.2 `ImageAug3D`

作用：

- 做图片几何增强
- 典型包括 resize、crop、flip、rotate
- 生成 `img_aug_matrix`

这是图片 pipeline 里最关键的一步之一。

你最该关注：

- `new_imgs`
- `data["img"]`
- `type(data["img"][0])`
- `data["img"][0].size`
- `flip`
- `data["img_aug_matrix"]`

特别注意：

- 不要只看局部变量 `img`
- `img` 往往只是循环中的“当前输入图片”
- 真正的输出通常在 `new_imgs` 或 `data["img"]`

也就是说，看到类似这种代码时：

```python
data["img"] = new_imgs
data["img_aug_matrix"] = transforms
```

要优先看：

```python
new_imgs[0].size
data["img"][0].size
type(data["img"][0])
```

如果当前图像仍是 `PIL.Image`，那就看 `.size`，不要看 `.shape`。

### 18.3 `GlobalRotScaleTrans`

作用：

- 做全局几何增强
- 主要影响点云和 3D 几何关系
- 通常不是修改图片像素的主步骤

你最该关注：

- `results["img"]` 有没有明显变化
- `results["lidar_aug_matrix"]`

如果你当前只想理解图片像素流，这一步可以轻看。

### 18.4 `ImageNormalize`

作用：

- 做图像归一化
- 常见是减均值、除方差
- 可能会把图片从 PIL 转成 numpy

你最该关注：

- `type(results["img"][0])`
- `results["img"][0].shape`
- `results.get("img_norm_cfg")`

典型现象是：

- shape 可能不变
- dtype 和数值范围变了
- 可能从 `PIL.Image` 变成 `np.ndarray`

### 18.5 `GridMask`

作用：

- 在训练时对图片做网格遮挡增强

你最该关注：

- `results["img"]` 类型
- `results["img"]` shape

这一步通常：

- 会改图像内容
- 不太会改图像 shape

### 18.6 `DefaultFormatBundle3D`

作用：

- 把前面比较“松散”的图片格式整理成更接近模型输入的格式

这是另一个非常关键的步骤。

你最该关注：

- `type(results["img"])`
- `results["img"].shape`

这里很可能会看到图片数据从：

- `list[PIL.Image]`
- 或 `list[np.ndarray]`

变成：

- 堆叠后的 array
- 或 tensor

这里通常是图片 shape 最值得认真看的地方。

### 18.7 `Collect3D`

作用：

- 选择最终送给模型的 key
- 同时打包 meta 信息

你最该关注：

- 最终有没有 `img`
- `img` 是什么类型
- `meta_keys` 里和图片相关的矩阵是否被保留

当前配置里，和图片强相关的 `meta_keys` 有：

- `camera_intrinsics`
- `camera2ego`
- `lidar2camera`
- `camera2lidar`
- `lidar2image`
- `img_aug_matrix`

---

## 19. 这条图片流真正该怎么理解

如果把图片数据单独拿出来看，可以把它理解成下面这条主线：

```text
图片路径
-> 多视角 PIL 图片列表
-> 几何增强后的图片
-> 归一化后的图片
-> 格式整理后的模型输入
-> 被 Collect3D 打包给模型
```

你调试时，不要总想着模型里的 `x.shape`，而要想着：

> `results["img"]` 现在到底是路径、PIL、numpy，还是 tensor？

---

## 20. 当前最推荐的观察变量

如果你现在只追图片流，推荐统一观察这些表达式：

```python
type(results["img"])
len(results["img"]) if isinstance(results["img"], list) else None
type(results["img"][0]) if isinstance(results["img"], list) and len(results["img"]) > 0 else None
results["img"][0].size if isinstance(results["img"], list) and len(results["img"]) > 0 and hasattr(results["img"][0], "size") else None
results["img"][0].shape if isinstance(results["img"], list) and len(results["img"]) > 0 and hasattr(results["img"][0], "shape") else None
results["img"].shape if hasattr(results["img"], "shape") else None
results.get("img_shape")
results.get("ori_shape")
results.get("pad_shape")
results.get("img_aug_matrix")
results.get("scale_factor")
```

---

## 21. 关于“为什么看起来没变化”的一个常见误区

在 `ImageAug3D` 里，很多人会看到：

```python
img: <PIL.Image ... size=1600x900>
```

然后以为增强没有生效。

但这往往只是因为你看到的是：

- 循环中的局部变量 `img`

而不是：

- 变换后的 `new_imgs`
- 或最终写回去的 `data["img"]`

所以在 `ImageAug3D` 里，如果你想判断是否真的变了，优先看：

```python
new_imgs[0].size
data["img"][0].size
type(data["img"][0])
```

不要只盯着局部变量 `img`。

---

## 22. 你现在最值得重点看的三个 class

如果不想把每一步都深挖，最建议重点看这三个：

1. `LoadMultiViewImageFromFiles`
2. `ImageAug3D`
3. `DefaultFormatBundle3D`

原因是：

- 第一个告诉你图片最初是怎么进来的
- 第二个告诉你图片怎么被几何增强
- 第三个告诉你图片什么时候真正变成模型输入格式

---

## 23. 当前已经确认的数据前处理结果

下面这部分是已经通过实际断点确认过的，不是猜测。

### 23.1 图片主线：从读图到最终打包

按 `configs/nuscenes/default.yaml` 当前这条图片处理链，已经确认的变化是：

#### 第 1 步：`LoadMultiViewImageFromFiles`

输入：

- 6 路相机图片路径

输出：

- `results["img"]` 是 `list[PIL.Image]`
- `len(results["img"]) == 6`
- 单张图尺寸：

```python
results["img"][0].size == (1600, 900)
```

注意：

- 这里的 `PIL.Image.size` 顺序是 `(W, H)`

同时初始化了这些字段：

```python
results["img_shape"] == (1600, 900)
results["ori_shape"] == (1600, 900)
results["pad_shape"] == (1600, 900)
```

#### 第 2 步：`ImageAug3D`

作用：

- 对图片做几何增强
- 典型包括 resize / crop / flip / rotate
- 同时生成 `img_aug_matrix`

实际断点确认结果：

- `results["img"]` 仍然是 `list[PIL.Image]`
- 单张图尺寸变成：

```python
results["img"][0].size == (704, 256)
```

注意：

- 这里 `(704, 256)` 仍然是 `(W, H)`
- 这正对应配置里的目标图像尺寸 `image_size: [256, 704]`

一个重要提醒：

- 在 `ImageAug3D` 里不要只盯局部变量 `img`
- 局部变量 `img` 往往是循环中的输入图片
- 真正的输出要看 `new_imgs` 或 `data["img"]`

#### 第 3 步：`ImageNormalize`

作用：

- 把图片从 `PIL.Image` 转成 `torch.Tensor`
- 执行归一化

实际断点确认结果：

- `results["img"]` 变成 `list[torch.Tensor]`
- `len(results["img"]) == 6`
- 单张图 shape：

```python
results["img"][0].shape == torch.Size([3, 256, 704])
```

注意：

- 这里已经不是 `(W, H)` 了
- tensor 的 shape 顺序是 `(C, H, W)`

所以：

- `PIL.Image.size == (704, 256)`
- 对应 tensor 就是：

```python
(3, 256, 704)
```

#### 第 4 步：`GridMask`

作用：

- 训练时对图片施加网格遮挡增强

当前理解：

- 它主要改图片内容
- 通常不改图片 shape

所以在 shape 追踪上，可以先记成：

```python
输入单张图 shape == (3, 256, 704)
输出单张图 shape == (3, 256, 704)
```

#### 第 5 步：`DefaultFormatBundle3D`

作用：

- 把前面分散的多视角图片 tensor 整理成统一格式
- 先 `stack`，再包装成 `DataContainer`

关键代码形态：

```python
results["img"] = DC(torch.stack(results["img"]), stack=True)
```

实际断点确认结果：

- 原来：

```python
results["img"]` 是长度为 6 的 list
每个元素 shape == (3, 256, 704)
```

- 执行后：

```python
type(results["img"]) == DataContainer
type(results["img"].data) == torch.Tensor
results["img"].data.shape == torch.Size([6, 3, 256, 704])
```

这一步的含义：

- `6`：6 个相机视角
- `3`：RGB 通道
- `256`：高
- `704`：宽

注意：

- 这里还不是 batch 维
- 这是“一个 sample 内部的 6 视角”

#### 第 6 步：`Collect3D`

作用：

- 不再改变图片 shape
- 负责把图片和其他字段一起打包成交给模型的最终输入结构

图片这条线在这一步的关键结论：

- `img` 以 `DataContainer` 形式进入最终输出
- 其中图片主张量 shape 已经固定为：

```python
(6, 3, 256, 704)
```

---

## 24. 当前图片前处理流程的一句话总结

可以把当前已经确认的图片前处理链写成：

```text
图片路径
-> list[PIL.Image]，单张 size=(1600,900)
-> list[PIL.Image]，单张 size=(704,256)
-> list[torch.Tensor]，单张 shape=(3,256,704)
-> DataContainer(torch.Tensor)，整体 shape=(6,3,256,704)
-> 被 Collect3D 打包进入最终模型输入
```

---

## 25. 为什么 `img_shape` 看起来和真实图片不一致

调试过程中已经观察到：

- 真实图片在 `ImageAug3D` 后已经变成 `(704, 256)`
- 但有些时候：

```python
results["img_shape"] == (1600, 900)
results["ori_shape"] == (1600, 900)
results["pad_shape"] == (1600, 900)
```

这说明在当前项目里：

- `img_shape / ori_shape / pad_shape` 不一定会在每个 transform 后同步更新
- 真正可靠的当前图片格式，应优先看：
  - `results["img"]`
  - `results["img"][0].size`
  - `results["img"][0].shape`
  - `results["img"].data.shape`

换句话说：

- shape 追踪时，优先相信真实图片对象本身
- 不要只相信这些历史记录字段

---

## 26. `Collect3D` 之后当前最终保留下来的主要字段

当前调试中，已经确认最终输出里至少保留了这些内容。

### 26.1 主输入

- `img`
- `points`

其中：

```python
img.data.shape == (6, 3, 256, 704)
```

### 26.2 检测 / 3D 标注

- `gt_bboxes_3d`
- `gt_labels_3d`

### 26.3 occupancy 监督

- `voxel_semantics`
- `mask_lidar`
- `mask_camera`

### 26.4 几何与增广相关 meta

- `camera_intrinsics`
- `camera2ego`
- `camera_ego2global`
- `lidar2ego`
- `lidar2camera`
- `camera2lidar`
- `lidar2image`
- `img_aug_matrix`
- `lidar_aug_matrix`
- `occ_aug_matrix`
- `metas`

其中如果后面要继续追“图像如何投影到 3D / BEV”，最关键的通常是：

- `camera_intrinsics`
- `lidar2image`
- `img_aug_matrix`

---

## 27. 当前对 `meta` 的理解

可以先把 `meta` 理解成：

```text
不是主数据本身，而是主数据的说明书、几何关系和增广记录
```

也就是说：

- `img` 是模型真正要“看”的图片张量
- `meta` 是模型理解“这张图如何对应真实空间”所需要的辅助信息

在这个项目里，`meta` 尤其重要，因为后面多相机图像特征要和激光雷达 / BEV / 3D 空间对齐。

---

## 28. 下一阶段：从 `daocc_occ3d_nus_w_mask.yaml` 进入模型内部

前面已经把：

- `configs/nuscenes/default.yaml`
- 数据集 pipeline
- 图片前处理

这部分看清楚了。

接下来如果要继续 debug 模型内部，建议切换到这份真正训练用的配置：

```text
configs/nuscenes/occ3d/daocc_occ3d_nus_w_mask.yaml
```

原因是：

- `default.yaml` 更像基础数据与 pipeline 定义
- `daocc_occ3d_nus_w_mask.yaml` 里真正定义了当前实验要用的模型结构
- 后续看 backbone / neck / vtransform / decoder / head，应该以这份配置为准

可以先把这一阶段理解成：

```text
数据进入模型前
-> BEVFusion.forward
-> 相机与激光特征提取
-> 多模态融合
-> decoder
-> object head / occ head
```

---

## 29. `BEVFusion` 里最推荐的断点位置

文件：

```text
DAOcc/mmdet3d/models/fusion_models/bevfusion.py
```

这个文件里函数很多，但如果刚开始追 shape，不需要全打断点。

最推荐先保留下面这些断点。

### 29.1 第一优先级断点

#### 1. `forward`

目的：

- 确认模型真正收到的输入长什么样
- 看 `img`、`points`、各类几何矩阵是不是都已经进来了

最该看：

```python
type(img)
img.shape
len(points)
type(points[0])
camera_intrinsics.shape if hasattr(camera_intrinsics, "shape") else type(camera_intrinsics)
img_aug_matrix.shape if hasattr(img_aug_matrix, "shape") else type(img_aug_matrix)
```

如果这一层已经过了 dataloader/collate，通常你应该开始看到：

```python
img.shape == (B, 6, 3, 256, 704)
```

#### 2. `forward_single`

目的：

- 这是单次样本/单次前向的真正主逻辑入口
- 大部分 shape 变化会从这里开始

最该看：

- `img.shape`
- `len(points)`
- 当前 `features` 列表是怎么被填充的

#### 3. `extract_camera_features`

目的：

- 这是图片从输入张量变成相机特征的核心路径

这里最值得认真看。

最该关注的几个位置：

```python
B, N, C, H, W = x.size()
x = x.view(B * N, C, H, W)
x = self.encoders["camera"]["backbone"](x)
x = self.encoders["camera"]["neck"](x)
x = x.view(B, int(BN / B), C, H, W)
x = self.encoders["camera"]["vtransform"](...)
```

你要记录的关键 shape 通常是：

1. 进入函数时：

```python
(B, 6, 3, 256, 704)
```

2. 拉平成多视角 batch 后：

```python
(B*6, 3, 256, 704)
```

3. backbone 输出后：

- 空间分辨率下降
- 通道数上升

4. neck 输出后：

- 仍是图像特征
- 通道数通常被统一到配置里的某个值

5. reshape 回多视角后：

```python
(B, 6, C, h, w)
```

6. `vtransform` 输出后：

- 开始进入 BEV / 3D 特征空间
- 这一步通常会更接近别人画图里的中间特征

### 29.2 第二优先级断点

#### 4. `extract_lidar_features`

目的：

- 看点云特征怎么提取

如果你当前先以图片流为主，这一步可以先轻看。

#### 5. `self.fuser` 调用前后

目的：

- 看 camera feature 和 lidar feature 怎么融合

你要关注：

- camera branch 输出 shape
- lidar branch 输出 shape
- 融合后 shape

#### 6. `self.decoder["backbone"]` 和 `self.decoder["neck"]` 前后

目的：

- 看融合后的 BEV 特征如何进一步编码
- 这一步常常会出现多尺度输出

### 29.3 第三优先级断点

#### 7. 各个 head 调用处

重点是：

- `object` head
- `occ` head

尤其如果你后面想对上 occupancy 输出，就要继续看：

- `mmdet3d/models/heads/occ/bev_occ_head.py`

---

## 30. 推荐的最小断点方案

如果不想一开始就被太多断点打断，建议只开这 4 个：

1. `BEVFusion.forward`
2. `BEVFusion.forward_single`
3. `BEVFusion.extract_camera_features`
4. `BEVOCCHead2D.forward`

这样就足够先把：

```text
输入图片
-> 相机特征
-> BEV / volume 特征
-> occupancy 头输出
```

这条主线串起来。

---

## 31. 从现在开始该怎么看 shape

进入模型后，推荐你把观察重点分成 4 段。

### 第 1 段：模型输入

重点看：

- `img`
- `points`
- `camera_intrinsics`
- `lidar2image`
- `img_aug_matrix`

目标是确认：

- 图片张量是否已经是 `(B, 6, 3, 256, 704)`

### 第 2 段：相机特征提取

重点看：

- backbone 前后
- neck 前后
- reshape 前后

目标是确认：

- 从原图到多视角 feature map 的 shape 怎么变

### 第 3 段：视图变换与融合

重点看：

- `vtransform`
- `extract_lidar_features`
- `fuser`

目标是确认：

- 图像特征什么时候进入 BEV / 3D 空间
- 和 lidar 特征怎么对齐

### 第 4 段：decoder 与 head

重点看：

- decoder backbone
- decoder neck
- object head
- occ head

目标是确认：

- 最终输出给检测和 occupancy 分支的 shape 是什么

---

## 32. 当前阶段最建议记录的 shape 检查点

后面每到一个关键位置，建议至少记录下面这些 shape：

1. `img` 进入 `forward` 时
2. `x.view(B * N, C, H, W)` 之后
3. camera backbone 输出
4. camera neck 输出
5. `vtransform` 输出
6. lidar backbone 输出
7. fuser 输出
8. decoder backbone 输出
9. decoder neck 输出
10. occ head 输入与输出

如果能把这 10 个点记下来，后面基本就能把整条主线画出来。

---

## 33. 当前阶段的一句话行动建议

从现在开始，`default.yaml` 这一段可以先告一段落。

后续调试主线建议切到：

```text
configs/nuscenes/occ3d/daocc_occ3d_nus_w_mask.yaml
```

并以：

```text
BEVFusion.forward
-> forward_single
-> extract_camera_features
```

作为新的主入口继续追。

---

## 34. 后续 debug 的记录规则

从这一阶段开始，文档记录不再只写 shape，还要同时记录：

1. 当前是从哪个文件/函数进入的
2. 当前函数的作用是什么
3. 这个函数里调用了哪些关键子函数
4. 下一步应该点进哪个子函数继续看
5. 当前这一层最关键的 shape 变化是什么

也就是说，后面记录的风格要从：

```text
某个 class 输入输出 shape 是什么
```

升级成：

```text
从哪个入口进入
-> 当前 forward / 函数负责什么
-> 它调用了哪个关键子函数
-> 为什么要点进去看这个子函数
-> 进去之后又看到了什么
```

---

## 35. 后续记录的总入口：从 `train.py` 开始

因为实际运行的是训练脚本，所以后面的调用链记录应当从：

```text
tools/train.py
```

开始。

原因是：

- 这是程序的实际启动入口
- 这里会完成 config 加载、dataset/model 构建、训练流程启动
- 这里最终会走到 `model(...)`

所以后续记录要遵循这个原则：

```text
先从 train.py 开始
-> 找到 model 被调用的位置
-> 点进模型 forward
-> 再沿着关键子函数继续追
```

---

## 36. 后续文档记录模板

后面每往下 debug 一层，建议按下面这个模板记录。

### 36.1 基础模板

```text
文件：
函数：

这一层的作用：

关键输入：

关键输出：

这一层调用了哪些关键子函数：

下一步进入哪个函数：

为什么要进入这个函数：

这一层观察到的关键 shape：
```

### 36.2 例子：后面记录 `forward` 时的写法

```text
文件：mmdet3d/models/fusion_models/bevfusion.py
函数：forward

这一层的作用：
模型总入口，接收 dataloader 整理好的 img / points / 几何矩阵 / 标签等输入。

关键输入：
img, points, camera2ego, lidar2ego, lidar2camera, lidar2image,
camera_intrinsics, camera2lidar, img_aug_matrix, lidar_aug_matrix, metas ...

关键输出：
当前不直接做复杂计算，而是继续调用 forward_single。

这一层调用了哪些关键子函数：
forward_single

下一步进入哪个函数：
forward_single

为什么要进入这个函数：
真正的单次前向逻辑在这里展开，后续特征提取和融合都从这里开始。

这一层观察到的关键 shape：
img.shape = ...
```

---

## 37. 后续记录时要特别补充“函数作用”

从这一阶段开始，不只记录 shape，还要记录每个函数到底“负责什么”。

例如后面如果进入：

- `BEVFusion.forward`
- `BEVFusion.forward_single`
- `extract_camera_features`
- camera backbone 的 `forward`
- neck 的 `forward`
- `vtransform`
- `BEVOCCHead2D.forward`

都要补一句：

```text
这个函数的作用是什么
```

例如：

- `forward`：模型总入口，负责把输入分发到单次前向逻辑
- `forward_single`：组织 camera / lidar 分支、融合、decoder 和 head
- `extract_camera_features`：把多相机图像输入变成相机特征并送入视图变换
- backbone 的 `forward`：提取图像层级特征
- neck 的 `forward`：融合/整理 backbone 多尺度特征
- `vtransform`：把图像特征投影到 BEV / 3D 空间
- `BEVOCCHead2D.forward`：把 BEV 特征映射成 occupancy 预测

---

## 38. 后续进入子函数时的记录规则

如果在某个函数里看到了关键调用，例如：

```python
x = self.encoders["camera"]["backbone"](x)
```

后面的记录不应只写：

```text
进入了 backbone.forward
```

而应写成：

```text
在 extract_camera_features 中，A 函数调用了 B 函数：
self.encoders["camera"]["backbone"](x)

下一步点进去看 backbone.forward，
因为这里是图像从原始输入张量变成深层视觉特征的第一步。
```

同理，如果在 `forward` 中看到了：

```python
outputs = self.forward_single(...)
```

就要写：

```text
在 forward 中，核心工作并不展开，而是调用了 forward_single。
所以下一步进入 forward_single，看模型单次前向的真正执行逻辑。
```

---

## 39. 从现在开始推荐的调用链记录顺序

后续建议按下面这个顺序持续记录：

1. `tools/train.py`
2. 训练流程里 `model(...)` 被调用的位置
3. `BEVFusion.forward`
4. `BEVFusion.forward_single`
5. `extract_camera_features`
6. camera backbone 的 `forward`
7. camera neck 的 `forward`
8. `vtransform`
9. lidar 分支相关函数
10. `fuser`
11. decoder backbone / neck
12. `BEVOCCHead2D.forward`

这样记录的好处是：

- 你能看到完整调用链
- 不会只盯着局部 shape
- 后面回头看时，能知道“为什么当时要点进这个函数”

---

## 40. 当前阶段的执行原则

从现在开始，每继续 debug 一步，都尽量同时回答这 4 个问题：

1. 这个函数在整个模型里负责什么？
2. 这个函数调用了哪个关键子函数？
3. 为什么下一步要点进这个子函数？
4. 这一步最关键的 shape 变化是什么？

后面的文档会按这个标准持续补充。

---

## 41. 当前信息总汇

这一节把目前已经 debug 出来的信息按类别重新整理，方便后续继续追代码时快速回看。

---

## 42. 当前主要配置结论

### 42.1 数据前处理主配置

当前已经重点看过：

```text
configs/nuscenes/default.yaml
```

这份配置主要负责：

- 数据集定义
- train/test pipeline
- `samples_per_gpu`
- 图片预处理主线

当前已调整：

```yaml
samples_per_gpu: 1
```

这样做的原因是：

- 之前 `samples_per_gpu: 4`
- 图像进入相机 backbone 时一次会变成 `4 * 6 = 24` 张图
- 调试状态下很容易 OOM

### 42.2 当前模型主配置

后续模型内部调试应以：

```text
configs/nuscenes/occ3d/daocc_occ3d_nus_w_mask.yaml
```

为主。

原因：

- 这份配置真正定义了当前实验用的模型结构
- 后面 debug backbone / neck / vtransform / decoder / head 时，应该以它为准

### 42.3 deprecated 配置问题与修复

已检查并修改：

```text
configs/nuscenes/occ3d/deprecated/daocc_occ3d_nus_w_mask.yaml
```

原问题：

- camera backbone 实际使用 `ResNet50`
- 但 neck 仍配置成匹配 Swin 的：

```yaml
in_channels: [384, 768, 1536]
```

已修正为：

```yaml
in_channels: [512, 1024, 2048]
```

原因：

- `ResNet50` 对应多尺度输出通道更接近 `[512, 1024, 2048]`
- 原配置会导致 `GeneralizedLSSFPN` 出现通道不匹配

---

## 43. 数据前处理阶段总结

这一部分对应的是：

```text
dataset / pipeline / Collect3D
```

也就是“数据进入模型前”。

### 43.1 图片数据主线

当前已确认的图片处理链：

```text
图片路径
-> list[PIL.Image]，单张 size=(1600,900)
-> list[PIL.Image]，单张 size=(704,256)
-> list[torch.Tensor]，单张 shape=(3,256,704)
-> DataContainer(torch.Tensor)，整体 shape=(6,3,256,704)
-> 被 Collect3D 打包进入最终模型输入
```

### 43.2 各关键 class 的作用

#### `LoadMultiViewImageFromFiles`

作用：

- 读取 6 路相机图像
- 把图片路径变成 `list[PIL.Image]`

关键观察：

```python
len(results["img"]) == 6
results["img"][0].size == (1600, 900)
```

#### `ImageAug3D`

作用：

- 做图像几何增强
- resize / crop / flip / rotate
- 生成 `img_aug_matrix`

关键观察：

```python
results["img"][0].size == (704, 256)
```

说明：

- 配置 `image_size: [256, 704]`
- 对应 `PIL.Image.size == (704, 256)`

#### `ImageNormalize`

作用：

- `PIL.Image -> torch.Tensor`
- 做标准化

关键观察：

```python
results["img"][0].shape == (3, 256, 704)
```

#### `GridMask`

作用：

- 训练时做网格遮挡增强

当前理解：

- 改像素内容
- 通常不改 shape

#### `DefaultFormatBundle3D`

作用：

- 把 6 个视角的单张 tensor 堆叠
- 再包装成 `DataContainer`

关键观察：

```python
results["img"].data.shape == (6, 3, 256, 704)
```

#### `Collect3D`

作用：

- 不再改变图片 shape
- 负责把主数据和 meta 打包成交给模型的最终输入结构

### 43.3 `meta` 的当前理解

可以先把 `meta` 理解成：

```text
主数据的说明书 + 几何关系 + 增广记录
```

它不是主数据本身，但后面做相机到 BEV / 3D 投影时非常关键。

当前已经确认和图片分支最相关的 meta 主要包括：

- `camera_intrinsics`
- `camera2ego`
- `lidar2camera`
- `camera2lidar`
- `lidar2image`
- `img_aug_matrix`
- `lidar_aug_matrix`
- `occ_aug_matrix`

---

## 44. 从训练入口到模型入口的调用链

当前已经明确的主调用链是：

```text
tools/train.py
-> train_model(...)
-> runner.run(...)
-> train_step(...)
-> self(**data)
-> BEVFusion.forward
-> BEVFusion.forward_single
```

这说明：

- `train.py` 是程序实际启动入口
- `BEVFusion.forward` 只是单个 batch 的模型前向入口
- 它结束不等于整个训练结束

---

## 45. `BEVFusion` 当前职责划分

### 45.1 `forward`

作用：

- 模型总入口
- 接收 dataloader 整理好的 batch
- 把输入继续交给 `forward_single`

关键观察：

进入模型后，图片张量从单个 sample 的：

```python
(6, 3, 256, 704)
```

变成 batch 形式：

```python
img.shape == (4, 6, 3, 256, 704)
```

这里的 `4` 对应之前的 `samples_per_gpu: 4`。

### 45.2 `forward_single`

作用：

- 单次前向的总调度器
- 组织 camera 分支和 lidar 分支提特征
- 把两路特征放进 `features`
- 做 `fuser`
- 经过 `decoder`
- 送给 object / occ 等 head

一句话总结：

```text
forward_single 负责组织整条单次前向主流程
```

### 45.3 `extract_camera_features`

作用：

- 从多相机图像中提取 2D 视觉特征
- 再通过 `vtransform` 映射成相机分支的 BEV 特征

一句话总结：

```text
extract_camera_features = 图像特征提取 + 映射到 BEV 特征空间
```

---

## 46. 图片分支当前已确认的关键 shape

### 46.1 进入模型时

```python
img.shape == (4, 6, 3, 256, 704)
```

含义：

- `4`：batch size
- `6`：相机数
- `3`：RGB
- `256,704`：输入图像大小

### 46.2 `extract_camera_features` 入口

初始：

```python
x.shape == (4, 6, 3, 256, 704)
```

拉平后：

```python
x = x.view(B * N, C, H, W)
x.shape == (24, 3, 256, 704)
```

含义：

- `4 * 6 = 24`
- 这样做是为了适配 2D backbone 的输入格式 `(batch, C, H, W)`

### 46.3 camera backbone 输出

当前已确认：

- backbone 不是返回单个 tensor
- 而是返回多尺度特征 `tuple/list`

调试中已经观察到类似特征：

```python
x[0].shape == (24, 512, 32, 88)
```

说明：

- 原始图像已经变成深层视觉特征
- 空间尺寸变小
- 通道数变大

### 46.4 neck 后重新组织

neck 后代码会把特征重新恢复成：

```python
(B, N, C, h, w)
```

当前观察到 `vtransform` 调用前：

```python
x.shape == (4, 6, 256, 32, 88)
```

### 46.5 `vtransform` 输出

当前最关键的观察结果之一：

调用前：

```python
x.shape == (4, 6, 256, 32, 88)
```

调用后：

```python
x.shape == (4, 1280, 180, 180)
```

这说明：

- 多相机图像特征已经被映射到统一的 BEV 特征平面
- 相机维 `6` 被融合/投影掉
- 空间变成 `180 x 180` 的 BEV 网格

### 46.6 回到 `forward_single`

`extract_camera_features` 返回值，会赋给：

```python
feature = self.extract_camera_features(...)
```

也就是说：

- `extract_camera_features` 的返回值
- `forward_single` 里的 camera `feature`

本质上是同一个结果。

当前可记为：

```python
camera feature.shape == (4, 1280, 180, 180)
```

随后它会被：

```python
features.append(feature)
```

加入 `features` 列表，供后续与 lidar 特征融合使用。

---

## 47. 当前碰到过的主要问题

### 47.1 neck 通道不匹配

报错特征：

```text
expected 2304 channels, but got 3072
```

根因：

- 配置里 neck 按 Swin 的通道数写
- 实际 backbone 用的是 ResNet50

处理：

- 已在 deprecated 配置中把 `in_channels` 改成匹配 ResNet 的值

### 47.2 CUDA / MAGMA 异常

出现过一次更底层的：

```text
magma_queue_create_from_cuda_internal ... assertion failed
```

当前判断：

- 更偏 CUDA / MAGMA / GPU 运行环境异常
- 不像纯模型结构错误

### 47.3 OOM

出现过明确的：

```text
CUDA out of memory
```

当前判断：

- 在 `camera backbone (ResNet50)` 前向阶段就可能爆显存
- 调试状态下更容易因为缓存和中间激活导致显存不足

当前已采取的缓解措施：

```yaml
samples_per_gpu: 1
```

---

## 48. 关于 GPU 和进程的当前结论

当前已确认：

- 容器里 `ps aux | grep python` 只能看到当前容器命名空间下可见的 Python 进程
- 不一定能看到所有真实占 GPU 的进程

也就是说：

```text
ps 看不到，不代表 GPU 真空
```

还确认了：

- 有些 `nvidia-smi` 里显示的 PID，在 `ps` 里已经找不到
- 这更像 GPU 残留上下文 / 异常退出后的显存残留
- 普通 `kill` 不一定能清掉

---

## 49. 当前最重要的理解结论

截至目前，可以把整个主线先概括成下面这几条：

1. `default.yaml` 主要负责数据前处理和 dataloader 设置
2. `daocc_occ3d_nus_w_mask.yaml` 才是后续模型结构调试的主配置
3. 图片在进入模型前，最终会被整理成：

```python
(6, 3, 256, 704)
```

4. 图片进入模型后，会先变成：

```python
(B, 6, 3, 256, 704)
```

5. `extract_camera_features` 会把图像特征进一步变成相机分支的 BEV 特征
6. 当前已确认相机分支通过 `vtransform` 后可得到：

```python
(4, 1280, 180, 180)
```

7. `forward_single` 负责组织 camera / lidar 分支、融合、decoder 和 head

---

## 50. 后续最建议继续追的地方

如果继续 debug，最值得往下看的主线是：

1. `extract_lidar_features(points)`
2. `self.fuser(features)`
3. `self.decoder["backbone"](x)`
4. `self.decoder["neck"](x)`
5. `BEVOCCHead2D.forward`

因为这样就能把：

```text
camera feature
-> lidar feature
-> 融合特征
-> decoder 输出
-> occupancy 预测
```

整条线继续补齐。
