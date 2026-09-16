# M²BEV：先统一世界，再共享感知任务

论文：M²BEV: Multi-Camera Joint 3D Detection and Segmentation with Unified Bird’s-Eye View Representation（2022）

## 01 · 从六张照片到统一世界

这篇是 2022 年的《M²BEV: Multi-Camera Joint 3D Detection and Segmentation with Unified Bird’s-Eye View Representation》。如果用费曼法先把它压成一句话，它真正想解决的是：**一辆自动驾驶汽车有六个摄像头，每个摄像头看到的是不同方向的二维图片，但汽车真正需要的不是六份彼此独立的“照片理解”，而是一张统一的、以自己为中心的三维世界地图；而且这张地图最好能同时服务“哪里有车”和“哪里能开”这两类任务。**M²BEV 因此先把六个相机的二维特征统一搬到自车坐标系的三维空间，再压成一张 BEV 表示，然后在这同一张 BEV 上同时做 3D 目标检测和道路/车道分割。它真正重要的地方不是某个检测头，而是那个后来越来越常见的思想：**不同相机、不同任务，不要各自维护一个世界，先建立一份统一的 BEV 世界表示。**论文在 nuScenes 上的最好单模型结果达到 42.5 mAP 的 3D 检测和 57.0 mIoU 的 BEV 分割。([arXiv][1])

## 02 · 多相机与多任务为什么需要统一坐标

先想象传统多摄像头自动驾驶系统是什么样。前摄像头跑一个检测器，左前摄像头再跑一个，右前、后方也各自跑，最后还得把六个相机检测到的车融合起来、去重；与此同时，道路分割可能又是另一套完全独立的网络。这样的问题非常直观：**大家看的明明是同一个现实世界，却在六套二维坐标系、多个任务网络里重复理解。**更麻烦的是，一辆跨越两个相机视野边界的车可能被重复检测，而检测结果最后再做后处理融合通常不是端到端可学习的。M²BEV 的作者认为，更合理的做法是先统一坐标系：不管物体来自前相机还是后相机，最终都放进“以自车为中心，前后左右各多少米”的三维空间里，然后检测、分割这些任务都从这个统一世界出发。([arXiv][2])

## 03 · BEV：所有任务共用一张地图

这里的 BEV，也就是 Bird’s-Eye View，可以继续用“作战地图”理解。六台摄像头像六个观察员，各自在说“我这里看到一辆车”“我这里看到车道线”；M²BEV 不希望指挥官一直看六块屏幕，而是让所有人把观察结果标在同一张俯视地图上。这样一来，3D 检测关心的是“地图上这个位置有没有汽车、汽车多大、朝哪边”，地图分割关心的是“地图上这一块是不是可行驶区域、这里是不是车道边界”。两个任务虽然输出完全不同，却共享同一个空间坐标和同一批 BEV features，所以大部分计算可以共用。论文把这称为 unified BEV representation，也是 M²BEV 这个名字里的第二个 “M”——multi-view 加 multi-task。([arXiv][2])

## 04 · 不预测深度，先沿射线填入特征

真正困难的问题来了：**只有二维照片，没有 LiDAR，怎么知道图像里的一个像素应该落在三维世界哪个深度？**这本来是 camera-only 3D perception 最麻烦的地方。Lift-Splat-Shoot 这类方法会给每个像素预测一个离散深度分布，相当于说“这个像素有 20% 可能在 10 米、30% 在 12 米……”，再把特征沿深度抬进三维空间。M²BEV 做了一个非常大胆、甚至可以说“故意粗糙”的简化：**不预测深度分布，直接假设一个像素沿着它的相机射线，在所有可能体素里都填上同样的图像特征。**也就是说，如果不知道一个红色车的特征到底在 10 米还是 20 米，就暂时沿整条射线“涂一遍”，后面的网络再根据六个相机和任务监督自己把有用的三维结构整理出来。([arXiv][2])

## 05 · 用投影简化换取计算预算

这个做法从几何上当然没有真正解决深度歧义，但它换来了巨大的计算优势。LSS 要为每个像素显式展开几十个深度 bin，内存消耗很大；M²BEV 的投影本身没有可学习参数，也不用存一个大规模 depth distribution，所以论文报告 LSS 的 GPU memory 大约是它的 3 倍。正因为省下了这部分内存，M²BEV 可以使用更大的 ResNet/ResNeXt backbone 和原始 nuScenes 接近 1600×900 的高分辨率输入，而不是为了省显存把图片缩得很小。([arXiv][2]) 费曼式地说就是：**作者没有先花很多钱搞清楚“每个观察到底多远”，而是先把所有可能位置都写进地图，再让后面的网络自己消歧；省下来的计算预算则拿去看更清楚的图片和用更强的视觉 backbone。**

## 06 · S2C：把高度维放进特征通道

这些沿射线投影出来的特征先形成一个四维 voxel tensor，大致可以想成 \(X\times Y\times Z\times C\)：地面横纵坐标、高度和特征通道。下一步需要把高度 Z 压掉，形成真正二维 BEV feature。最朴素的做法是用 3D convolution，但作者发现太贵，于是提出一个特别简单的 **S2C，Spatial-to-Channel**：直接把高度维 Z reshape 到 channel 里，从 \(X\times Y\times Z\times C\) 变成 \(X\times Y\times(ZC)\)，然后只用成熟而便宜的 2D convolution。([arXiv][2]) 这个技巧听起来甚至不像“论文创新”，但消融很有说服力：三层 naive 3D conv 大约 230 GFLOPs、19ms，而三层 S2C+2D conv 大约 118 GFLOPs、3ms；省下来的预算还可以堆更多 2D 层去细化 BEV。([arXiv][2]) 这也是这篇很“工程派”的特点：不是所有提升都靠更复杂的 Transformer，有时候正确的数据布局本身就是架构创新。

## 07 · 动态匹配：在模糊几何中选择正样本

有了 BEV feature 后，检测头反而很简单。作者直接借用 PointPillars 的 anchor-based 3D detection head，在 BEV 上预测类别、3D box、朝向和速度。但这里又碰到一个问题：LiDAR 的 BEV 几何非常准确，所以传统做法可以用固定 IoU 阈值决定哪个 anchor 是正样本；M²BEV 的 camera-only BEV 因为没有准确深度，几何本来就模糊，如果还拿一把固定 IoU 尺子硬切“正/负”，很多本来可以学好的 anchor 会被误判。于是作者使用 **Dynamic Box Assignment**：先为每个 GT box 找一袋候选 anchor，再结合当前模型的分类置信度和定位质量动态决定哪些是真正的正样本。([arXiv][2]) 这个改动是检测提升最大的单项之一：naive baseline 是 19.7 mAP，换成 dynamic matching 后直接到 27.5，提升 7.8 个点。([arXiv][2]) 用费曼法说就是：**当地图本来就有点模糊时，不要用一条死规则判断“哪个框才算正确”，让模型边学边决定哪些训练样本最可信。**

## 08 · 远距离加权：让模型重视远处

地图分割这边又有另一种困难：**远处在 BEV 地图上占的物理面积和近处一样，但在原始摄像头里却只占很少像素。**比如离车 40 米的一段车道线，在图像里可能只有几像素，训练时特别容易被忽略。于是作者设计了 **BEV Centerness**，名字听起来像“中心权重”，实际做的事情恰恰是让**越远离自车的 BEV 区域权重越大**。权重大致从近处的 1 增到远处的 2，让模型在 loss 里为远距离错误付出更大代价。([arXiv][2]) 消融中，加入它以后 drivable-area / lane segmentation 都继续提升，而且论文观察到距离越远的区域改善越明显。([arXiv][2]) 这背后的道理很朴素：**训练数据在成像上天然偏爱近处，你就需要在损失函数里人为帮远处“提高音量”。**

## 09 · 用二维预训练帮助三维感知

这篇论文里另一个特别有历史意义的发现，是 **2D 预训练非常能帮助 3D 感知**。作者先在 nuImages 的大规模 2D detection 数据上训练 image backbone，然后再拿这个 backbone 去做 nuScenes 的 3D BEV 学习；同时训练 M²BEV 时还额外挂一个 FCOS 风格的 2D detection auxiliary head，用 3D GT box 投影回每个摄像头得到免费的 2D boxes。这个辅助头只在训练时存在，推理时直接删掉，所以没有额外部署成本。([arXiv][2]) 消融结果很明显：dynamic matching 后是 27.5 mAP，加入 nuImages 2D detection pretraining 后到 33.2；而且使用 nuImages 预训练、只用 50% 的 nuScenes 3D 数据，可以达到和 ImageNet 预训练、使用 100% 3D 数据接近的效果。([arXiv][2]) 这其实提前体现了后来 foundation model 很核心的一种思想：**昂贵的 3D 标签可以少一点，只要你先用便宜、大量的 2D 数据学到足够好的视觉表示。**

## 10 · 检测、分割与系统效率

最终效果上，camera-only 的 M²BEV detection-only 在 nuScenes test 达到 42.9 mAP、47.4 NDS，joint 版本是 42.5 mAP、46.5 NDS；当时的 PGD 是 38.6 mAP、44.8 NDS，而 DETR3D 虽然用了额外深度预训练，mAP 是 41.2。([arXiv][2]) BEV segmentation 上，M²BEV segmentation-only 达到 drivable area 77.2 mIoU、lane 40.5，而 Lift-Splat-Shoot 是 72.9 和 19.9，车道线提升尤其明显。([arXiv][2]) 更有意思的是速度：单独把 FCOS3D 检测和 LSS 分割两套系统拼起来只有大约 0.9 FPS，而一个同时输出检测和分割的 M²BEV 大约能做到 4.2 FPS；共享 BEV 后，加第二个任务几乎没有明显增加推理成本。([arXiv][2])

## 11 · 共享计算不等于任务互相促进

不过这里有一个非常值得认真读、而不是被“multi-task”标题带过去的结果：**联合训练并没有让两个任务互相提高。**在 ResNet-50 消融里，单独 detection 是 34.8 mAP，joint 变成 34.0；单独 segmentation mIoU 54.6，joint 变成 52.3。作者自己明确承认，3D detection 和 map segmentation 没有表现出正向 multi-task transfer，反而都轻微下降。([arXiv][2]) 为什么？因为“车在哪里”和“可行驶区域在哪里”虽然都存在同一个空间，但统计关系并不完全一致，例如很多车本来就可以停在不可行驶区域，任务梯度也可能发生冲突。所以 M²BEV 真正证明的是：**共享 BEV 可以极大提高系统效率，不等于共享任务一定提高单任务精度。**这是一条很重要的多任务学习经验。

## 12 · 与 BEVFormer 对照：投影与注意力

如果把 M²BEV 和你刚刚读的 **BEVFormer** 放在一起看，两篇其实非常适合连续读。M²BEV 在 2022 年 4 月先提出一个很清楚的框架：**多摄像头先统一到 dense BEV，再在 BEV 上做多个任务。**它构建 BEV 的方式更直接：利用相机几何，把二维 feature 沿着 ray 填进 3D voxel，再把高度压缩。BEVFormer 随后则换了另一种更“Transformer”的答案：不显式把每条射线都填满，而是在 BEV 平面放 queries，用 spatial cross-attention 主动去相机图里找对应信息，同时还加入 temporal self-attention 读取历史。M²BEV 更像“**先几何投影，再 CNN 整理**”，BEVFormer 更像“**BEV query 主动从图像和历史中取信息**”。两者共同确立的核心却完全一样：**真正有价值的中间表示不是六张图片，而是一份自车中心的世界坐标表示。**

## 13 · 局限：深度、时间与相机标定

M²BEV 的局限也就从这里非常清楚了。第一，它把未知深度简单假设成沿射线均匀分布，虽然便宜，但几何模糊，远处和遮挡场景都会受影响；作者也承认和强 LiDAR 系统相比还有明显差距。([arXiv][2]) 第二，它完全是单帧系统，没有时间记忆，所以速度、遮挡、运动连续性主要只能从单帧视觉里推；论文结尾自己就把 tracking、motion prediction、trajectory forecasting 留给未来。([arXiv][2]) 第三，它非常依赖 camera calibration：小的 extrinsic noise 问题不大，但噪声超过一定程度以后检测和分割都会快速下降。([arXiv][2]) 第四，它的 BEV 是为了 3D 检测和地图分割设计的，还远不是后来 occupancy/world-model 那种真正完整的三维场景状态。

## 14 · 统一世界表示的演化

如果把它放到你最近一直读的“representation”主线里，会发现 M²BEV 其实很有历史意义。你刚读 Dexterity-BEV 时看到的是“机器人不同摄像头、不同身体、不同动作都应该统一到一个 3D/BEV frame”；这里 M²BEV 在更早的自动驾驶场景里已经在做同一种思想：**先把传感器偶然的坐标差异消掉，再让任务共享现实世界坐标。**BEVFormer 把这件事做成 attention，Dexterity-BEV 再把这个思想从 perception 延伸到 action；而今天的 world model 会继续问：“既然已经有统一 BEV 了，能不能预测下一时刻 BEV 会变成什么？”所以可以粗略把这条路线记成：**M²BEV：统一当前世界；BEVFormer：统一当前世界 + 历史；BEV/world model：统一世界 + 未来；Dexterity-BEV：连动作也放进统一世界坐标。**

## 15 · 用一分钟讲给朋友听

如果现在合上论文，用费曼法一分钟给朋友讲，我会这样说：**一辆车有六个摄像头，以前检测汽车、分割车道常常是多套不同系统，各自在二维图片里工作，最后再融合。M²BEV 认为应该先把所有摄像头统一成一张以汽车为中心的俯视地图。它先用 CNN 提取六张图的特征，再根据相机内外参把每个二维像素特征沿着相机射线填进三维 voxel；因为没有深度，它干脆假设射线上各深度都先放同样特征，省掉昂贵的 depth distribution。接着用 Spatial-to-Channel 把高度维压进 channel，用便宜的 2D CNN 得到 BEV feature，然后同一份 BEV 同时接 3D detection 和 road/lane segmentation。作者又用动态 anchor 匹配、远距离 BEV loss 加权、2D detection 预训练和训练期辅助 2D supervision，把 camera-only 性能明显拉高。最后在 nuScenes 上做到 42.5 mAP 和 57.0 mIoU，而且一个同时做检测和分割的系统比把两套独立模型拼起来快约 4 倍。**([arXiv][2])

所以这一篇我建议最后只记一句话：**先统一“世界”，再统一“任务”。** 📖 更技术一点就是：**Multi-camera images → shared 3D voxel → efficient BEV representation → multiple heads。**而和 BEVFormer 连起来读时，最值得看到的不是谁分数高，而是 2022 年这一批工作一起把自动驾驶感知的中心问题从“怎样在每张图片里检测东西”，改成了一个更深的问题：**怎样让车内部维护一份统一、可供所有任务共享的世界表示。**

[1]: https://arxiv.org/abs/2204.05088 "[2204.05088] M$^2$BEV: Multi-Camera Joint 3D Detection and Segmentation with Unified Birds-Eye View Representation"

[2]: https://arxiv.org/html/2204.05088v2 "M2BEV: Multi-Camera Joint 3D Detection and Segmentation with Unified Bird’s-Eye View Representation"
