# BEVFormer：用多视角与时间记忆构建俯视世界

论文：BEVFormer: Learning Bird’s-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers（2022）

## 01 · 从六张画面到一张世界地图

这篇是 2022 年的《BEVFormer: Learning Bird’s-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers》。如果用费曼法先把论文压成一句话，它其实是在解决自动驾驶里一个非常朴素的问题：**车上有六个摄像头，每个摄像头看到的都是自己的二维画面，但真正开车时，我们更想知道的是“以汽车为中心，周围哪里有车、哪里有路、东西在什么位置、正在往哪里动”。那能不能让 Transformer 自动把六个摄像头的画面和过去几帧的记忆，整理成一张统一的俯视地图？** BEVFormer 的答案就是可以。它不先显式预测每个像素的深度再把图像“抬”到三维空间，而是先在鸟瞰图平面上放一张由很多 learnable queries 组成的网格，每个格子主动去不同摄像头里“查询”与自己对应的视觉信息，再去上一时刻的 BEV 记忆里查询时间信息，最后形成当前时刻统一的 BEV feature。论文当时在 nuScenes 上做到 56.9% NDS，比此前最佳 camera-based DETR3D 高 9 个百分点，而且已经接近部分 LiDAR 系统。

## 02 · BEV：用统一坐标理解周围空间

先把 **BEV，Bird’s-Eye View** 讲清楚。想象你坐在车里，前摄像头告诉你“前面有辆白车”，左摄像头告诉你“左边有一辆摩托”，后摄像头又看到另一辆车。每张图的坐标体系都不同，而且透视会严重扭曲距离：远处的车看起来小，近处的车看起来大。如果直接拿六张图片分别做检测，最后再把结果拼起来，模型很难真正建立一个统一的空间概念。BEV 则像从天空正上方看整辆车：自己在地图中心，前方 20 米就是地图前面 20 米，左边 5 米就是左边 5 米。这样做 3D 检测、车道分割、甚至以后做规划都会自然很多，因为模型真正关心的已经不是“图像第 200 个像素是什么”，而是“现实世界里这个位置是什么”。论文也明确把 BEV 看作连接多摄像头空间和时间信息的一种统一表示。

## 03 · 反过来，让地图主动寻找证据

问题是：**六张二维照片怎样变成这张俯视地图？**在 BEVFormer 之前，一个很自然的方法是先估计深度。比如模型看到前摄像头里的汽车，猜它距离摄像头 15 米，再根据相机参数把这个视觉特征投影到鸟瞰地图上。问题是，只要深度估错，后面的三维位置也跟着错，而且错误会层层累积。BEVFormer 的作者因此提出一个很有意思的反方向：**不要先要求模型精确回答“这个像素深度是多少”，而是让 BEV 地图上的位置主动去图像里找证据。**论文明确说，他们希望生成 BEV 时不要严格依赖显式深度估计，而是使用 attention 自适应地聚合视觉特征。

## 04 · BEV Queries：每个格子一个提问者

于是出现了整篇论文第一个核心概念：**BEV Queries。**假设我们在汽车周围铺一张 200×200 的网格，每个格子代表现实世界里固定的一块区域，比如“车左前方 10 米、左边 3 米的位置”。每一个格子都有一个 learnable query，你可以把它想象成一个不断问问题的小侦探：“我负责的这块真实空间里现在有什么？”这些 query 本身一开始什么都不知道，只知道自己在 BEV 平面的哪个位置。经过训练以后，它们学会从摄像头和历史信息里寻找对应证据。论文中 BEVFormer 使用六层 encoder，每层都围绕这些 grid-shaped BEV queries 进行 temporal self-attention、spatial cross-attention 和 feed-forward 更新，最后得到当前时刻的 BEV feature。

## 05 · 空间注意力：几何定位，灵活采样

接下来最关键的是 **Spatial Cross-Attention**。假设 BEV 地图中某个格子代表“汽车右前方 12 米”。这个真实位置理论上可能出现在前摄像头，也可能出现在右前摄像头，但肯定没必要让这个 query 去六张图的所有像素里暴力搜索。BEVFormer 先把 BEV 格子沿高度方向变成一根“柱子”，在不同高度取几个 3D reference points，比如地面、汽车高度等，然后利用已知的相机内外参，把这些三维点投影进各个摄像头。投影落在哪几张图里，就只去那些 camera views 附近取特征。接着使用 deformable attention，不是把整张图所有像素都算一遍，而是在 reference point 附近只采样少量最有价值的位置。 用费曼法说就是：**我想知道地图上这个格子有什么，不需要翻遍六台摄像机的所有画面；我先根据几何知道“它大概应该出现在这些画面的这里”，再让神经网络在附近灵活地找最有用的视觉证据。**

为什么要做 deformable attention，而不是普通 Transformer 的全局 attention？因为自动驾驶图像太大、摄像头又多。如果每个 BEV query 都和六张图每个 feature pixel 相互 attention，显存和计算成本会非常夸张。论文消融结果很能说明这一点：用 global attention 的版本训练显存大约达到 36GB，而且性能反而不如基于局部 deformable attention 的 Spatial Cross-Attention；只盯着精确 reference points 又太死，感受野不足。完整的 local deformable attention 在 nuScenes val 上达到 0.448 NDS，而 points-only 是 0.423，global 是 0.404。 这里真正值得记住的思想是：**几何告诉模型“去哪里找”，attention 再决定“附近到底看什么”。**这比纯手工几何投影更灵活，也比纯全局神经网络搜索更高效。

## 06 · 时间注意力：让地图记得上一帧

但如果论文只有“六个摄像头 → 一张 BEV”，它还不会这么重要。BEVFormer 第二个真正厉害的地方是：**它把时间也直接放进 BEV 表示里。**想象一辆车这一帧被大货车挡住了。如果只看当前六张图片，它可能完全不可见；但上一帧你明明看到它在这里，而且正在向前移动。人开车时当然不会因为一辆车被挡住 0.2 秒就觉得它凭空消失。BEVFormer 因此把上一时刻生成的 BEV feature 保存下来，在下一帧作为 memory 使用。这就是 **Temporal Self-Attention**。当前 BEV query 不只问摄像头“现在这里有什么”，还会问上一帧 BEV：“刚才这个地方有什么？”

## 07 · 先对齐自己的运动，再使用记忆

这里还有一个很实际的问题：**自己的车也在动。**上一帧地图上的“前方 10 米”到了这一帧并不是同一个现实位置，所以如果直接把历史 BEV 叠上来，会错位。作者会根据 ego-motion，先把上一帧 BEV 对齐到当前汽车坐标系，再做 temporal attention。消融实验显示，不做 ego-motion alignment 时 NDS 从 51.7% 降到 51.0%。 这个提升看起来只有 0.7 点，但思想非常重要：**记忆不是简单把过去存下来，而是要先把过去转换到“现在的坐标系”，过去才有意义。**这和你之前读机器人 MEM 那篇有一点很漂亮的呼应：现实世界的记忆必须考虑身体移动以后参考系已经变化。

## 08 · 时间让速度与遮挡变得可理解

Temporal Self-Attention 为什么效果这么明显？看速度估计最直观。一张静止照片里，你很难判断旁边那辆车是停着还是以 40km/h 向前开；要知道速度，天然就需要比较多个时间点。nuScenes val 上，静态版 BEVFormer-S 的 mAVE，也就是平均速度误差，是 **0.802 m/s**；加入 temporal information 后降到 **0.394 m/s**，几乎减半。 在测试集上，BEVFormer 的 mAVE 是 0.378，而此前很多 camera-based 方法还在 0.8–1.5 左右。 所以这篇论文很好地说明了一件很基本的事情：**时间不是视觉里的额外 feature，很多物理属性本来就只有通过时间才能观察出来。**

遮挡也是一样。论文专门按照目标物体的可见程度，把 nuScenes 切成 0–40%、40–60%、60–80%、80–100% 四组。加入 temporal information 的完整 BEVFormer 在所有组都有更高 recall，尤其在最低可见度，也就是物体大部分被遮住的那一组，优势更明显。 用费曼法说就是：**当前眼睛看不到，不代表世界里不存在；记忆可以补当前视觉的洞。**这一思想后来几乎会出现在所有复杂自动驾驶/机器人感知系统里。

## 09 · 递归记忆：只保存压缩后的世界状态

这篇论文还有一个非常重要但容易被忽略的设计：它不是把过去很多帧全部堆起来，然后一次性做大 Transformer，而是像 RNN 一样，**每次只保存上一时刻已经压缩好的 BEV state**。上一帧 BEV 本身已经融合了更早的信息，所以它可以递归地把历史往前传。作者明确说，这种方式比直接 stacking 多个历史 BEV 更省计算，也能减少无关历史信息干扰。 训练时他们会从过去 2 秒里取 4 个时间点，递归生成历史 BEV；消融发现从 1 帧增加到 4 帧，NDS 从 0.448 持续涨到 0.517，到了 4–5 帧以后基本开始饱和。 这说明短时间历史已经提供了大量价值。

## 10 · 统一表示支持多种感知任务

有了统一 BEV feature 后，后面的任务反而变简单了。3D object detection 可以拿这张 BEV feature 接一个修改版 Deformable DETR，直接预测物体的三维中心、尺寸、方向和速度；地图分割则可以接 segmentation head，预测道路、车道线、车辆区域等。 这和你前面读 foundation model 时经常看到的思想很像：**真正强大的东西不是一个特别好的“汽车检测器”，而是先学习一个通用 representation，然后很多任务共用它。**BEVFormer 虽然还不是今天意义上的 foundation model，但它已经明显在做“统一中间世界表示”这件事。

## 11 · 实验结果与纯视觉感知的意义

最终结果在当时非常强。使用 VoVNet-99 backbone 的完整 BEVFormer 在 nuScenes test 上达到 **56.9% NDS、48.1% mAP**；同设置的静态 BEVFormer-S 是 49.5% NDS，DETR3D 是 47.9%。更值得注意的是，这个纯 camera 模型已经和 SSN 这种 LiDAR baseline 的 56.9% NDS 持平，虽然更强的 LiDAR 方法依然更高。 在 Waymo 上也比 DETR3D 明显提升。 所以 BEVFormer 的历史意义不是“摄像头从此超过 LiDAR”，而是第一次很强地证明：**如果把多相机空间几何和时间记忆组织得足够好，纯视觉也可以建立相当强的三维世界表示。**

## 12 · 从独立检测到统一世界表示

如果把这篇放进更长的自动驾驶技术史里，它真正推动的是一种范式：**从“每个摄像头各自检测物体”，转向“先构建统一 BEV 世界，再在 BEV 上做所有事情”。**以前像六个人各拿一个望远镜，分别喊“我这里看到一辆车”；BEVFormer 更像让六个人共同维护一张作战地图。地图上的每个格子不断从六个方向收集证据，还记得几百毫秒以前发生了什么。这样后面的检测、地图理解、运动预测乃至规划，都有机会共享一个统一坐标系。后来的很多 BEV 感知、BEV-based planning、occupancy prediction，本质上都受益于这种“先建立 ego-centric world representation”的思想。

## 13 · 连接机器人记忆与世界模型

而如果把它和你最近一直读的机器人 world model、MEM、MotionWAM 连起来，会发现一个非常清楚的演化。BEVFormer 还主要在做 **perception memory**：多相机 + 历史 → 当前世界状态；机器人 MEM 进一步把历史分成短期视觉和长期语义；MotionWAM 又开始问“除了记过去，还能不能学未来 dynamics”；Motus2 再往前把“未来预测”用于候选动作规划。所以可以粗略地把路线写成：**BEVFormer：把过去和多视角压成统一当前世界；World Model：从统一当前世界预测未来；Action Model：再利用未来决定怎么行动。**从这个角度看，BEVFormer其实是后来很多 embodied world representation 思想的一个重要早期节点。

## 14 · 局限：标定、高度与历史压缩

当然，它也有很明显的局限。第一，它严重依赖准确的 camera calibration，因为 Spatial Cross-Attention 要把 BEV reference points 投影进各摄像头；相机外参出错会影响定位。论文附录也专门测试了 calibration noise。 第二，它的 BEV 本质仍然主要是二维地面网格，用不同高度 reference points 去取信息，并不是完整的 3D occupancy world model，所以对于多层结构、复杂高度关系表达有限。第三，它主要解决感知，没有把 prediction/planning/action 放到统一 end-to-end 模型里。第四，历史主要通过上一帧 BEV recurrently 传递，这是一种高效压缩，但很久以前的细节也可能逐渐丢失。

## 15 · 用一分钟讲给朋友听

如果现在合上论文，用费曼法一分钟给朋友讲，我会这样说：**自动驾驶车上有六个摄像头，但每张图都只是不同方向的二维透视画面。BEVFormer 先在汽车周围建立一张俯视网格，每个网格位置都有一个 learnable query，主动问：“现实世界这个位置现在有什么？”它根据相机几何把这个位置投影到对应摄像头，再用 deformable attention 从附近提取视觉特征，于是六张图被融合成统一 BEV 地图。它还把上一时刻的 BEV 保存下来，根据自己的车辆运动先对齐，再通过 temporal self-attention 融进当前 BEV，所以即使某辆车现在被遮住，也可以利用刚才的记忆；速度估计也因此大幅变准。最后同一张 BEV 可以同时拿去做 3D 检测和地图分割。在 nuScenes 上，它把 camera-only 3D detection 推到 56.9% NDS，比当时 DETR3D 高 9 个百分点。**

所以这篇我建议最后只记一句话：**不要让六个摄像头分别理解六个世界，而要让它们共同维护一个随时间更新的世界地图。** 📖 更技术一点就是：**BEV Queries 负责“世界里的哪里”，Spatial Cross-Attention 负责“现在看到了什么”，Temporal Self-Attention 负责“刚才这里发生过什么”。**BEVFormer真正改变的不是一个检测头，而是感知系统内部“世界应该怎样表示”的方式。
