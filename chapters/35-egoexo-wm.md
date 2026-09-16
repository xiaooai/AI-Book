# EgoExo-WM：把旁观经验变成第一视角预测

论文：EgoExo-WM: Unlocking Exo Video for Ego World Models（2026）

## 01 · 用身体动作连接两种视角

这篇是 2026 年 5 月的《EgoExo-WM: Unlocking Exo Video for Ego World Models》。如果用费曼法先把整篇压成一句话，它真正想解决的是：**第一视角 world model 很适合机器人，因为机器人未来真正看到的就是“自己的眼睛”，但第一视角数据特别少，而且第一视角偏偏又最看不清自己的身体；第三人称视频却遍地都是，而且能清楚看到人的全身动作。那么能不能把互联网上大量“别人做事情的视频”，翻译成“如果是我自己在做，我眼里会看到什么”，然后拿这些视频训练第一视角 world model？** EgoExo-WM 的答案是可以，而连接这两个视角的桥梁不是语言，而是 **3D human motion——人的三维身体动作**。作者先从第三人称视频里恢复人的 3D 姿态，把它一方面当作“动作标签”，另一方面用来指导 exo→ego 视频转换；最后得到一组“第一视角画面 + 对应全身动作”，就可以像真正的第一视角具身数据一样训练 world model。([arXiv][1])

## 02 · 第一视角数据为何难以扩展

先理解作者为什么觉得 egocentric world model 很重要。假设你想训练一个机器人，它现在看到桌子上的杯子，准备伸手。一个真正有用的 world model 应该能回答：“如果我身体往前一步、右手伸出去、手腕转一下，**我接下来会看到什么？**”如果这个预测足够好，机器人就能在真正行动以前先在内部试几种动作。可是第一视角视频有两个天然麻烦。第一个是贵：必须真的让人戴摄像头到处做各种事情，规模远小于互联网普通视频。第二个更加讽刺：**第一视角最看不见的恰恰是自己。**摄像机装在头上，你能看到桌子，却看不到自己的背、腿、很多时候连手也会被物体挡住，所以很难仅凭 ego video 得到准确的全身 3D action。相反，第三人称视频里整个人通常清清楚楚，而且 HowTo100M、教学视频、生活视频这些数据量巨大。作者因此说，要 scale egocentric world models，就不应该只等人继续戴相机采数据，而应该想办法把现有 exocentric video 变成第一视角训练经验。([arXiv][1])

## 03 · 视角转换不能只求画面相似

这件事最难的地方是：**第三人称画面和第一人称画面不是简单“换个摄像头角度”就结束了。**假设第三人称视频里一个人在切菜，你从旁边能看见人的身体、菜板、刀和厨房；但如果真的站进这个人的头里，手会变得很大，身体大部分消失，很多桌面区域又会被手臂遮挡。第三人称里甚至可能根本看不到“眼睛正前方”的某些东西。所以 exo→ego 本质上是个欠定问题：不存在唯一正确答案。更糟的是，普通视频生成器很容易产生“视觉上差不多，但动作是错的”视频，例如左右手对调、手穿过物体、手的位置和真实身体姿态对不上。对看视频的人这可能只是一个小瑕疵，对 world model 却非常危险，因为你等于在告诉模型：“做这个动作以后，我的手会出现在错误位置。”([arXiv][1])

## 04 · EgoX-Body：用骨架约束生成

所以作者提出 **EgoX-Body**，核心思想特别直白：**生成第一视角视频时，别只告诉生成模型场景长什么样，还要把人的骨架显式告诉它。**系统首先从第三人称视频恢复完整 3D body pose，然后在原始 exo video 上画出人体 skeleton，作为一个结构化动作提示。与此同时，它又根据 3D 身体姿态和场景几何推算“如果摄像机在这个人的头上，双手大概应该出现在第一视角画面的哪里”，再画一个 egocentric hand-skeleton prior。于是视频生成器同时拿到两个强约束：第三人称侧告诉它“人实际上怎么动”，第一人称侧告诉它“从自己的眼睛看，手大概应该落在哪里”。最后再用 video diffusion model 把这些条件融合成新的第一视角视频。([arXiv][1])

可以把这个过程想成电影拍摄。原来你只有一个摄影师从房间角落拍演员。普通 exo-to-ego 模型像让 AI 猜：“如果演员额头上也有 GoPro，那条视频大概是什么样？”EgoX-Body 则额外给 AI 一份演员每一帧的 3D 骨架，以及“GoPro 视角里手应该大致在这里”的草图。于是它不再完全凭想象，而是**被身体运动学约束住了**。论文的定性例子里，加入 body/hand priors 后，手部位置和动作明显比原版 EgoX 更贴合原视频中的真实行为。作者也很谨慎地说，exo-to-ego conversion 本身不是他们唯一或最终的技术重点，将来完全可以替换成更强的视角转换器；真正核心的是：**只要能把 exo video 转成 action-aligned ego experience，第三人称数据就能开始进入 egocentric world-model training。**([arXiv][1])

## 05 · 69 维动作：具体的身体运动

接下来要理解他们的 **action 到底是什么**。这里不是机器人的电机角，也不是一句“reach for mug”，而是人的 whole-body 3D motion。每一步 action 一共是 **69 维**：根节点的 3D 位移，加上 22 个身体关节各自的 3 维相对旋转。([arXiv][1]) 这个选择很重要。语言“拿杯子”太粗，因为同一句话可以对应很多不同动作；某个隐藏 latent action 又很难解释。3D body motion 则比较接近真正具身控制需要的信息：身体往哪里移动、躯干怎么转、胳膊怎么伸。用费曼法说就是：world model 不只是知道“人在拿杯子”，而是收到一份更具体的动作剧本：“身体向左前方移动这么多，肩膀这么转，手臂这样抬。”

## 06 · 在视觉隐空间里预测未来

有了“当前第一视角 + 具体身体动作”，world model 就学习一个特别朴素的问题：**执行这个动作以后，下一刻第一视角世界会变成什么样？**不过作者没有直接预测像素视频，因为像素太昂贵，而且会让模型花大量容量预测光照、纹理等不一定和规划有关的东西。他们使用预训练的 **DINOv3-L** 把第一视角图像编码到 latent space，然后让 world model 根据过去一段视觉 latent 和当前 69D body action，预测下一帧的视觉 latent。训练目标基本就是让预测 latent 靠近真实未来图像对应的 DINO latent。([arXiv][1]) 这和你之前读 MotionWAM 时的思想非常相似：**真正有用的不一定是生成一幅好看的未来图片，而是学习一个包含未来几何和语义变化的内部表示。**

## 07 · 手腕一致性：让预测关注行动

不过仅仅让 DINO latent 接近，还有一个风险：模型可能把“房间整体差不多”预测得很好，却把最重要的手预测错了。比如真实未来里右手已经伸向杯子，但预测 latent 里手的位置偏了很多，整体 feature distance 仍然可能不算特别大。于是作者专门加入一个 **wrist-position consistency loss**。他们在预测未来 latent 上接一个很轻量的 head，让它额外预测未来手腕在画面中的位置，再和训练视频提取出来的 wrist pseudo-label 对齐。([arXiv][1]) 这相当于告诉模型：“预测厨房背景当然重要，但我真正关心的是行动；别把人的手和身体运动当成可以忽略的视觉小细节。”

## 08 · 公平比较：替换而非增加训练数据

现在才来到整篇论文最重要的实验设计。作者原本有大约 **200 小时 Nymeria 第一视角数据**，这些数据还有同步的全身 mocap，是很珍贵的 ego + action 数据。为了公平，他们不是简单给自己额外加数据、然后和 baseline 比，而是保持总训练量大致都是 200 小时：**Ego-WM** 用完整 200 小时 ego data；**EgoExo-WM** 则用 190 小时 Nymeria + 10 小时由第三人称视频转换出来的数据。还有一个非常关键的 **Naive EgoExo-WM**：同样是 190 小时 Nymeria + 10 小时 exo video，但它直接把原始第三人称视频塞进去，不做 exo→ego action alignment。这样就可以回答一个核心问题：成绩变好到底只是因为“数据更多样”，还是因为“你真的成功把第三人称经验翻译成了第一人称经验”？([arXiv][1])

这 10 小时第三人称数据来自三个很不一样的来源：HowTo100M 大约 5 小时，CrossTask 大约 1 小时，100 Days of Hands 大约 4 小时。作者特别说明，**10 小时不是因为 exocentric data 只有这么多，而是因为当前 exo-to-ego video generation 计算很贵，他们的算力只允许转这么多。**([arXiv][1]) 这个点非常关键：整篇论文真正的 ambition 不是“10 小时合成数据提升了一点成绩”，而是：如果互联网 exo video 可以通过这个桥梁使用，那么未来可扩展的数据池会比现有 ego dataset 大几个数量级。当前实验更像是在证明“管道是通的”。

## 09 · 未来预测与身体一致性的实验

结果确实支持这件事。作者在四组**训练外**数据上评估：HOMAGE、LEMMA，以及 Ego-Exo4D 的 Bike 和 Cooking。测试不是只看下一帧，而是做 **2 秒 open-loop rollout**，也就是预测结果继续喂回自己，再预测下一步，总共 8 帧、4Hz。([arXiv][1]) 在 HOMAGE 上，只用 ego 数据的 Ego-WM 最终 2 秒 DINO latent L2 误差是 0.069，直接混 raw exo 的 Naive EgoExo-WM 是 0.065，而完整 EgoExo-WM 降到 **0.057**；wrist PCK@20 则从 Ego-WM 的 0.313 提高到 **0.404**。LEMMA 上，wrist PCK@20 从 0.433 提到 **0.515**，平均 PCK 从 0.527 到 **0.618**。([arXiv][1]) 这意味着不仅未来场景 latent 更准，预测的手部动作也更符合真实未来。

尤其值得注意的是，**Naive EgoExo-WM 明显不等于完整 EgoExo-WM。**在某些视觉 L2 指标上直接加 exo video 也有一点帮助，因为场景内容确实更丰富；可是 hand/body consistency 往往明显不如经过转换的数据。作者据此强调：收益不只是来自“看了更多第三人称画面”，而是来自把这些画面转换成与 ego world model 一致的 **observation-action format**。([arXiv][1]) 用费曼法说就是：你想教一个第一人称模型游泳，单纯给它看电视转播当然有一点帮助；但如果能把电视转播翻译成“假如是你自己在水里，你眼睛会看到什么、身体会怎么动”，学习价值会高很多。

## 10 · 用世界模型为候选动作排序

不过 world model 如果只是“预测未来更准”，还不够证明它真的对 agent 有用。所以作者又做了 **planning**。给模型一个当前第一视角和一张目标图片，例如“最后我希望自己走到水槽左边”或者“最后希望 cereal 已经被倒出来”，然后 UniEgoMotion 一次提出 **4 条不同的 3D human motion sequences**。EgoExo-WM 不直接生成动作，而是对每一条候选动作都在内部 rollout 未来，看看最后预测到的 visual latent 哪一个最接近目标图像的 latent，然后选最好的那一条。([arXiv][1]) 这就是很经典的 Model Predictive Control 思想：**我不一定自己发明动作，但我可以当裁判——先想象每种动作会发生什么，再选最可能达到目标的。**

这个角色和你前面读 Motus2 时的 **Policy → Simulator → Evaluator** 很容易联系起来。EgoExo-WM 其实更早、更简单：UniEgoMotion 是 proposal model，world model 同时充当 simulator 和一种 implicit evaluator，因为最后只需要比较预测 future latent 和 goal latent 的距离。它会问：“动作 A 以后我眼里的世界离目标还有多远？动作 B 呢？”然后选择最近的。结果在 HOMAGE、LEMMA、Ego-Exo4D Bike 和 Cooking 四组 planning test 上，使用 EgoExo-WM 排序的动作序列在 whole-body MPJPE 和 Wrist MPJPE 上都比“不做 world-model ranking”和只用 ego-data world model 更低。比如 HOMAGE 的 whole-body MPJPE 从 UniEgoMotion 本身的 0.404 降到 Ego-WM 的 0.383，再降到 EgoExo-WM 的 **0.362**；Ego-Exo4D-Bike 从 0.292 → 0.267 → **0.245**。([arXiv][1])

## 11 · 第三人称视频如何扩展规划经验

这件事其实说明了论文更深的意义：**第三人称视频最终不是只在提高“视频预测”，而是在扩大 planner 的物理经验。**假设 Nymeria 原本很少见骑自行车、某些家庭整理动作或者特殊物体互动，那么只用 Nymeria 训练的 world model面对候选动作时，会不知道某些行为后果应该是什么。加入转换过来的互联网视频，相当于给它看过更多“人做事以后世界怎样变化”，因此它在几个候选未来之间判断哪一个靠谱时更有依据。([arXiv][1]) 这和语言模型中“预训练知识让推理更有底子”的关系非常像：world model 的规划能力上限，也会受到它过去见过多少种物理互动的限制。

## 12 · 学动作与学动态的区别

如果把这篇和你之前读的 **Human-to-Robot Transfer** 放在一起，会发现两篇都想吃互联网/人类数据，但切入点完全不同。Human-to-Robot 那篇问的是：**人的动作能不能直接成为机器人 policy 的训练数据？**EgoExo-WM 问的是：**第三人称人类视频能不能成为第一视角 world model 的“预测经验”？**前者更接近学 policy，后者更接近学 dynamics。你可以把两者理解成：一个人在说“看别人做，我学他的动作”；另一个人在说“看别人做，我学世界在动作下会怎样变化”。真正成熟的 embodied foundation model 很可能两种都要。

## 13 · 连接 MotionWAM 与 SAGE 的数据路线

它和 MotionWAM 的关系也特别直接。MotionWAM 使用大量第一视角视频去学习“未来动态表示”，但第一视角数据天然稀缺；EgoExo-WM 正好提供一个潜在的数据扩容方向：**世界模型未必只能吃原生 ego video，大量第三人称视频可以先被重新投影到第一人称语义空间。**从这个角度看，EgoExo-WM 更像在解决 world-model scaling 的**数据源问题**，而 MotionWAM / Motus2 更像在解决 world model 学会以后**怎样用来控制和规划**。

它和你刚读的 SAGE 也构成一种非常有意思的对照。SAGE 的思路是：“现实数据不够，我自动生成更多**世界**和机器人轨迹。”EgoExo-WM 的思路则是：“现实里其实已经有海量视频，只是视角不对，我把现有的**观察经验重新翻译**一下。”一个走 simulation generation，一个走 view conversion。二者背后其实是同一个大问题：**未来 embodied scaling 的核心，不只是谁的 Transformer 更聪明，而是谁能把原本不能用于机器人训练的数据池变成可用数据。**

## 14 · 限制：转换成本、动作细节与时长

不过这篇论文现在离真正“互联网规模”还非常远。最明显的限制就是，他们实际只转换了大约 **10 小时** exocentric video，因为 video diffusion 转换的推理成本比较高。([arXiv][1]) 第二，EgoX-Body 对复杂遮挡、非常精细的手—物体接触、小物体 manipulation 仍然会失败，甚至有生成结果退化成大面积黑白画面的情况。第三，动作空间里的 SMPL body pose 并没有 articulated hand，所以精细手指动作并没有真正完整建模。第四，转换器目前只能处理大约 **49 帧** clip，论文 planning horizon 也只有 **2 秒**，长程 rollout 的误差累积还完全没有解决。([arXiv][1]) 所以这篇不是在说“我们现在可以拿整个 YouTube 训练完美 world model”，而是在证明：**exo→ego→world model 这条数据管道在原则上确实有效。**

## 15 · 人类动作空间与机器人身体的边界

还有一个边界也很重要：这篇主要是 **human-centered egocentric world model**，并没有直接拿真实机器人做 manipulation 实验。论文说它可以用于未来 robotics、AR coaching 和 assistive system，但当前 planning 实验仍然是在“人的 3D motion”空间里选动作。([arXiv][1]) 所以从 humanoid robot 的角度，它更像提供了一层“人类第一视角物理预测 prior”；后面仍然需要解决 human body → robot embodiment 的 grounding，就像你前面读 Ψ₀、Human-to-Robot、ω-0 时一直看到的那个 gap。

## 16 · 具身数据来源的持续扩展

如果把这一篇放回你最近读的整个路线，会出现一条越来越清楚的数据演化：**LingBot-VLA / GEN 系列问“真实机器人或真实物理数据能不能继续 scale”；Human-to-Robot 问“人的第一视角动作数据能不能转给机器人”；MotionWAM 问“第一视角视频能不能教 world dynamics”；SAGE 问“训练世界能不能自动生成”；EgoExo-WM 则再追问一句——为什么连第一视角视频都必须真的戴摄像头拍？互联网上绝大多数第三人称视频，能不能也被翻译成 embodied experience？**这其实是很自然的一步：当模型越来越强以后，研究开始不断“解锁”过去因为格式不对而被浪费的数据。

## 17 · 用一分钟讲给朋友听

如果现在合上论文，用费曼法一分钟讲给朋友听，我会这样说：**训练第一视角 world model 最大的问题是 ego video 少，而且从自己的眼睛看反而看不清自己的身体。EgoExo-WM 发现第三人称视频正好相反：互联网里非常多，而且能清楚看到全身动作。所以作者先从第三人称视频恢复人的 3D body pose，把这个 pose 一方面当作 world model 的 69 维 action，另一方面拿去约束视频生成器，把第三人称视频转换成“如果摄像头在这个人头上，会看到什么”的第一视角视频。这样就得到了一对对‘第一视角观察 + 对应身体动作’。再用这些数据训练一个在 DINOv3 latent space 里预测未来的 world model，同时加手腕位置 loss 保证动作别预测歪。实验只替换了 200 小时训练数据里的 10 小时，就在多个完全不同的第一视角数据集上明显改善了未来预测，而且 world model 还能对几条候选全身动作做内部 rollout，选出最接近目标画面的那条，在规划指标上也持续优于只用 ego data 的模型。**([arXiv][1])

所以这篇我建议最后只记一句话：**第一视角 world model 不一定只能从第一视角经验学习；如果能找到正确的身体桥梁，观察“别人”也可以变成预测“自己”。** 📖 更技术一点就是：**Exocentric video → 3D human motion → action-aligned egocentric experience → world model → planning。**如果 MotionWAM 告诉我们“预测未来能给机器人 dynamics intuition”，那 EgoExo-WM 紧接着补上的问题就是：**为了学这种未来直觉，我们能不能把互联网上海量第三人称人类行为，都变成第一视角的物理经验。**

[1]: https://arxiv.org/pdf/2605.15477 "EgoExo-WM: Unlocking Exo Video for Ego World Models"
