# MotionWAM：用世界模型的动态表示驱动全身动作

论文：MotionWAM: Towards Foundation World Action Models for Real-Time Humanoid Loco-Manipulation（2026）

## 01 · 用未来动态帮助全身行动

这篇是 2026 年 6 月的《MotionWAM: Towards Foundation World Action Models for Real-Time Humanoid Loco-Manipulation》。如果把它和你前面刚读的 Ψ₀、ω-0 放在一起，会发现这三篇都在攻同一个大方向：**人形机器人不能只是“上半身会操作、下半身负责别摔倒”，而应该把走路、蹲下、转腰、调节身高、手臂操作，甚至脚去踩踏板、踢球这些行为，统一成一个真正的 whole-body policy。**但 MotionWAM 的独特之处是，它把 **World Action Model，世界动作模型** 放到了中心位置。它不满足于“看到当前图像，直接预测动作”，而是先利用一个视频世界模型内部对未来的“想象”，再让动作模型读取这种未来动态信息来决定怎么动。最关键的是，它没有真的把完整未来视频一步步生成出来，因为那太慢，而是只读取视频扩散模型在一次前向传播中的**中间去噪特征**。你可以把整篇论文先记成一句话：**机器人不需要真的把未来电影完整渲染出来，只需要偷看一下“未来大概往哪儿走”的内部想象，就能做出更好的全身动作。**([arXiv][1])

## 02 · 静态语义与动态直觉

先用费曼法想一个最简单的例子。假设一个人形机器人要把桌下的衣篮拿出来放到桌上。传统 VLA 往往是：看当前画面，理解“拿篮子”，然后直接预测动作。问题是，这个任务不是单纯伸手，它需要先下蹲、调整腰部、靠近篮子、双手抓住、站起来、转身，再把篮子放到桌上。当前画面本身并没有明确告诉你“如果我现在这样蹲、这样伸手，下一秒视野会怎么变化”。World model 的意义就在这里：它逼模型学习“**我的动作和世界后续变化之间是什么关系**”。MotionWAM 认为，单纯从静态图像和文字预训练出来的 VLM 虽然很懂“篮子是什么”，却不天然懂“身体向前一步以后，篮子会怎样进入视野”“蹲下时相机高度怎样变化”“脚踢球以后球会怎么走”。这些是 temporal dynamics 和 contact physics，视频世界模型比静态 VLM 更适合学。([arXiv][2])

## 03 · 一次前向中的未来想象

这就是 MotionWAM 和普通 VLA 最根本的差别。普通 VLA 更像 \(当前画面 + 语言 → 动作\)；MotionWAM 更像 \(当前画面 + 语言 → 对未来视觉变化的内部想象 → 动作\)。论文把它描述成 **predict video dynamics, then invert to action**。不过“预测未来”这里千万不要理解成模型真的先生成一段高清视频，再看视频决定动作。那样会太慢。作者使用的是一个 Video DiT，它来自 Cosmos-Predict2.5-2B 这类视频世界模型。通常 diffusion / flow model 要经过很多次 denoising 才得到未来画面，而 MotionWAM 只在某个固定 flow timestep 上跑**一次 forward pass**，然后直接截取 Transformer 某一层的 hidden state。这个 hidden state虽然还没有变成清晰未来图像，却已经包含“场景大概朝哪里变化”的信息。作者把这称为一种 **one-shot imagination**。([arXiv][2])

用费曼法类比，就是你准备伸手接一个球。你脑中未必会真的生成一段“未来 0.5 秒高清电影”，但你有一种很快的预感：球会往右下方走，所以手应该提前往那里伸。MotionWAM 就是把这种“未来预感”从视频世界模型里抽出来。Video DiT 看当前 egocentric camera、语言目标和一份未来噪声，单次前向以后产生中间特征；Motion DiT 再读取这些特征，加上 proprioception 和当前动作噪声，通过 flow matching 生成 whole-body motion latent。这样它既利用了世界模型的动态先验，又避免真的把未来视频完整生成。([arXiv][2])

## 04 · 让整个身体参与任务

这篇第二个非常重要的点，是**统一全身动作空间**。传统 humanoid loco-manipulation 常常是分层控制：高层 manipulation policy 负责上半身，给手臂精细 joint target；低层 locomotion controller 只收“向前走多少、转多少、身体多高”这些 coarse base commands。结果就是腿基本只负责“把身体运过去并保持平衡”，根本没有自己的任务语义。你让机器人“踢球”，这种结构就很尴尬，因为“脚本身就是工具”，可高层 policy 没有权限直接表达“右脚去踢这个球”。MotionWAM 因此把 locomotion、躯干、身高变化、脚部交互和手操作都放进**同一个 whole-body motion latent**。([arXiv][2])

这个 latent 建立在 SONIC whole-body controller 上。SONIC 本身会把不同全身运动压进一个共享 latent，再用 FSQ 做离散化。MotionWAM 最终预测的东西大致分两部分：一部分是 SONIC 的 motion token，负责总结走路、转腰、调高度、脚交互等全身意图；另一部分是连续手部通道，比如 gripper 或 dexterous hand command。([arXiv][2]) 用费曼法说，就是过去系统给腿的命令只有“走过去”，现在模型可以对整个身体说：“这一步右脚要参与任务，身体同时降低，腰朝左转，双手准备接。”所以它不只是 whole-body **balance**，而是 whole-body **task execution**。

这也是为什么论文里有“踢足球”“踩踏板”这种任务。传统 decoupled 上下身政策就算上半身很聪明，也很难让脚主动成为任务执行器；MotionWAM 的 unified latent 让腿不再只是搬运身体。作者明确强调，脚现在拥有了 task-driven action vocabulary。([arXiv][2]) 这其实是一个很大的概念变化：**人形机器人最有价值的地方，本来就是整个身体都有自由度，如果最后还把腿当轮子用，其实浪费了人形形态。**

## 05 · 三阶段学习世界与动作

训练方法也非常值得读，因为它再次证明“别把所有东西一开始就联合训练”往往更有效。MotionWAM 使用三阶段训练。第一阶段只训练视频世界模型，而且完全不需要 action label。他们混了大约 **2136 小时**第一视角人类视频和 humanoid 视频，只让 Video DiT 学未来帧生成，把原来比较通用的视频模型适配成“机器人头部第一视角下世界通常怎样变化”的 dynamics prior。([arXiv][2]) 这一步很聪明，因为视频便宜、动作标注贵。如果一开始就要求每段视频都必须有 robot action，数据规模马上被卡死。作者认为第一阶段真正的瓶颈不是 action diversity，而是**egocentric visual dynamics**。

第二阶段才把 Motion DiT 接上来，用不同来源、不同 end-effector、不同 action annotation 的 Unitree G1 数据做 **cross-embodiment action post-training**。为了避免新来的动作信号把第一阶段好不容易学到的视频 dynamics prior 洗掉，训练时继续保留 video loss，联合优化 \(\mathcal L_{motion}+\mathcal L_{video}\)。([arXiv][2]) 这和你前面读 Knowledge Insulation 的精神很接近：新动作学习不能把旧知识直接冲掉。只是这里保护的不是静态 VLM 语义，而是视频世界模型的动态先验。

第三阶段才是真正针对那 9 个 humanoid loco-manipulation task 做少量 whole-body teleoperation fine-tuning，每个任务约 **200 条 episode**。([arXiv][2]) 于是三阶段的分工特别清楚：**第一阶段学“世界怎么动”，第二阶段学“这些动态怎样映射到机器人动作”，第三阶段学“在这个具体身体和具体任务里怎么用”。**这和 Ψ₀ 那篇“人类视频负责表示、humanoid 数据负责 embodiment control”其实高度相似，只不过 MotionWAM 第一阶段学的是**视频未来动态**，而不是 action-token representation。

## 06 · 九个真实全身操作任务

实验设计也很有针对性。作者在 Unitree G1 上设计了 9 个真实任务，而且刻意保证没有一个任务能靠纯上半身解决。任务包括 Pick-and-Place Bottle、Kick Soccer、Retrieve Item、Load Cart、Toss Garbage、Lift Basket、Stock Shelves、Wipe Board 和 Do Laundry。([arXiv][2]) 这些任务分别逼机器人用腰、调身高、蹲走、脚参与任务、身体和手协同。比如 Kick Soccer 直接要求脚成为操作器；Lift Basket 要从桌下拿衣篮，就必须下蹲和调身体高度；Load Cart 需要一边推车、一边把衣服装进去；Wipe Board 则要求身体范围和手部擦拭配合。这样才能真正检验“whole-body”是不是口号。

## 07 · 相同动作接口下的模型比较

结果很明显。在所有模型使用**相同 Stage-3 demonstration、相同观察、相同语言、相同 proprioception，而且最后都通过同一个 SONIC action interface 输出动作**的条件下，MotionWAM 9 个任务平均成功率 **76.1%**，最强 baseline GR00T-N1.7 是 **43.9%**，绝对提升超过 32 个百分点；π0.5 在这个设置下整体低于 20%。([arXiv][2]) 尤其是需要强全身参与的任务差距更大：Kick Soccer、Load Cart、Retrieve Item 都大约高 40 个百分点，Wipe Board 高约 45 个百分点。([arXiv][2])

这里最值得注意的对比其实不是“MotionWAM 赢了 π0.5”，因为这些模型本来设计目标不同，而是作者专门做了一个**参数规模匹配的 Qwen3DiT**：同样用 Motion DiT 和统一动作空间，只把视频 world-model backbone 换成 Qwen3-VL 的静态视觉语言 backbone。结果这个 VLM-only baseline 在很多 locomotion-heavy task 上几乎崩掉。([arXiv][2]) 这才更直接支持论文真正想证明的事情：**同样是 2B 级 backbone、同样动作头，能预测视频动态的 world-model prior 比只懂图片语义的 VLM prior 更适合复杂 whole-body control。**

为什么会这样？因为静态 VLM 通常特别会回答“这是什么”“目标在哪”，但人形动作真正难的是“如果身体向这里倾，下一秒会不会失稳”“手和物体接触以后，物体怎么动”“蹲下时相机视野和可达范围怎么变化”。这些都属于 dynamics。你可以把 VLM 和 WAM 的差别理解成：**VLM 像一个看图很聪明的人，WAM 更像一个看过大量视频、对‘接下来会发生什么’有直觉的人。**对静态 tabletop pick-and-place，两者差距可能不大；对需要走、蹲、踢、推、擦这种连续动力学任务，后者的优势就更明显。

## 08 · 视频预训练与动作对齐缺一不可

三阶段消融也非常干净。完整三阶段在选出的五个任务上平均成功率约 **70%**；去掉 Stage 1 的 egocentric video pretraining，只保留 Stage 2+3，掉到 **59%**；去掉 Stage 2 的 cross-embodiment action grounding，只靠 Stage 1+3，则直接掉到 **42%**。([arXiv][2]) 这说明两步解决的是不同问题：Stage 1 给的是“第一视角未来动态直觉”，Stage 2 给的是“这种动态和机器人动作之间到底怎么对应”。只会预测未来，没有足够 action grounding，不行；只会 action mapping，没有专门适配过 egocentric dynamics 的视频 prior，也不够强。

## 09 · 世界动作模型怎样接近实时

然后来到整篇论文最关键的工程问题：**World Action Model 不是都很慢吗？**没错。像 Cosmos Policy 这类 WAM 通常要把未来视频 iterative denoise 完，再生成动作，所以实时性很差。作者报告在一张 NVIDIA A100 上，Cosmos Policy 同等级规模大约只有 **0.7 Hz** chunk-wise 输出频率；MotionWAM 有大约 2.5B trainable params，却能达到 **4.9 Hz**，快约 7 倍；GR00T-N1.7 是 6.5 Hz，Qwen3DiT 是 9.0 Hz。([arXiv][2]) 也就是说，MotionWAM 虽然比纯 VLA 稍慢，但已经进入作者认为可以支持 closed-loop humanoid balance 的实时区间。

为什么它能快这么多？核心就回到前面那个 one-shot imagination：**不要把未来视频生成完。**传统 WAM 像“我必须先把未来一整幅画画完，然后看图决定动作”；MotionWAM 像“画到第一笔时，我已经从脑子里知道这幅画大概往哪边发展，于是直接拿这个内部表示去决定动作”。这实际上是在利用生成模型的**中间计算状态**，而不是最终生成结果。这个想法非常漂亮，因为很多 foundation model 内部其实已经形成了比最终输出更丰富的表示，未必每次都需要真的把输出 decode 出来。

## 10 · 与 ω-0 的未来表示路线对照

如果把它和 ω-0 放一起看，两篇都在讲 world model + whole-body humanoid，但技术哲学不一样。ω-0 更像：**预测压缩后的 future visual latent，用未来 latent 直接帮助动作生成**；MotionWAM 更进一步地说：**连 future latent 都不用完整生成，只读 video diffusion 中间 hidden state 就行。**所以两篇可以看成同一个方向上的不同压缩级别：完整未来视频太慢 → 压成未来视觉 latent → 再压成“生成未来过程中已经出现的内部 dynamics feature”。越往后，越强调**不用把未来真的渲染出来，预测未来这件事本身产生的中间表示就有价值。**

## 11 · 无动作标签视频的预训练价值

和 Ψ₀ 比较也很有意思。Ψ₀ 的核心观点是“human data 和 humanoid action distribution 差异太大，所以 staged training”；MotionWAM 同样用了 staged training，但 Stage 1 更激进：**完全不需要动作标签，只用第一视角视频学 world dynamics。**Stage 2 再用 heterogeneous G1 数据做 action grounding。两篇共同指向一个很重要的经验：**未来大规模 humanoid pretraining 可能不需要每一小时数据都有精确机器人动作。**海量人类 egocentric video 可以先负责“学物理世界怎么变化”，真正稀缺的机器人 action data 再负责把这个世界知识 grounding 到某个身体。

## 12 · 预测是任务，表示是资产

这篇还有一个我觉得特别值得长期记住的思想：**World model 的价值可能不在“模拟世界”，而在“提供 dynamics representation”。**以前一说 world model，很容易想象机器人先在脑中模拟十种未来、再规划最优轨迹。但 MotionWAM 展示了一条更轻的路线：根本不要求能看见未来，只要 world model 因为被训练去预测未来，所以它内部被迫形成了一种对 dynamics 有用的表示。然后 policy 直接吃这种表示即可。换句话说，**prediction 是训练任务，representation 才可能是最终资产。**这和你最早读 GPT-1 时“next-token prediction 只是训练游戏，真正得到的是可迁移表示”其实非常像。

## 13 · 当前验证范围与局限

当然，这篇的边界也很明显。第一，Stage 3 只在 Unitree G1 上验证，三阶段 recipe 能不能无缝迁到其他 humanoid 硬件还没有证明。第二，它没有严格做 novel-object OOD generalization，训练和测试物体视觉上有相似性。第三，整个系统只用一个头部第一视角相机，所以一旦物体离开视野或者头部姿态偏离训练分布，就容易失去 grounding、停住。([arXiv][2]) 第四，76.1% 成功率虽然比 baseline 强很多，但距离真正长期无人值守仍然很远。第五，SONIC 仍然承担了底层 whole-body decoding，所以它也不是完全“一个大网络直接控每个电机”。

## 14 · 用一分钟讲给朋友听

如果现在合上论文，用费曼法一分钟讲给朋友听，我会这样说：**MotionWAM 想让人形机器人真正一边走、一边蹲、一边用手和脚做任务。普通 VLA只看当前图像和语言直接预测动作，但作者认为复杂人形控制更需要“未来动态直觉”。于是他们拿一个视频世界模型，不真的生成完整未来视频，而是在一次去噪前向里直接截取中间 hidden feature，相当于让机器人快速获得“世界接下来大概会怎么变化”的内部预感；再让一个 Motion DiT 根据这个动态表示生成统一 whole-body motion latent，把走路、转腰、调身高、脚交互和手操作放在一个动作空间里。训练分三步：先用约 2136 小时第一视角视频学 dynamics，再用多来源 Unitree G1 数据把 dynamics grounding 到动作，最后每个任务用约 200 条 whole-body teleop 示例微调。最终它在 9 个真实 G1 任务上平均成功率 76.1%，比最强 baseline 43.9% 高出 32 个百分点，而且因为不真正把未来视频 denoise 完，推理速度达到 4.9Hz，比同类 world-model policy 快约 7 倍。**([arXiv][1])

所以这一篇我建议最后只记一句话：**机器人不一定需要“看见未来”，但它需要一个为了预测未来而学出来的动态表示。** 📖 更技术一点就是：**Use world-model hidden dynamics, not fully generated future video, to drive unified whole-body action。**如果 Ψ₀ 的核心是“人类视频教动作意义，humanoid 数据教身体怎么动”，那么 MotionWAM 又往前补了一块：**第一视角视频还可以教机器人‘这个世界接下来通常怎样变化’，而这种未来直觉本身，就可以成为全身控制的强先验。**

[1]: https://arxiv.org/abs/2606.09215 "[2606.09215] MotionWAM: Towards Foundation World Action Models for Real-Time Humanoid Loco-Manipulation"
[2]: https://arxiv.org/html/2606.09215v1 "MotionWAM: Towards Foundation World Action Models for Real-Time Humanoid Loco-Manipulation"
