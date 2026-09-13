# Motus2：提出动作、预演后果、评价并学习

论文：Motus2: A Self-Evolving General World Model for Dexterous Manipulation（2026）

## 01 · 会做、会想象、会评价

这篇是《Motus2: A Self-Evolving General World Model for Dexterous Manipulation》。如果把你前面读过的 π*0.6、RLT、MEM、MotionWAM 这些工作放在一起，Motus2 想解决的是一个更“闭环”的问题：**机器人不应该只有一个会出动作的 policy，它还应该能想象“如果我这么做会发生什么”，再判断“这个结果好不好”，最后把这种判断重新拿回来改进自己的 policy。**也就是说，一个真正能自我进化的机器人系统，至少需要三种能力：**会做、会想象、会评价。**过去很多 world model 已经能预测未来，很多 VLA 也会直接输出动作，但两者往往还是两个松散模块，甚至“预测未来”只是辅助训练目标，并没有真正参与决策和学习。Motus2 的核心创新就是把 **Policy、Simulator、Evaluator** 三个角色塞进**同一套共享参数的 General World Model** 里：Policy 提议几种未来动作，Simulator 分别想象执行这些动作后世界会变成什么样，Evaluator 再判断哪种未来更接近任务目标，然后既可以当场选更好的动作，也可以把这种评分变成训练信号，让下一版 policy 本身变得更好。([Scirate][1])

先用费曼法把它想成一个人在拧灯泡。普通 VLA 更像：“我看到灯泡和灯座，根据过去模仿数据，我直接伸手去拧。”Motus2 更像一个会在脑子里预演的人：“我可以稍微往左一点抓，也可以右一点抓，也可以先调整手指；如果选方案 A，灯泡可能会歪；方案 B 可能更对齐；方案 C 可能抓不稳。那我先做 B。”更进一步，如果它反复发现“这种抓法总能让任务进度提高”，它不只是每次临场选择 B，而是会**真的更新自己的 policy，让以后更容易直接提出 B 这种好方案。**这就是这篇论文所谓的 self-evolving：不是“模型每次部署以后自动无限自我重写”，而是 policy、world simulation、value evaluation 形成一个可以不断产生训练信号的闭环。([Luca's Proxy][2])

## 02 · 动作、未来与价值的信息流

这三个角色在数学上其实非常直观。给当前上下文 \(c_t\)，Policy 先产生一段动作 \(A_t\)；Simulator 再根据 \(c_t+A_t\) 预测未来视觉状态 \(Z_t\)；Evaluator 最后根据当前状态、动作和预测未来，估计一个任务进度值 \(Y_t\)。论文把这个因果关系写成 **Action → Future → Value**。最重要的是，这三条链不是三个独立网络，而是**同一个 shared-parameter model 的三种“使用接口”**。作者用 attention mask 把信息流控制得很严格：生成动作时不能偷看未来画面；模拟未来时可以看动作；评价时可以同时看动作和未来。这样就避免了一个很常见的问题：训练时 future frame 已经存在，如果动作头能看到未来，那它就等于考试时偷看答案。([Luca's Proxy][2])

这里可以把它和你刚读过的 MotionWAM 对比一下。MotionWAM 也用 world model 的未来动态帮助动作，但更像“**world model 给 policy 提供更好的 dynamics representation**”；Motus2 更进一步，把 future prediction 直接变成一个**可操作的模拟器接口**。不是只说“我的 hidden state 里有未来感”，而是真的可以：给我动作 A，我预测后果；给我动作 B，我预测另一个后果；然后比较。这就开始接近经典 model-based planning 的味道。你可以把 MotionWAM 想成“会预感未来”，Motus2 更像“会拿几个候选未来做比较”。

## 03 · 用多个候选进行当场规划

这也让 test-time planning 变得非常自然。实际推理时，Policy 可以一次提出 \(N\) 个不同 action chunks，Simulator 分别生成它们可能造成的视觉后果，Evaluator 给每个分支打一个“任务进度分”，最后选分最高的那条执行。这就是 **Best-of-N planning**。执行完以后不继续相信很长的幻想，而是重新看真实世界，再规划下一步，所以它本质上是一种 receding-horizon planning。([Luca's Proxy][2]) 用费曼法说：不是一次性幻想“未来五分钟怎么做”，而是“先想象接下来这一小步的几个选择，选最好的，真的做完，再重新看世界”。这比长时间完全依赖 imagined rollout 稳得多，因为 world model 一旦预测错，误差不会无限滚下去。

## 04 · 在想象中改进动作策略

但只做 planning 还有一个限制：**你只是从 policy 现有的候选里挑最好的，policy 本身没有变。**如果它永远只会提出一堆很一般的动作，你再怎么选也有上限。所以 Motus2 第二步是用 **DiffusionNFT** 做 model-based RL。Evaluator 给候选动作打分，高分动作让 policy 概率往它们靠，低分动作则让 policy 远离；同时保留一个 reference policy，避免更新一步跑太远。([Luca's Proxy][2]) 这点和 DeepSeek-R1 的 GRPO、π*0.6 的 RECAP 很有呼应：都是“先生成多个行为，再用某种相对质量信号重新塑造分布”。区别是 Motus2 的 reward 不只是直接看真实最终结果，而是**先让 world model 模拟后果，再让 value model评价后果**，因此可以在 imagination 里产生额外的 policy improvement 信号。

这也是为什么这篇真正叫 **model-based reinforcement learning**。Model-free RL 更像“我真的做了这一步，最后成功/失败，所以以后提高/降低这一步概率”；Motus2 则多了一层“我可以先在内部模拟几种做法，再用 predicted outcome 训练”。如果 simulator 足够准确，这就能减少真实机器人试错成本。尤其灵巧手任务里，一个失败 rollout 很贵：物体可能掉了、场景要 reset、还可能损坏东西。World model 给你的价值，就是可以把很多“如果我这么做呢？”留在脑内先试。

## 05 · 失败数据不必用来模仿

Motus2 还有一个我很喜欢的训练设计：**不是所有数据都应该拿来模仿。**真实机器人收数据时，有成功轨迹，也有失败、笨拙、低质量轨迹。传统 behavior cloning 如果把失败轨迹也当 action target，会直接教机器人“失败时应该这样做”。很多系统因此简单把失败数据删掉。但 Motus2 说，失败数据虽然不值得模仿，却仍然非常有价值：它告诉你“这个动作会导致什么后果”，也告诉你“这种后果不好”。所以作者做了 trajectory-dependent supervision routing：**成功、精心筛选的 demonstration 才监督 Policy；失败和 suboptimal trajectory 则主要训练 Simulator 和 Evaluator。**([Luca's Proxy][2]) 这句话很值得记住：**坏动作不是好老师，但它是好教材。**

费曼式地讲，学生做错一道数学题，你不会把错误解法作为“请照着写”的标准答案，但这个错误仍然非常有教育价值：你能学到“这么想会走到哪里”“为什么不对”。Motus2 在机器人里也是一样：失败 rollout 不该训练“怎么行动”，但非常适合训练“世界在这种动作后会怎样变化”和“这种变化离任务目标更近还是更远”。这让数据利用率明显提高，因为现实机器人最宝贵、最容易被浪费的数据，恰恰就是那些失败经验。

## 06 · 从第一视角人类数据到机器人

数据侧也是这篇很大的重点。Motus2 建了一个大约 **13 万小时 egocentric human interaction corpus**，其中既有大量单目第一视角视频，也有同步双目第一视角视频。单目数据负责“广”，包括大量不同环境、物体和操作；双目数据则额外提供隐式深度线索，而且更容易恢复准确的 3D 手部姿态。总数据里大约有 10 万小时级别的单目来源，双目则来自 Ropedia、EgoScale、LightWheel、JD-Group、CyberOrigin 等多个数据源，总计达到万小时级。([Luca's Proxy][2]) 训练流程也很有层次：先大规模单目视频预训练，再加入同步双目和动作，最后才进入机器人 domain 的 mid-training，用机器人轨迹和专门收集的人机对齐数据做 grounding。([Luca's Proxy][2])

## 07 · 双目数据的物理信息与扩展

为什么特别强调**双目**？因为灵巧操作和“拿杯子”这种粗粒度 manipulation 很不一样。手指之间几毫米的深度误差，就可能导致抓空、打滑或者碰撞。单目图像可以学很多语义，但真实 3D 几何很难完全确定；左右眼的视差则天然给你更强的深度信息。所以 Motus2 的 scaling 不只是“数据小时数更多”，还在升级**数据的物理信息量**：从 monocular 到 stereo，再到 tactile。可以把这条线记成：**先看得多，再看得立体，最后摸得到。**

而且作者真的测了 stereo egocentric data 的 scaling law。他们用 2k、4k、10k、20k 小时的嵌套双目数据子集做预训练，然后看 held-out human action prediction error。结果随着数据从 2k 增加到 20k，小数据模型更早 plateau，大数据模型能达到更低验证误差；最优误差和数据规模的 log 之间近似线性下降。([Luca's Proxy][2]) 这和你前面 GEN-0、LingBot-VLA 一直看到的趋势很一致：**机器人/具身领域还远没有进入“数据加了也没用”的饱和区。**更有意思的是，这里不是纯机器人遥操作数据，而是 stereo human egocentric data，也就是说“人类身体经验”也在出现可测量 scaling trend。

## 08 · 灵巧操作实验与领域适配

主实验也很直接。作者选了五个目标机器人任务：Place Ball、Put Phone、Attach Eraser、Screw Bulb、Multi-Finger，覆盖空间放置、对齐、持续接触和多指协同。所有方法用相同 target-task SFT 数据。结果如果从一个普通 WAN 视频 backbone 直接 SFT，平均成功率是 0%；先经过大规模 egocentric pretraining，再 SFT，平均到 **51%**；再加机器人 domain 的 mid-training，Motus2 平均达到 **84%**，其中 Place Ball 和 Attach Eraser 是 100%，Screw Bulb 90%，Multi-Finger 70%，Put Phone 60%。([Luca's Proxy][2]) 这组实验最想说明的是：**互联网/视频基础模型不是直接就会机器人，真正有效的是“人类交互预训练 → 机器人 grounding → 具体任务适配”这个层次化过程。**

这里一个很扎眼的结果是 π0.5 在这组 matched evaluation 里是 0%。([Luca's Proxy][2]) 这个数字不要简单读成“Motus2 全面碾压 π0.5”，因为这是作者自己设计的高自由度灵巧手任务和特定 observation/action interface，对 π0.5 并不是它原始最擅长的设置。更合理的读法是：**这套实验专门强调 stereo egocentric、dexterous hand、multi-finger/contact-rich manipulation，而 Motus2 的训练数据和结构就是针对这些东西设计的，所以 domain alignment 带来很大优势。**

## 09 · 规划与强化学习分别贡献什么

然后看“自我进化”到底带来多少收益。作者在 Put Phone 和 Multi-Finger 上做了一个很干净的 2×2 实验：不开 planning、不开 MBRL，平均 65%；只加 Best-of-N planning 到 **67.5%**；只加 MBRL 到 **72.5%**；两个都加到 **75%**。([Luca's Proxy][2]) 这个结果很有意思，因为 planning 和 RL 的作用被拆开了：planning 是**不改参数，在 inference 时选更好的候选**；MBRL 是**真的改 policy，让候选分布本身变好**。二者还能叠加，说明它们不是完全重复的机制。

这和你之前读 reasoning LLM 时特别像。对于 LLM，可以通过 best-of-N / reranking 提高当次回答，也可以通过 RL 真正改变模型参数；前者像“同一个脑子多想几次，选最好”，后者像“训练以后，下次第一次就更容易想对”。Motus2 把同一种逻辑放到了机器人动作上。Policy 先“想几个动作”，world model 想象未来，value model当 judge；如果只是 test-time planning，就挑最好的执行；如果进入 MBRL，就把 judge 结果反过来更新 policy。**机器人 world model 开始同时承担 test-time reasoning 和 training-time learning 的基础设施。**

## 10 · 完整历史与压缩记忆的取舍

然后是记忆。Dexterous manipulation 特别容易被自身手遮挡，比如手抓住一个物体以后，当前相机里反而看不到接触点；某些关键信息几秒前出现，现在已经离开视野。所以论文比较了三类 working memory 思路：固定 sliding window、global autoregression、hybrid memory。Global autoregression 会保留完整历史视觉 latent，让每一步都能访问所有过去；hybrid memory 则只保留最近窗口和少量压缩 memory tokens。([Luca's Proxy][2]) 实验在 Find Square 和 Press Button 两个记忆任务上做：真实机器人里 global autoregression 平均 **57.5%**，hybrid memory 只有 **25%**；仿真也是 78% 对 52%。([Luca's Proxy][2])

这个结果挺有意思，因为你前面读 MEM 时看到的是“长期历史最好压缩成高层 memory”；这里却发现**完整 global autoregression 更强**。两者并不矛盾。MEM 面对的是十几分钟长任务，完整保留所有高清历史几乎不现实；Motus2 这组 probe 更集中在较短的、细粒度视觉证据，压缩 memory 可能丢掉关键像素细节。所以可以得到一个更通用的结论：**记忆不是越压缩越好，也不是越完整越好，而是取决于未来决策到底需要“语义事实”还是“精细视觉证据”。**找一个之前看过的小方块，可能真的需要保留原始视觉；记“我已经洗过盘子”，语言摘要就够了。

## 11 · 触觉专家补充接触信息

然后是这篇另一个重要维度：**触觉。**视觉再强，也有一种信息永远很难看出来——“到底碰到了没有、抓得紧不紧、纸有没有撕开、杯子是不是开始滑”。所以 Motus2 在大 backbone 之外加了一个 lightweight tactile expert。它不每来一次触觉就重跑整个巨大模型，而是让 backbone 先生成一段中间 action chunk，真正执行每个小 sub-chunk 前，再用最新触觉对动作做细调。([Luca's Proxy][2]) 这特别像人伸手抓纸杯：视觉先告诉你“手大概伸到这里”，真正接触以后，指尖压力告诉你“再松一点，否则捏变形”“现在抓住了，可以往外拉”。

而且 tactile expert 训练时不只学“根据现在的力修正动作”，还会预测**执行这个动作以后接下来会出现什么触觉**。这个 force prediction 只作为训练信号，部署时不用真的生成未来触觉。([Luca's Proxy][2]) 这和 MotionWAM “预测未来视频是为了学动态表示，不一定需要真的把未来视频用出来”的思想非常像：**prediction 是训练手段，好的 physical representation 才是资产。**实验里，Pull Out Paper Cup 成功率从 65% 到 75%，Tear Paper 从 55% 到 70%，平均从 **60% 提升到 72.5%**。([Luca's Proxy][2]) 对高自由度手来说，这 12.5 个百分点说明“看见”真的不能完全替代“摸到”。

## 12 · 不同数据监督不同目标

这篇论文还有一个很值得你记住的“数据哲学”：它把**不同质量的数据真正分工**了。高质量 expert demonstration 教 policy“应该怎么做”；失败数据教 simulator“这么做世界会变成怎样”；value labels 教 evaluator“这种未来到底好不好”；人类第一视角视频提供巨大 dexterity prior；双目提供 3D；机器人数据提供 embodiment grounding；触觉提供 contact truth。换句话说，作者不再追求“找一种万能数据，全部拿来 behavior cloning”，而是把 heterogeneous data 按“它究竟知道什么”来使用。([Luca's Proxy][2]) 这和 π0.7 “给数据增加 context 后再利用”、Ψ₀ “不同数据放到不同训练阶段”其实是同一个大趋势：**数据本身没有绝对好坏，关键是不要让它监督错误的目标。**

## 13 · 从动态表示到决策闭环

如果把 Motus2 放回你最近读的 world-model 路线里，会看到它和几篇论文分别占了不同位置。**MotionWAM** 的重点是 world-model hidden dynamics 帮助 whole-body action；**ω-0** 强调 future latent 和 whole-body humanoid；**Motus2** 则把 world model 从“更好的 action representation”进一步推进成了真正的**决策闭环**：Policy 提议，Simulator 预演，Evaluator 打分，然后 Planning 负责当场选择，MBRL 负责长期改进。可以把它压成一句话：前两者更像“world model 帮我更会动”，Motus2 更像“world model开始帮我做选择和学习”。

## 14 · 真实反馈与想象反馈的取舍

和 π*0.6 / RECAP 比也很有意思。RECAP 是**真实 deployment experience → value → advantage → policy improvement**；Motus2 则更 model-based：**Policy 先生成候选 → Simulator 想象 → Evaluator 打分 → policy improvement**。一个更依赖真实 rollout，一个更依赖 imagined rollout。如果 simulator 够准确，Motus2 理论上能少做很多真实试错；但如果 world model 幻想不准，policy 就可能被错误 imagined reward 带偏。所以两条路线背后的核心 trade-off 是：**真实世界反馈贵但真，想象反馈便宜但可能错。**

## 15 · 当前局限与智能体学习循环

这也是 Motus2 目前最大的风险。它的 MBRL 增益是真实测出来的，但幅度还不是那种“从 30% 到 90%”的巨大跨越：65% 基线加 planning 到 67.5%，加 MBRL 到 72.5%，两者一起 75%。([Luca's Proxy][2]) 这说明闭环方向有效，但 simulator/value 的精度还远没有达到“靠脑内模拟就能彻底替代真实训练”的程度。另外，完整 global autoregression 虽然更强，但它的 KV cache 和 attention 会随着 episode 长度不断增长，长期任务成本会越来越高；触觉数据又很难 scale，因为人手和机器人手的形态不同，传感手套变形本身还会产生噪声。论文自己把这些都列为主要限制。([Luca's Proxy][2])

如果把整条历史拉长一点，你会发现 embodied AI 正在经历一个和语言模型很像的转变。早期机器人是 **Observation → Action**；π0 时代是更强的 **Vision + Language → Action**；MEM 加了 **History**；π*0.6 加了 **Experience → Improvement**；MotionWAM 加了 **Future Dynamics**；Motus2 则试图统一成：**Past + Present → Candidate Action → Imagined Future → Value → Better Action → Better Policy。**这已经不再只是一个“动作预测模型”，而是一个非常初步的**agent learning loop**。

## 16 · 用一分钟讲给朋友听

如果现在合上论文，用费曼法一分钟讲给朋友听，我会这样说：**Motus2 想做一个不仅会动作，还会“想象后果、判断好坏、从这种判断中变强”的机器人世界模型。它把同一个模型当成三个东西用：Policy 生成几种可能动作，Simulator 预测每种动作之后视觉世界会怎么变化，Evaluator 给这些未来打任务进度分。推理时可以像下棋一样生成多个候选，先在脑子里模拟后再选最好的；训练时还可以把这些分数通过 DiffusionNFT 反过来更新动作 policy，让以后更容易直接提出好动作。它同时把失败数据留下来：失败动作不拿去模仿，但可以用来学“这种动作会导致什么结果、这个结果有多差”。数据方面，它用了大约 13 万小时第一视角人类操作数据，从单目扩展到双目，再用机器人和人机对齐数据做 grounding；还加入长历史记忆和触觉 refinement。最终 egocentric pretraining 把五个灵巧任务平均成功率从 0% 推到 51%，机器人 mid-training 再到 84%；在额外的策略优化实验里，MBRL 和 test-time planning 还能把 65% 基线推到 75%；触觉又让接触密集任务平均提高 12.5 个百分点。**([Luca's Proxy][2])

所以这篇我建议最后只记一句话：**Motus2 想把机器人从“会模仿动作”推进到“会提出动作、想象结果、评价结果，再用结果反过来改进自己”。**📖 更技术一点就是：**Policy → Simulator → Evaluator → Planning/RL → Better Policy。**如果 MotionWAM 的 world model 更像“给机器人未来直觉”，那么 Motus2 的 world model 已经开始真正扮演一个**内部试验场**：先在脑内尝试，再把成功和失败都变成下一轮学习的燃料。

[1]: https://scirate.com/?date=2026-09-06&page=120&range=31&utm_source=chatgpt.com "Top arXiv papers"
[2]: https://vpn.luca-server.com/px/aHR0cHM6Ly9hcnhpdi5vcmcvaHRtbC8yNjA4LjMwMjM3djE "Motus2: A Self-Evolving General World Model for Dexterous Manipulation"
