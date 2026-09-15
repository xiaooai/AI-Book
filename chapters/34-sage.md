# SAGE：自动生成机器人可以学习的世界

论文：SAGE: Scalable Agentic 3D Scene Generation for Embodied AI（2026）

## 01 · 先扩展机器人可以学习的世界

这篇是 2026 年 2 月的《SAGE: Scalable Agentic 3D Scene Generation for Embodied AI》。如果把你前面一直在读的 π 系列、GEN 系列、LingBot-VLA 放在一起看，SAGE 解决的不是“机器人模型应该长什么样”，而是一个更底层、甚至可能决定机器人 scaling 能不能继续的问题：**训练机器人需要海量不同的真实世界，但真实世界的数据太贵、太慢、也太危险；既然这样，我们能不能像生成文本和图片一样，自动生成“机器人可以真正进去训练”的三维世界？**注意这里关键不是做一张看起来漂亮的 3D 房间，而是生成一个**可以直接扔进物理模拟器里、物体不会穿模、不会悬空、质量和碰撞属性合理，而且还专门适合某个机器人任务训练的世界**。SAGE 的核心思想就是把“造仿真世界”从一个固定程序，变成一个会使用工具、会看结果、会自己返工的 AI agent。你告诉它“我要训练机器人把苹果拿起来放进碗里”，它不只是生成一间漂亮厨房，而会理解这个任务必须有苹果、碗、可接近的桌子和合理空间，然后不断调用 3D 生成、布局、视觉检查和物理模拟工具，直到这个世界既看起来合理，又真的能在 Isaac Sim 里稳定运行。([arXiv][1])

## 02 · 漂亮的场景，还要能用于训练

先用费曼法想象你要雇一个室内设计师，但这个设计师设计的房间不是给人看的，而是给机器人训练的。普通 3D 生成模型很像一个特别会画效果图的设计师：你说“生成一个卧室”，它可能给你一间非常漂亮的卧室，但枕头其实悬在床上 10 厘米，椅子腿插进地板，柜子和墙重叠，桌上的杯子一开物理模拟就直接掉下去。对于生成图片这些问题无所谓，可对机器人训练是致命的，因为机器人真的要碰这些东西。另一种程序化生成器则像一个极其保守的建筑工程师：东西很稳定，但永远从固定模板里拼“床 + 柜子 + 桌子”，开放世界多样性很差。SAGE 想把两者结合：**foundation model 负责想象和开放词汇，物理模拟器负责说“现实里行不行”，agent 负责在两者之间不停协调。**论文认为一个真正适合 embodied AI 的仿真环境至少要同时满足四件事：真实、多样、可直接进入模拟器，以及针对目标任务有用。([arXiv][2])

## 03 · Agentic Loop：用工具生成与返工

所以 SAGE 最核心的创新其实不是某一个 3D generator，而是一个 **agentic loop**。它通过 MCP 把不同能力包装成工具，agent 根据当前场景状态决定下一步该调用什么，而不是作者提前把“第一步做 A、第二步做 B、第三步做 C”写死。工具里最基本的四个角色分别是 Scene Initializer、Asset Placer、Asset Mover 和 Asset Remover。Scene Initializer 先看你的需求，决定房间多大、墙和地面长什么样、任务需要哪些物体；比如你说“pick an apple and place it in a bowl”，它就知道苹果和碗不能缺。Asset Placer 再根据文字条件去生成或放置物体；需要新的开放词汇物体时，会用 TRELLIS 从文本直接生成 3D asset，并用 VLM估计物体大小、质量、金属度、粗糙度等属性。放错了以后，Mover 可以重新安排位置，Remover 则把不应该出现的东西删掉。([arXiv][2])

## 04 · 视觉与物理：两种不同的验证

如果只有这些 generator，系统仍然会犯一个很典型的“生成式 AI 问题”：**每一步看起来都差不多合理，但错误会累积。**比如先放了一张床，再放床头柜，再放灯，再放枕头，最后可能整个房间语义上不协调，或者物理上根本不稳定。所以 SAGE 加了两个“审稿人”，一个管“看起来对不对”，一个管“物理上对不对”。第一个叫 **Visual Critic**，它看当前场景的 top-down 和多个角度渲染，检查房间是否完整、物体关系是否合理、有没有该加却没加的东西，甚至会建议“把这个物体移一下”“这里还缺一个东西”。第二个叫 **Physics Critic**，它不靠语言模型猜，而是真的把场景丢进 Isaac Sim，让重力和碰撞跑起来，看看物体会不会掉、穿模或者整体失稳。如果不稳定，它就把失败反馈给 agent，让 agent换更小的物体、换位置、重新生成。([arXiv][2])

这两个 critic 的分工很值得记住，因为它代表一种越来越普遍的 agent 思路：**不要指望生成器一次就生成对，而是把“生成”和“验证”拆开。**Visual Critic 更像美术总监，知道“这间厨房看着不完整”；Physics Critic 更像结构工程师，不关心漂不漂亮，只问“把重力打开以后会不会塌”。实验里只加 Visual Critic，会明显增加场景完整度和视觉质量；只加 Physics Critic，则把碰撞比例从 7.8% 降到 1.9%，稳定率提高到 99.6%；两个都用时，碰撞进一步降到约 **0.8%**，稳定率达到 **100%**。([arXiv][2]) 更大范围的三类室内场景统计里，SAGE 平均 collision rate 是 **1.9%**、stability **99.9%**，而 Holodeck 分别约 22.0% 和 63.8%，SceneWeaver 约 32.8% 和 67.7%。([arXiv][2]) 这很好地说明了一个道理：**“像真的”跟“在物理上是真的”是两种不同的能力。**

## 05 · 三层增强：让场景变化，任务不变

然后 SAGE 开始解决第二个真正和 scaling 有关的问题：一间好房间不够。假设机器人只在一间厨房里训练 100 万次，它最后很可能只是把这间厨房背熟了。所以作者设计了三层 augmentation。第一层是 **object configuration-level**：物体种类不变，但位置换，比如杯子每次出现在桌上不同位置。第二层是 **object category-level**：任务还是“拿杯子”，但杯子本身换形状、颜色、材质和外观；LLM 先扩写描述，再让 TRELLIS 生成新的 3D 对象。第三层是 **scene layout-level**：连整个房间几何、背景家具和无关物体都重新生成，只保留任务语义不变。([arXiv][2]) 用费曼法说就是：**不要让学生反复背同一道题，而是保持知识点不变，把数字、题型外观甚至整张试卷背景都换掉。**

这个设计最重要的点，是它把“场景多样性”和“任务一致性”分开控制。假如任务是“拿起桌上的可乐罐，移动到另一张桌子再放下”，场景可以从卧室变成办公室、家具全部重排，但训练系统必须保证：可乐罐仍然有地方可以拿、另一张桌子仍然能走到、机器人路径没有被家具堵死。普通生成式 3D 系统可能很会随机化，但随机到最后任务都不能做了；SAGE 的所有 augmentation 做完以后都重新跑 physics validation，因此生成的不只是 diverse scenes，而是 **diverse yet usable scenes**。([arXiv][2])

## 06 · SAGE-10k：规模化生产训练世界

为了证明它真的能 scale，作者还直接生成了一个 **SAGE-10k** 数据集：1 万个场景，横跨 50 种 room types 和 50 种 styles，总共包含大约 **56.5 万个独立生成的 3D objects**。([arXiv][2]) 这点特别值得和你之前读 GEN-0 对照。GEN-0 的 scaling 瓶颈是“怎样持续获得更多真实 physical interaction”；SAGE 则从另一个方向说：**如果真实世界太慢，能不能规模化生产训练世界本身？**GEN-0 像在扩建“机器人现实世界互联网”，SAGE 更像在造一个自动扩张的“机器人 Minecraft/模拟宇宙”。

## 07 · 从仿真场景到动作示范

但一堆漂亮稳定的 3D 世界还不是训练数据，因为机器人真正需要的是**动作 demonstration**。所以 SAGE 又自动把场景转成 robot trajectories。对于 Pick-and-Place，它先从 depth image 用 M2T2 找可能的 grasp pose，再通过 CuRobo 做 collision-aware inverse kinematics 和 motion planning；对于 Mobile Manipulation，则再加 RRT 规划移动底盘路径。轨迹执行过程中如果发现抓取失败、目标不可达、发生碰撞，或者物体最后没有到正确位置，就直接过滤掉。([arXiv][2]) 所以完整数据流水线是：**任务文字 → 3D 世界 → 物理验证 → 场景变体 → 运动规划 → 动作 demonstrations → imitation learning。**这个链条真正有野心的地方是，人理论上只需要给一句任务描述，剩下的数据工厂可以自动滚起来。

## 08 · 用两个任务检验数据引擎

论文为了验证这件事，选了两个比较清楚的任务。第一个是 Franka Panda 的 Pick-and-Place：把桌上的 mug 拿起来放进 bowl。第二个更长，是移动机械臂 Mobile Manipulation：机器人随机出生在房间某处，先导航到有可乐罐的桌子，抓起可乐，再移动到另一张桌子，最后放下。作者最终自动生成了 **2.8 万多条** Pick-and-Place demonstrations，覆盖 **264 个不同物体**；Mobile Manipulation 则接近 **5 万条 demonstrations、50 个不同场景**。([arXiv][2]) Policy 本身反而没有追求很复杂，用 Diffusion Policy / Robomimic 做 imitation learning，因为这里真正要测试的是：**数据引擎能不能把一个普通 policy 喂强，而不是靠一个特别强的 policy 掩盖场景质量。**([arXiv][2])

## 09 · 场景与示范增加，策略能力如何变化

结果最重要的不是某个最终成功率，而是出现了作者想要的 **scaling curve**：场景数量和 demonstration 数增加以后，policy 成功率持续上升，而且逐渐逼近那个拥有完整 3D 场景信息的 privileged motion planner。([arXiv][2]) 在 held-out scenes 上，Pick-and-Place 的 motion planner 测试成功率是 **57.7%**，学出来的 policy 已经达到 **50.0%**；Mobile Manipulation 的 privileged planner 是 **52.8%**，learned policy 是 **46.0%**。([arXiv][2]) 这个差距已经不算特别大，尤其要注意 learned policy 部署时只看到局部 RGB/depth 观察，而 motion planner 拥有完整场景几何。这说明合成 demonstration 不是只在 teacher 本身能做的状态里有用，它确实能蒸馏出一个从视觉直接行动的 policy。

## 10 · 比较基线，检验泛化能力

更值得看的是 baseline。作者构造了两个简化版本：一个大致模仿 SceneWeaver，去掉 physics critic 并把生成式 3D asset 换成检索；另一个进一步把 agent 换成固定 pipeline，类似 Holodeck 的思路。结果即使给它们**相同数量的场景和 demonstrations**，两个 baseline 的 scaling 速度仍然明显更慢，最终成功率不到 SAGE policy 的三分之一。([arXiv][2]) Cross-evaluation 更有意思：SAGE 训练出来的 Mobile Manipulation policy 在 SAGE 测试场景上成功率 **46.0%**，即使拿去 baseline 生成的场景里也有 39.1% 和 24.7%；反过来 baseline policy 在 SAGE 场景中只有约 14.4% 和 13.1%。([arXiv][2]) 这说明 SAGE 的优势不是简单“训练分布和测试分布一样所以背得好”，而是**多样且物理可靠的训练世界真的提高了泛化**。

## 11 · 质量控制也必须随规模扩展

这里就能看出为什么 physics validation 不只是“让图片好看”的工程细节。如果仿真世界里有大量桌子轻微穿模、杯子物理属性奇怪、抓取点不稳定，你仍然可以生成几万条动作轨迹，但 policy 学到的会是一个有系统偏差的“假物理世界”。Scaling 这种数据反而可能把错误放大。所以 SAGE 的一个深层观点是：**合成数据的 scaling 只有在质量控制也能 scaling 时才有意义。**这和你前面读语言模型数据清洗、DeepSeek scaling law 的逻辑是一样的：100 万条好数据和 1000 万条垃圾不是线性关系；机器人更敏感，因为错误场景不仅语义不对，还可能违反物理规律。

## 12 · 让智能体协调生成与物理工具

这篇的 agentic 思路也值得和传统 procedural generation 对比。过去 ProcTHOR、Infinigen 这类程序化方法的优势是稳定、便宜、物理结构可控，但开发者必须提前写很多规则；生成一个“赛博朋克游戏室”这种开放词汇场景时非常麻烦。纯 LLM/3D generator 则很灵活，但物理一致性差。SAGE 的做法有点像让 LLM 当项目经理：**规则不再完全写死在代码里，而是 agent 根据场景状态动态选择工具；真正不能妥协的底线——碰撞、重力、稳定性——仍由硬物理模拟器负责验证。**所以它不是“让 LLM 取代 simulator”，而是让 LLM 决定什么时候该调用 simulator。这种“neural reasoning + deterministic tools + critic loop”的组合，其实和现在 agentic coding、科学发现 agent 的范式高度相似。([arXiv][2])

## 13 · 开放词汇与长尾场景

它也展示了很强的 open-vocabulary scene generation：除了 bedroom、kitchen、living room，还可以直接要求 gym、office、cyberpunk game den、starry-night bedroom 等风格；系统还可以从参考图片里用 Qwen3-VL抽取风格和物体属性，实现 image-conditioned generation，或者接 PartNet-Mobility 生成包含可开关抽屉、柜门等 articulated objects 的场景。([arXiv][2]) 这点对机器人很重要，因为真正想 scale 时，不能永远只训练“苹果、杯子、桌子”这几个 benchmark 物体。语言和生成模型的价值就在于可以打开长尾对象和长尾房间类型。

## 14 · 数据基础设施，而非新的机器人策略

但要注意，这篇和你最近读的 π0.7、GEN-1.5 很不一样。SAGE **不是新的 VLA foundation model，也没有证明一个 policy 可以 zero-shot 做几百个任务**；它的机器人实验其实只有两个代表任务，而且 Mobile Manipulation 还被拆成四个 sequential policies。([arXiv][2]) 它真正贡献的是数据基础设施：**怎样自动制造适合 embodied learning 的 world + demonstration。**所以更准确地说，它研究的是“foundation environment/data engine”，而不是“foundation policy”。

## 15 · 真实经验与合成世界的两条路线

如果把它和 GEN-0 放在一起，你会看到 embodied AI 现在有两种非常清楚的 scaling 哲学。GEN-0/GEN-1 的路线是：**尽可能收更多真实 physical interaction，因为真实身体经验才是 physical intelligence 的燃料。**SAGE 的路线则是：**现实世界数据太贵，所以把 simulator 生成链条自动化，先 scale worlds，再 scale demonstrations。**前者数据贵但真实，后者便宜、可控、可以无限随机化但有 sim-to-real gap。真正未来的系统很可能是两者闭环：大量 simulation pretraining → 少量 real-world grounding → 找真实失败模式 → 让 scene agent专门生成这些 failure cases → 再训练。

## 16 · 从 TANGO 的 PET 到通用数据引擎

这也和你刚读的 TANGO 有很直接的关系。TANGO 为了训练 humanoid 在复杂障碍里侧身、下蹲、跨步，自己构造了 PET 管线来自动制造“难导航场景 + whole-body demonstrations”；SAGE 则把这个思想往更一般方向推：**以后不应该每篇机器人论文都手工写一个 PET，而应该有一个 agent，根据你的任务自动替你造整个训练世界和老师轨迹。**比如你未来说“我要训练 humanoid 在拥挤厨房里一边端盘子一边避障”，理想中的 SAGE 类系统应该自动生成不同厨房、不同桌椅布局、不同盘子、不同物理属性，再生成候选轨迹和训练数据。SAGE 现在还没做到这么复杂，但它代表的是这个方向。

## 17 · 局限：仿真验证、任务范围与生成速度

论文目前的限制也非常明确。它主要集中在**室内、刚体物理**，虽然可以接 articulated assets，但柔性物体、液体、衣服、户外环境都还不是主能力；自动 action generation 目前主要还是 pick、place、navigation 的组合；实验完全在 simulation 里，没有展示真正 sim-to-real 到实体机器人的闭环。作者自己也把 online RL 和 real-robot closed-loop validation 列为下一步。([arXiv][2]) 另外，生成速度也还谈不上“按一下立刻有”：TRELLIS 单个物体原本约 15 秒，8 GPU 并行后平均约 2–3 秒；Isaac Sim 每个 placement candidate 验证大约 1–2 秒，一个约 20 个物体的完整场景平均仍要**约 10 分钟**生成。([arXiv][2]) 这对离线大规模数据工厂完全可以接受，但离实时 world generation 还很远。

## 18 · 用一分钟讲给朋友听

如果现在合上论文，用费曼法一分钟给朋友讲，我会这样说：**SAGE 不是在训练一个更聪明的机器人，而是在造一台“自动生产机器人训练世界”的机器。你只需要说“我要训练机器人把杯子拿起来放进碗里”，一个 AI agent 就会决定房间里需要什么，调用 3D 生成器造家具和物体，再让视觉 critic 检查场景像不像真的，让 Isaac Sim 的 physics critic 检查东西会不会穿模、掉下来；发现问题以后它自己移动、删除或重新生成物体。等一个世界通过检查后，再自动改变物体位置、形状和材质，甚至重新生成整个房间布局，扩成大量不同训练场景；最后用抓取预测、IK 和路径规划自动生成 robot demonstrations，再训练 Diffusion Policy。作者生成了 1 万个场景、约 56.5 万个独立 3D 物体，并在 pick-and-place 和 mobile manipulation 上看到明显 scaling：场景和示范越多，policy 越接近拥有完整 3D 信息的 privileged planner。**([arXiv][2])

所以这篇我建议最后只记一句话：**不要只 scale 机器人模型，也要 scale 机器人可以学习的“世界”。** 📖 更技术一点就是：**Task prompt → Agentic scene generation → Visual/Physics critics → Scene scaling → Automatic demonstrations → Policy scaling。**如果 GEN-0 问的是“机器人有没有自己的互联网级经验数据”，那 SAGE 提供了另一种答案：**也许其中很大一部分互联网，不需要等待现实世界慢慢发生，而可以由 agent 自动生成、自动验证、自动变成训练数据。**

[1]: https://arxiv.org/abs/2602.10116 "[2602.10116] SAGE: Scalable Agentic 3D Scene Generation for Embodied AI"

[2]: https://arxiv.org/html/2602.10116v2 "SAGE: Scalable Agentic 3D Scene Generation for Embodied AI"
