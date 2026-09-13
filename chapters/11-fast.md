# FAST：用压缩定义机器人的动作语言

论文：FAST: Efficient Action Tokenization for Vision-Language-Action Models（2025）

## 01 · 动作应该怎样变成 token？

这篇是 2025 年 1 月的《FAST: Efficient Action Tokenization for Vision-Language-Action Models》。如果你刚刚读完 π0，这一篇其实像是在回答 π0 留下的一个非常具体的问题：**π0 为什么要用 flow matching，而不是直接像 GPT 一样，把机器人的动作也变成 token，然后一个 token 一个 token 往后生成？**过去很多 VLA 的直觉确实是“语言可以 token 化，动作也可以 token 化”，但作者发现，机器人动作尤其是 20Hz、50Hz 这种高频连续控制，如果用传统“每个时间步、每个动作维度分别离散化”的办法，会产生大量高度重复、彼此极其相似的 token，结果 Transformer 很容易学会一个没什么用的捷径：**复制上一步动作。**于是这篇论文真正研究的，不是重新发明一个机器人基础模型，而是研究一个看似不起眼、实际非常关键的问题：**机器人动作到底应该怎样被切成 token，才能真正适合 next-token prediction？**作者提出的答案就是 FAST，Frequency-space Action Sequence Tokenization，用频域压缩把连续动作先“压缩”，再 token 化。最终它让 autoregressive VLA 能学会过去很难学的高频灵巧操作，而且和 π0 结合后，在约 1 万小时机器人数据上达到和 diffusion/flow-based π0 接近的表现，同时训练最多快约 5 倍。([arXiv][1])

## 02 · 高频动作中的复制捷径

先把传统方法的问题想清楚。假设一个机械臂以 50Hz 控制，也就是一秒钟要输出 50 次动作；双臂机器人可能每个时间点有 14 个连续动作维度。最简单的 tokenization 是：每个动作维度都切成 256 个小格子，比如关节角度落在哪个格子就对应哪个 token。那一秒钟的动作就会变成 \(50\times14=700\) 个 token。论文里的 T-shirt folding 恰好就是这个量级：传统方法一秒动作平均需要约 **700 个 token**。问题还不只是 700 很长，而是这 700 个 token 之间高度相关。机器人在连续抬手时，上一帧关节角是 0.301，下一帧可能只是 0.304，再下一帧 0.307。于是模型做 next-token prediction 时，很容易发现：“不用理解我要怎么折衣服，直接预测下一个动作和刚才差不多就行。”训练 loss 看起来甚至会下降得很好，但真正 rollout 时模型只是维持或复制动作，根本没学到完整轨迹。作者用一个简单的曲线插值实验验证了这一点：同样一条连续曲线，采样频率越高，传统 binning tokenization 的 autoregressive 模型反而学得越差，最后甚至退化成复制第一个动作。([arXiv][1])

这个现象非常值得停一下，因为它和你前面读 GPT 的经验刚好相反。语言中，“猫坐在桌子上”每个 token 通常都带来一些新的信息；但高频机器人动作中，连续两个时间点可能几乎完全一样。论文把它解释成 **marginal information content 太低**：已经知道前面动作以后，下一个 token 几乎没有多少新增信息。([arXiv][1]) 用费曼法说，就是老师每天给学生出这样的题：“昨天数字是 100，今天是多少？”答案 101；下一题“101 后面是多少？”答案 102。学生很快能把卷子做得很漂亮，但他根本没有学会数学，只学会“加一”。所以 FAST 的关键不是让 Transformer 更大，而是**重新设计题目，让每一个预测 token 都更有信息。**

## 03 · DCT：从时间域转向频率域

作者想到的办法特别像 JPEG 压缩图片：既然机器人动作是一段平滑的时间序列，那我们为什么一定要在“时间域”逐帧记录？可以先把它变换到**频率域**。这就是 DCT，Discrete Cosine Transform，离散余弦变换。你可以把一段一秒钟的机械臂轨迹理解成一条曲线，而 DCT 会告诉你：这条曲线可以由哪些不同频率的余弦波叠加出来。最低频的系数描述“大方向”——比如这一秒手臂整体从左向右移动；稍高频描述中等尺度弯曲；很高频才描述微小抖动和快速变化。真实机器人动作通常比较平滑，所以绝大多数信息集中在少数低频系数里，高频系数往往很小。([arXiv][1]) 这和 JPEG 很像：照片里相邻像素通常相似，所以不必把每个像素完全独立保存，可以先转频域，主要保存重要频率成分。

## 04 · 量化与 BPE：压缩动作序列

于是 FAST 的第一步，就是把一段 action chunk 的每一个动作维度都先做 DCT。第二步，把很小的频率系数通过 scale-and-round 量化掉，于是大量系数变成 0；第三步，再把这些系数排成序列，而且刻意**先放低频成分，再放高频成分**，这样 autoregressive 模型首先预测“整段轨迹的大轮廓”，然后才逐渐补细节；最后再使用你在 GPT tokenizer 里已经见过的 **BPE，Byte Pair Encoding**，把频繁出现的系数组合进一步合并成更少的 token。([arXiv][1]) 所以 FAST 可以压缩成一个非常好记的流程：**Action trajectory → DCT → quantization → BPE → action tokens。**

为什么还要在 DCT 后面再加 BPE？因为 DCT 虽然把主要信息集中到了少量系数，却会产生大量 0。假设压缩以后是一串“12, -3, 1, 0, 0, 0, 0, 0……”；如果仍然一个系数一个 token，那 Transformer 还是得浪费很多步骤预测 0。BPE 就像语言里把频繁出现的字符组合压成一个词一样，把这些反复出现的系数组合压缩掉。论文消融实验也发现，只做 DCT、不做 BPE 已经比传统 tokenization 好，但 rollout 表现仍然明显差于完整 FAST，因为大量重复的 0 会稀释学习信号，同时使 autoregressive inference 变慢。([arXiv][1]) 所以 DCT 负责“把信息集中起来”，BPE 负责“把冗余真正从 token 序列里挤出去”。

## 05 · 从 700 个 token 到 53 个

这个压缩到底有多明显？看几个数字就很直观。对于低频的 BridgeV2 5Hz 数据，传统方法一秒大约需要 35 个 action tokens，FAST 约 20 个，压缩 1.75 倍；DROID 15Hz 从 105 降到 29，大约 3.6 倍；Table Bussing 20Hz 从 140 降到 28，约 5 倍；最夸张的是双臂 T-shirt folding 50Hz，从 **700 个 token 降到大约 53 个**，压缩约 **13.2 倍**。([arXiv][1]) 更有意思的是，不管控制频率怎么变化，FAST 最终往往每只机械臂一秒只需要大约 30 个 token。作者认为这说明 FAST 捕捉到的是动作轨迹本身的“真实复杂度”，而不是被采样频率绑架。换句话说，同一个抬手动作，你用 10Hz 录或者 50Hz 录，物理动作本身并没有复杂 5 倍；传统 tokenization 却制造出 5 倍 token，FAST 则尽量消掉这层人为冗余。

## 06 · 是架构问题，还是表示问题？

这里就能看懂这篇论文和 π0 的关系了。π0 原论文选择 flow matching 的一个重要原因，就是连续、高频、灵巧动作不太适合简单的 autoregressive token generation。FAST 的作者却问：**会不会不是 autoregressive modeling 本身不适合机器人，而只是我们的 action tokenizer 太差？**实验结果相当支持这个观点。在 20Hz 的 table bussing 和 50Hz 的 T-shirt folding 上，传统的 naïve tokenization 几乎学不出有效策略；换成 FAST以后，同一个 autoregressive Transformer 就能学起来。([arXiv][1]) 所以这篇论文实际上是在替“GPT 式机器人”翻案：**不是 next-token prediction 天生不能做灵巧动作，而是先得把动作表示成适合预测的 token。**

## 07 · FAST+：通用的动作词典

然后作者又更进一步，提出 **FAST+**，一个“通用机器人 action tokenizer”。如果 FAST 每换一种机器人还要重新训练一个 BPE 词表，那仍然不像自然语言 tokenizer 那么方便。于是作者收集了约 **100 万段、每段一秒的真实机器人动作**，包含单臂、双臂、移动机器人，不同 action space、不同控制频率，训练一个统一的 BPE vocabulary。([arXiv][1]) 然后拿一些 tokenizer 从来没见过的新机器人数据测试，FAST+ 依旧普遍能把 action token 数至少压缩约 2 倍，而且在很多数据上压得更多；拿 FAST+ 直接训练策略时，效果也基本接近为每个数据集单独训练的 FAST tokenizer。([arXiv][1]) 这一步的意义很像语言里的通用 tokenizer：以后造 VLA，也许不用每个机器人自己发明一套“动作词典”，直接拿一个通用动作 tokenizer 就行。

这就带来了一个很有意思的概念：**机器人动作也许真的存在某种“词汇”。**当然 FAST token 并不是“抓”“放”“左移”这样人类可读的语义词，它更多是频域中的运动模式。但从建模角度，它起到了类似语言 token 的作用：把一长串原始连续信号压成较少的、信息密度更高的离散符号。你可以想象，一个 token 可能隐含表达“这一秒手腕整体平滑向左移动，并略微旋转”这种轨迹模式，而不是只表达“此刻关节角是 0.371”。这就是 FAST 最深的思想：**好的 token 不应该忠实地逐采样点复述现实，而应该把重复的低层信息压缩掉，让模型预测真正有意义的变化。**

## 08 · 跨模型验证与 DROID 泛化

作者还把 FAST 放进 OpenVLA，而不只是 π0，结果在高频 T-shirt folding 上同样明显改善训练。([arXiv][1]) 这很重要，因为它说明 FAST 不是专门针对 π0 架构写的 trick，而更像一个可以插到各种 autoregressive VLA 前面的通用表示层。它甚至不需要训练复杂的神经网络 tokenizer：DCT 是解析变换，真正需要学习的主要就是 BPE 词表，整个系统只有很少的超参数。论文还比较了 FSQ 这种 learned vector-quantization 方法，发现 FAST 在高频灵巧任务上通常一样好或更好，同时实现更简单。([arXiv][1]) 这也是这篇论文很漂亮的一点：**它解决问题的方法并不是增加一个更复杂的模型，而是找到更正确的表示。**

DROID 实验尤其值得注意。DROID 是一个比较大、比较“野外”的多任务机器人数据集，环境、物体和行为都很杂。此前 autoregressive VLA 很难直接在 DROID 上训练出一个可以真正 zero-shot 泛化的通用策略；FAST 则让模型首次能够在完整 DROID 上有效训练，并直接拿到完全没见过的新环境里，只用自然语言 prompt 就执行 pick-and-place、擦桌子、开关柜门、水龙头等操作。作者在三个大学校园的不同环境里做了测试。([arXiv][1]) 这和你前面读 GPT-2/GPT-3 时的故事非常像：一旦表示和训练接口足够统一，模型就开始有机会从大而杂的数据中学出 generalist behavior，然后仅靠 prompt 指定具体任务。

## 09 · 与 flow matching 的正面对比

再来看 FAST 和 π0 原来的 flow matching 正面对比。π0 原版是 diffusion/flow 风格：从噪声开始，多步去噪得到连续 action chunk；π0-FAST 则完全可以像 GPT 那样 autoregressively 生成 action tokens，再把 token 解码回连续轨迹。单任务、小数据集上，两者表现基本相当；在更大的 Table Bussing 数据上，FAST 版本收敛明显更快，大约用 **1/3 的训练 steps** 就能到很高性能，而且在 DROID 上作者观察到 autoregressive FAST 版本对语言命令的遵循甚至更好。([arXiv][1]) 作者没有声称这证明 autoregressive 一定优于 diffusion，但它确实说明：**如果 tokenization 做对，next-token prediction 在机器人里也可能成为非常强的统一训练目标。**

## 10 · 大规模训练的效率收益

然后他们把 π0-FAST 放到真正的大规模数据上：和 π0 使用同一套跨机器人混合数据，里面包括约 **9.03 亿个自有 robot timesteps**，以及 DROID、BridgeV2、OXE 等开放数据，总体就是你上一章看到的约 **1 万小时机器人数据**。结果 π0-FAST 在 table bussing、T-shirt folding、grocery bagging、toast、laundry folding 这些任务上总体能够匹配 diffusion π0，而训练 GPU 时间大约减少 **5 倍**。([arXiv][1]) 对大规模 foundation model 来说，这不是小优化：如果一次训练原本需要几千 GPU 小时，5 倍意味着可以做更多实验、更快迭代，也更容易扩大数据和模型规模。

## 11 · 训练更快，推理仍有代价

但 FAST 也不是全赢。它最大的明显缺点恰恰来自 autoregressive decoding：**推理比较慢。**论文报告，在 NVIDIA 4090 上，diffusion π0 生成一秒 action chunk 大约需要 **100ms**，而 π0-FAST 大约需要 **750ms**。原因是 FAST 通常仍然要顺序生成 30–60 个 action tokens，而且需要动用完整约 2B 参数的语言模型 backbone；原版 π0 的 diffusion action decoding 则主要用约 300M 参数的 action expert、做大约 10 个去噪步骤。([arXiv][1]) 所以 FAST 出现了一个很典型的 trade-off：**训练快很多，但当前推理反而慢。**在叠衣服这种相对静态任务里问题不大，但如果机器人要接飞过来的球，750ms 就可能太慢。作者也因此把 speculative decoding、quantization、专用推理 kernel 等 LLM 世界成熟的加速技术列为下一步方向。([arXiv][1])

## 12 · 接入自回归模型的工具箱

这其实让 FAST 特别有意思，因为它把机器人技术重新接回了大语言模型的整套生态。π0 原版为了连续动作，引入一个专门的 flow-matching action expert；FAST 则说，如果能把连续动作压成高质量 token，我们就可以重新使用**整个 autoregressive Transformer 工具箱**：标准 cross-entropy / next-token prediction、成熟的 token vocabulary、scaling infrastructure，以及未来 speculative decoding、KV cache、量化、推理 kernel 等等。论文甚至强调 FAST 不需要修改底层预训练 Transformer。([arXiv][1]) 所以它真正想争取的并不仅仅是“折衣服得分高一点”，而是让机器人动作尽可能变成和文字一样，可以直接进入通用 sequence-modeling 范式。

## 13 · 连续轨迹与离散动作语言

如果把 π0 和 FAST 放在一起读，两篇论文其实代表两种很漂亮的路线。**π0 的观点是：机器人动作本来就是连续信号，不要勉强把它离散化，直接用 flow matching 生成连续轨迹。FAST 的观点则是：动作可以离散化，只是不能笨拙地逐时间点量化；先对时间序列压缩，找到更合适的 action tokens以后，autoregressive modeling 完全可以工作。**目前论文自己也很谨慎，说 diffusion 和 autoregressive VLA 到底哪个最终更好仍然没有定论，两边在训练速度、推理速度、语言 grounding 和表达能力上各有 trade-off。([arXiv][1]) 这点非常值得记住，因为真正优秀的论文往往不是宣布“旧路线死了”，而是把一个以前以为是架构问题的东西重新定位成了表示问题。

现在把它放回你前面一直在读的历史主线里，就特别有意思。GPT 系列的一条核心经验是：**tokenization + next-token prediction + scaling** 可以成为非常强的统一学习接口。但进入机器人以后，大家发现物理动作不像文字，自然语言 token 那么容易定义，于是 π0 转向 flow matching。FAST 则又把问题推进一步：**也许 physical intelligence 的瓶颈之一，不只是模型本身，而是我们还没有找到足够好的“动作语言”。**一旦连续运动可以压成信息密度高、跨机器人通用的 token，那么机器人也可以更彻底地继承语言模型几十年来积累下来的 sequence modeling 方法。

## 14 · 用一分钟讲给朋友听

如果现在合上论文，用费曼法一分钟讲给朋友听，我会这样说：**以前想让机器人像 GPT 一样生成动作，通常会把每个时刻每个关节位置单独离散成 token。但机器人动作很平滑，高频控制时相邻 token 几乎一样，所以模型只要学会复制上一帧就能得到很低 loss，反而学不会真正的动作轨迹。FAST 的解决办法是先把一秒钟动作做 DCT，把轨迹从时间域变成频率域，让少量低频系数描述动作整体形状，再量化这些系数，并用 BPE 把大量重复模式压缩掉。结果像双臂 50Hz 折衣服这种任务，一秒动作可以从大约 700 个 token 压到 53 个左右。这样普通 autoregressive Transformer 就能学会过去很难学的高频灵巧控制。作者进一步用 100 万段真实机器人动作训练 FAST+ 通用 tokenizer，并把它和 π0 组合，在约 1 万小时机器人数据上训练，最终性能接近原本的 diffusion π0，但训练算力最多减少约 5 倍。**([arXiv][1])

所以这篇论文我建议最后只记一句话：**Compression makes actions predictable。**📖 或者用更完整的话说：**FAST 不是让机器人“更会预测 token”，而是先重新定义什么才应该算一个有意义的 action token。**语言模型成功的重要前提之一，是我们没有让模型逐字节机械地预测所有冗余，而是通过 tokenizer 把文字压成合适的符号；FAST 提出的观点就是，机器人也需要自己的 tokenizer。这样把 π0 和 FAST 连起来，你就能看到机器人基础模型现在正在探索的一个核心分叉：**未来的动作生成，究竟更像 diffusion——直接生成连续世界，还是更像 GPT——先发明一种好的“动作语言”，再预测下一个 token。**

[1]: https://arxiv.org/html/2501.09747v1 "FAST: Efficient Action Tokenization for Vision-Language-Action Models"
