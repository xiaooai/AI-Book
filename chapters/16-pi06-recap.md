# π*0.6：让机器人从自己的经验中学习

论文：π*0.6: a VLA That Learns From Experience（2025）

## 01 · 部署以后，还能继续进步吗？

这篇是 Physical Intelligence 的《π*0.6: a VLA That Learns From Experience》。如果把你前面一路读过的 π0 → FAST → Hi Robot → π0.5 → Knowledge Insulation → RTC 接起来，这一篇其实跨过了一个很重要的门槛：前面的工作大多还在回答“怎样让机器人更会学、更会泛化、更会实时执行”，而 π*0.6 开始认真回答另一个更像“真正智能体”的问题：**机器人部署以后，能不能从自己实际犯的错误里继续进步？**也就是说，机器人不只是把人类 demonstrations 学一遍然后冻结，而是上线工作、自己尝试、失败、被人纠正，再把这些真实经验重新吃回训练流程，让下一版更快、更稳、更少犯错。作者把这套方法叫 **RECAP：RL with Experience and Corrections via Advantage-conditioned Policies**。最终他们展示的 π*0.6 能在真实环境里折复杂衣物、组装纸箱、用专业咖啡机做 espresso，而且在一些最难任务上，加入真实部署经验以后，单位时间完成任务的 throughput 提高超过 2 倍，失败率大约减半。

## 02 · 从模仿示范到亲自练习

先用费曼法理解作者为什么觉得 imitation learning 还不够。假设你请一个师傅演示 1000 次怎么折衣服，机器人把这些演示学得很好。问题是，真正上线以后机器人会犯一些师傅演示里从来不会犯的错误：衣袖卡住、衬衫翻面、纸箱两个纸板粘在一起、咖啡粉没压紧、portafilter 插歪。**这些错误恰恰是机器人自己的分布，而不是人类专家的分布。**如果训练数据全是“人做对时是什么样”，机器人就很少看到“我已经把事情搞砸一点了，现在怎么救回来”。这就是 imitation learning 著名的 distribution shift / compounding error：一次小错误把机器人带到陌生状态，接下来每一步都会越来越不像训练数据。作者因此强调，真正达到熟练水平不能只靠看别人演示，还得靠“练习”。自主 rollout 提供的就是这种练习数据——机器人真正会走进哪些坑、在哪些地方慢、什么情况下失败。

这和你刚刚读 DeepSeek-R1 时其实有一个很漂亮的呼应。R1 的思路是：大模型光模仿人写好的推理链还不够，最好还能通过 reward 让它自己尝试，然后强化那些最终成功的思路。π*0.6 把这个思想真正搬到了物理世界：**不要只模仿人类动作，也让机器人产生自己的 trajectory，再根据结果好坏去学习。**区别是，语言推理里一条数学答案对不对通常很好自动判断；现实机器人就麻烦多了。折衣服“成功”可能涉及折得对不对、方向对不对、有没有在规定时间完成；espresso 还涉及液体、器具和多阶段流程。因此这篇论文最大的难点不是“强化学习三个字”，而是怎么把 RL 做成一个真的能在复杂真实机器人上规模化工作的训练 pipeline。

## 03 · RECAP：做、评估、学习、再做

整套方法叫 RECAP，它有三个循环步骤。第一步，**让机器人真的去做任务**，收集 autonomous rollouts，并给每个 episode 一个最终 outcome/reward；必要的时候，人类 teleoperator 可以在线接管几步，纠正机器人快要犯的大错。第二步，用目前所有数据训练一个 **value function**，让它学会看当前画面和语言任务以后判断：“从现在这个状态继续下去，最终成功的希望多大、离任务完成大概还有多远？”第三步，用这个 value function 判断每个动作是“比当前策略正常水平更好”还是“更差”，然后让 VLA 更多模仿高 advantage 的动作。做完以后再把新 policy 放回机器人，继续收数据，再重复一轮。 所以整个循环非常像人练技能：**先做 → 看结果 → 分析哪些动作帮了忙 → 下次更多采用好动作 → 再做。**

## 04 · Value 给状态打分，Advantage 给选择打分

这里必须把 **value 和 advantage** 讲清楚，因为它是整篇论文的灵魂。Value 可以理解成“站在当前状态，我预计最后能有多好”。比如机器人正在做 espresso，现在已经把咖啡粉装进 portafilter，而且姿势很正，也许 value 很高；如果已经把粉撒了一桌，value 很低。Advantage 问的问题则更加具体：**“我刚才这个动作，相比在这个状态下一般会做的动作，是让局面变好了还是变坏了？”**假设机器人本来正在顺利折衬衫，这时某个动作把衣服边缘重新扯乱了，那这个动作 advantage 就是负的；如果它成功把折歪的袖子调整回来，这个动作就是正 advantage。数学上论文使用 value 估计动作前后的进展，再得到 advantage。 用最简单的话说：**value 给状态打分，advantage 给选择打分。**

## 05 · 用优势条件学习更好的动作

真正很巧妙的一步叫 **advantage conditioning**。传统 RL 很容易让人想到 PPO：根据 reward 算梯度，然后直接调整 policy，让好动作概率上升、坏动作下降。但对 π0.6 这种巨大的 flow-matching VLA 来说，这种 policy-gradient 方法实现复杂、训练也容易不稳定。作者换了一个很像“条件生成”的办法：他们直接把一个额外的文字条件塞给模型——**“Advantage: positive” 或 “Advantage: negative”。**训练的时候，模型同时看到状态、任务、动作，以及这个动作属于 positive 还是 negative advantage；到了真正运行的时候，只要求它生成 **positive advantage 条件下的动作**。 这件事其实特别漂亮，因为它把一个强化学习问题部分转成了 VLA 本来就很擅长的 conditional modeling 问题。

可以把它类比成训练一个厨师模型。你不必复杂地改变每一道动作的 loss 权重，而是给训练视频打标签：“这是更好的做法”“这是更差的做法”。模型最后学到两个分布：如果我按照较差方式做，动作通常是什么样；如果我要求“做得比现在更好”，动作又是什么样。真正部署时，你永远给它“positive”这个条件，于是等于从已有经验中抽取出一个更优策略。论文背后有理论保证：在某些条件下，根据 advantage 的单调函数去条件化 policy，可以得到比 reference policy 更好的策略。 费曼式地说就是：**不是把失败数据扔掉，而是告诉模型“这是失败方向”，然后部署时明确要求它走成功方向。**

## 06 · 失败数据与人类纠正的价值

这一点解决了机器人 RL 中另一个很现实的问题：**坏数据其实也有用。**普通 SFT 往往只敢训练“正确 demonstrations”，因为如果把机器人失败 rollout 原样拿进去做行为克隆，相当于教机器人模仿自己的错误。RECAP 不一样，因为它知道动作的 advantage。失败 episode 里也可能有前半段做得很好，成功 episode 里也可能夹着低质量动作；value function 可以提供更细粒度判断。于是数据不再简单分成“整条好/整条坏”，而是可以说“这个状态下，这一步值得学，那一步不值得学”。这让自主 rollout 真正变成有价值的训练资源，而不是大量需要丢弃的失败录像。

人类 intervention 在里面扮演的角色也很讲究。机器人自主运行时，如果快要发生明显灾难，比如马上把东西掉地上或者彻底卡死，人类 teleoperator 可以接管，执行几步正确动作，再把控制权还给机器人。论文把这些 intervention 默认当作 positive advantage 的样本。 但作者特别强调：**intervention 不是全部答案。**人每次接管会打断轨迹，而且人类自己也不保证每次都特别快、特别优雅。它主要解决的是“探索不够”——如果机器人从来没见过怎样从某种失败状态恢复，人直接示范一次很有帮助；但要把整体速度从 10 分钟提高到 5 分钟，人类 intervention 不一定告诉你该怎么优化。真正的 subtle improvement 仍然来自 reward + value + 自主经验。

## 07 · 从 π0.6 到 π*0.6

再说模型本身。论文里的 π0.6 是 π0.5 的继任者，底层仍然保留你前面已经熟悉的很多东西：它可以用 flow matching 生成 action chunks，也可以输出高层 subtask 文本；训练采用 Knowledge Insulation，同时预测连续动作和 FAST 离散动作，并阻断 flow action expert 的梯度对 backbone 的破坏；预训练同时混 robot data 和 web vision-language data。π0.6 相比 π0.5 又做了几个升级：**加入更多机器人平台的数据，base VLM 换成 Gemma 3 4B，action expert 扩大到 860M 参数。**动作仍然以 50Hz 输出。 所以你可以把 π0.6 理解成：“π0.5/KI 那套已经比较成熟的 generalist VLA 又扩大升级了一版。”

而 **π*0.6** 里的那个星号，真正代表的不是单纯“π0.6 Plus”，而是**π0.6 加上 RL 能力**。论文给它增加了 advantage condition，因此它可以在相同 observation 下根据“positive / negative advantage”生成不同动作分布。然后整个 RECAP 会先在大规模多任务、多机器人 demonstration 数据上做一次 **offline RL pretraining**，这批预训练数据达到数万小时量级；之后再把模型用 demonstrations 适配到具体 downstream task，最后开始 on-robot collection，自主运行、收 reward、训练 value、重新训练 policy。 所以和 π0.5 最大的路线区别是：**以前 pretraining 主要问“人类做过什么”；现在连 pretraining 都开始问“哪些行为更好”。**

## 08 · 从模仿行为到提高生产力

这其实是很重要的范式变化。传统机器人 foundation model 的预训练数据一般都是专家 demonstrations，于是一个隐含上限是：**模型最多把 demonstration distribution 模仿得特别好。**如果示范者平均折一件衣服 100 秒，单纯 imitation 很难自然产生一个 50 秒的策略，因为训练目标一直在奖励“像示范者”。RL 则允许模型利用 reward 去超过数据里的平均行为：只要机器人自己发现某种更快又成功的动作，value function 就会认为它更好，下轮训练就能放大它。因此作者强调他们的目标不只是提高 success rate，也要提高 **throughput——每小时真正成功完成多少次任务**。 这是从“机器人会不会做”向“机器人能不能像工人一样稳定高效地干活”的转变。

## 09 · 衣物、纸箱与咖啡任务

论文选的三个任务都很能体现这一点。第一类是洗衣服，最难版本覆盖 **11 种衣物**，包括毛巾、衬衫、毛衣、牛仔裤、T 恤、短裤、polo、裙子、长袖、袜子和内衣，需要处理真正柔软、形状变化巨大的物体。第二类是组装纸箱，涉及压平纸板、折叠、固定并最终堆放，是明显的长程任务。第三类是做 espresso：机器人要拿 portafilter、放到磨豆机、磨豆、压粉、把 portafilter 锁进专业咖啡机、拿杯子、萃取、最后端出来。 这些任务每次可能持续 **5–15 分钟**，比“拿方块放到碗里”那类短 benchmark 更接近真实劳动。

## 10 · 真实经验带来的效率提升

实验里最值得记住的结果是：当他们从“offline RL + SFT”继续加入真实 robot deployment 数据、完整跑 RECAP 后，**复杂衣物折叠和 espresso 两个任务的 throughput 都超过翻倍，同时失败率大约降低一半。** 这里要注意 throughput 比 success rate 更严格。一个机器人成功率从 90% 升到 95% 很好，但如果每次速度慢了一倍，现实生产力反而下降；throughput 同时奖励“做对”和“做快”。在比较容易的 T-shirt/shorts folding 上，SFT 后成功率已经很接近上限，但 RECAP 仍然能让 throughput 显著提高。 这非常清楚地说明 RL 学到的不只是“避免失败”，还有**动作效率和流畅度**。

他们还真的做了多轮 self-improvement。比如折 T-shirt/shorts 和箱子组装，都重复执行“部署 → 收数据 → 更新 value → 更新 policy”。洗衣任务经过迭代以后 throughput 总体又提高约 **50%**；纸箱组装经过第二轮以后，throughput 达到大约 **2 倍**，最终在规定时间里完成折箱和贴标的成功率约 90%。 这就是标题里 “Learns From Experience” 最直接的含义：**不是训练完就结束，而是工作一阵以后真的比原来的自己更会工作。**

## 11 · 用少量轨迹纠正具体坏习惯

论文还有一个我很喜欢的小实验，因为它特别像“纠正坏习惯”。他们故意构造一种很具体的衣服摆放方式，使 baseline 经常把 T-shirt 折成**领口朝下**，而评判要求领口必须朝上。这个失败模式不是一个广义的“模型不够聪明”，而是非常具体的坏习惯。于是作者跑两轮 RECAP，每轮只收大约 **600 条轨迹**，最终 policy 在这个 adversarial setup 下成功率达到 **97%**，而且速度也很快。 这说明 RL 不只是整体平均分加一点，也可以在部署中发现“机器人总在这里犯同一种错”，然后用针对性的经验把这个 failure mode 洗掉。

## 12 · 为什么选择迭代式离线强化学习

为什么不用你在 DeepSeek-R1 里见过的 PPO 或类似 policy-gradient 方法？论文专门做了比较。同样给 AWR、PPO 和 RECAP 使用相同 on-robot 数据，AWR 和 PPO 都明显不如 advantage-conditioned extraction；PPO 为了在 off-policy 设置下稳定甚至需要非常小的 trust-region，结果虽然不炸，却很难真正取得好提升；AWR 能有不错成功率，但动作更慢，所以 throughput 较差。 这其实是一个很现实的信号：**语言模型 RL 上流行的方法不能直接照搬到大型 flow-based VLA。**机器人数据昂贵、明显 off-policy、动作是连续高维 distribution，而且不能不停在真实机器上重新采 fresh on-policy batch，因此一个能反复利用所有历史数据的 offline-RL 式方法更合适。

这也正是 RECAP 很有价值的一点：**demonstrations、旧 policy 的 rollouts、新 policy 的 rollouts、人类 corrections，全都可以留在数据池里继续用。**它不用像严格 on-policy PPO 那样说“这批数据是旧模型产生的，过期了，不能用了”。真实机器人数据每一分钟都很贵，所以“数据能不能重复吃”是极其现实的工程问题。作者把它设计成 iterated offline RL：收一批机器人数据，停下来重新训练，再部署下一版；不是 policy 在机器人运行的同时每秒更新。 这虽然没有“实时在线学习”那么科幻，却稳定、可控，而且对真实硬件安全得多。

## 13 · 长期运行与可靠性

最终他们还用非常长时间运行来证明 reliability 已经开始接近“实用”。论文报告，最终 π*0.6 可以**连续 13 小时做 espresso drinks**，在一个新家庭环境里**连续超过 2 小时折没见过的衣物而无需中断**，还可以在工厂里组装真正用于包装的纸箱。 这类结果和 benchmark 成功率的感觉很不一样。现实机器人真正难的不是做对一次，而是第 37 次、第 91 次、第 200 次碰到稍微不同的东西时还能继续工作。π*0.6 这篇论文其实是在把研究指标从“单 episode 会不会成功”慢慢推向“**长期运行时到底有多大生产力和可靠性**”。

## 14 · 尚未实现完全自主学习

当然，它还远远不是“机器人自己无限进化”。首先 reward 和 success labels 目前仍然有不少人类参与；人需要给结果打标签，还要提供 intervention，有时候还要帮忙 reset 现场。作者明确承认系统并非 fully autonomous。 第二，它的 exploration 还非常朴素：基本依赖 policy 本身的随机性加人类 intervention，并没有一个特别强的主动探索机制。第三，RECAP 目前是“收一批数据 → 离线更新 → 再部署”的循环，而不是机器人边工作边实时改变自己。作者也把自动 reward、自动 reset、更聪明的 exploration 和真正 concurrent online RL 列为未来方向。 所以它证明的是“部署经验可以稳定改善大型 VLA”，而不是已经实现一个完全自主的终身学习机器人。

## 15 · 从基础模型到持续改进的智能体

如果现在把 PI 这一系列论文连起来，整条线就非常清楚了。**π0：先让一个 generalist VLA 会做连续灵巧动作；FAST：让动作可以高效 token 化；Hi Robot：让高层语义规划和低层操作分层；π0.5：让网页知识、不同机器人、语言和动作共同训练，获得 open-world generalization；Knowledge Insulation：在学动作时保护原 VLM 的知识；RTC：让一个推理较慢的大 VLA 也能一边思考一边实时行动；π*0.6：再进一步，让机器人在真正部署以后通过 reward、失败和人类纠正继续学习。**这已经形成了一个很完整的 foundation-agent 路线：**先预训练获得通用能力，再泛化到新世界，再实时执行，最后从真实经验中自我改进。**

如果再把它和你之前读的 GPT / DeepSeek-R1 历史放在一起，就更有意思。语言模型先经历了一个路线：**pretraining → instruction tuning → RLHF → reasoning RL。**机器人现在似乎在走一个有点类似但更难的路线：**robot pretraining → VLA post-training → open-world deployment → real-world RL。**区别在于，语言模型生成一个错误答案的成本很低，机器人做错一步可能真的把杯子打碎，所以机器人 RL 必须同时解决安全、数据效率、off-policy learning 和人工干预这些问题。π*0.6 的意义就在于，它开始展示“大型 generalist VLA + 真机强化学习”不仅理论上可行，而且已经可以在长程、柔性物体、液体、工业纸箱这些不太玩具化的任务上得到显著收益。

## 16 · 用一分钟讲给朋友听

如果现在合上论文，用费曼法一分钟讲给朋友听，我会这样说：**以前的 π0.5 像一个受过大量训练、毕业以后去工作的机器人，但它上班后基本不会因为自己的工作经历继续变强。π*0.6 给它加入了强化学习循环：机器人先真的去折衣服、做咖啡、组纸箱，成功或失败都会记录下来；人类必要时可以接管纠正。系统再训练一个 value function，判断每个状态离成功有多远，并给动作算 advantage，判断“这一步比正常表现更好还是更差”。然后 VLA 学习在“Advantage: positive”条件下应该怎么动作，下一版机器人就更倾向采用那些真正带来成功、速度和恢复能力的行为。经过反复部署和训练，复杂洗衣和 espresso 的每小时成功完成量能超过翻倍，失败率约减半，而且一些具体坏习惯可以用几百条真实 rollout 明显纠正。**

所以这篇我建议最后只记一句话：**π0.5 learns from other people’s experience；π*0.6 starts learning from its own experience。** 📖 更技术一点就是：**Pretraining tells the robot what good behavior looks like；RECAP lets deployment tell it which of its own actions actually make things better。**如果说 π0.5 的关键是“到了没见过的世界还能做事”，那么 π*0.6 往前迈的这一步就是：**到了那个世界以后，不要永远停留在第一天的水平——让每一次真实工作，都有机会变成下一版模型的训练数据。**
