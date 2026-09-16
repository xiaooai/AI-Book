# EgoMimic：让人类第一视角动作直接训练机器人

论文：EgoMimic: Scaling Imitation Learning via Egocentric Video（2024）

## 01 · 把人当作另一种机器人身体

这篇是 2024 年的《EgoMimic: Scaling Imitation Learning via Egocentric Video》。如果用费曼法先把整篇论文压成一句话，它真正想解决的是一个机器人领域非常现实的瓶颈：**机器人学习需要大量动作示范，但让人遥控机器人收数据太慢了；人类自己每天却已经在做无数抓、放、折衣服、装东西的动作。那么能不能不要把“人类视频”只当成辅助视觉资料，而是直接把“人的手”当作另一种机器人身体，让人类第一视角视频像机器人遥操作数据一样，直接训练低层动作 policy？** EgoMimic 的答案是可以，但前提是你得认真解决人手和机械臂之间的坐标系、动作分布、外观、运动速度这些差异。它最终用 Project Aria 眼镜采集人类第一视角视频和 3D 手轨迹，再和真实双臂机器人遥操作数据一起训练同一个 policy；在三个真实长程 manipulation task 上，加入人类数据后整体表现明显提高，而且还能把**只在人类数据里出现过的新颜色甚至新场景**迁移给机器人。([arXiv][1])

## 02 · 数据瓶颈：人类示范与机器人遥操作

先想象传统机器人 imitation learning 是怎么收数据的。你要教机器人折 T 恤，就得坐在机器人旁边，用遥操作设备一遍又一遍控制它：右手抓衣服、左手拉袖子、失败了重新摆好衣服、再来一次。这件事的问题不只是贵，而是**机器人本身就是数据采集瓶颈**：一次只有一台机器在工作，动作又比人慢，失败还得 reset。论文举出的 Object-in-Bowl 数据特别直观：一个小时，人可以直接用手完成大约 **1400 次示范**，同样一个小时机器人遥操作只能得到大约 **135 次**。也就是说，如果一份人类演示能够和机器人演示产生差不多的训练价值，那么数据采集效率可能立刻提高一个数量级。([arXiv][2]) 这就是 EgoMimic 的根本赌注：**robotics 的“互联网数据”也许不是等待全世界装满机器人，而是利用人本来就在做的事情。**

## 03 · 先把动作放进同一个坐标系

但这件事听起来容易，真正做的时候马上遇到第一个问题：**人的手和机器人的手根本不是同一种 action。**人戴着 Aria 眼镜走动时，摄像头本身也在移动；机器人相机通常固定在机身上。假设我现在看到自己的右手向前移动 10 厘米，这个“向前”到底是相对世界、相对桌子，还是相对我此刻转动中的头？如果不统一坐标系，同样一个动作在人类数据和机器人数据里可能完全不一样。EgoMimic 的解决办法是利用 Aria 的 SLAM，得到眼镜每一时刻在世界中的 pose，然后把未来整段手部轨迹重新转换到**当前这一帧相机坐标系**里。机器人 end-effector trajectory 也做类似转换。这样训练模型时，不管数据来自手还是机械臂，它面对的共同问题都变成：“以我现在看到世界的这个视角，接下来末端应该往哪里移动？”([arXiv][2])

## 04 · 从相机与机器人形态减少差异

这里 Project Aria 其实非常关键。它不像传统 motion capture，需要在房间里架一圈相机；也不像手持夹具，需要人专门拿一个奇怪设备演示。Aria 本身就是一副大约 75 克的眼镜，前面有宽视野 RGB camera，旁边的 SLAM cameras 可以估计头部轨迹和双手 3D pose。更巧的是，作者**给机器人也装了一副同样的 Aria 眼镜**，而且位置尽量接近人的眼睛高度。为什么这么做？因为 cross-embodiment learning 最大的麻烦之一不是动作，而是视觉：人的 GoPro 看出去和机器人的 RealSense 看出去，FOV、曝光、颜色、位置都不同。既然最终目标是让两种数据共用一个 visual encoder，那最直接的办法就是从硬件层先让两个“眼睛”尽量像。([arXiv][2]) 这篇论文有一个很务实的特点：**不要只期待神经网络自己解决 domain gap，能在传感器和机械结构层消掉的差异先消掉。**

他们甚至专门造了一台比较“像人”的低成本双臂机器人。两只 ViperX 300S 机械臂倒挂在一个类似躯干的支架上，尺寸和运动范围比很多桌面 Franka 更接近人的手臂，同时两只手腕还各有摄像头。作者认为，如果机器人手臂很粗、很重、速度很慢，而人的手臂细、轻、动作快，模型会更加容易把两种数据看成两个完全不同的 domain。所以 EgoMimic 不是单纯一个算法 paper，而是从**数据采集硬件 → 机器人形态 → 数据处理 → policy architecture**整条栈一起设计。论文自己也把它称为 full-stack framework。([arXiv][2])

## 05 · 动作归一化：对齐统计尺度

即使坐标系和相机都尽量对齐，第二个麻烦还是存在：**人和机器人做同一件事情，动作分布还是不一样。**例如人的左右手能伸得更远、速度也更快；机器人关节限制不同，teleoperator 的动作又更保守。作者观察到尤其在左右方向上，人手 pose distribution 和 robot end-effector distribution 有明显差异。如果直接把两类数据一起训练，Transformer 很容易学出一个捷径：“哦，这种动作范围是人，那种是机器人”，然后内部依然形成两个分开的表示，所谓 co-training 只是把两个 dataset 放在同一个 batch 里而已。于是作者对 human pose/action 和 robot pose/action **分别做 Gaussian / Z-score normalization**，先减均值、除标准差，把两类动作在统计尺度上拉得更接近。去掉 action normalization 后，Object-in-Bowl 的任务得分从 128 掉到 79，说明这个看起来非常朴素的步骤其实很重要。([arXiv][2])

## 06 · 遮住外观，让模型关注动作

第三个 gap 更直观：**人的手长得和机械臂完全不同。**模型如果看到“皮肤颜色的手”就进入 human mode，看到“黑色金属机械臂”就进入 robot mode，依然不会真正学到共同动作结构。EgoMimic 的做法简单得有点粗暴：用 SAM 把人的手臂和机器人的机械臂都遮成黑色，再在上面画一条红色线表示 end-effector 的方向。这样 visual encoder 更难利用“这是肉手还是机械手”这个外观捷径，被迫把注意力更多放到**物体、空间关系和动作方向**上。消融实验也支持这个设计：去掉红线得分从 128 降到 112，再连 mask 一起去掉降到 95。([arXiv][2]) 用费曼法讲，就是考试时把两位学生的校服遮住，逼老师根据“他们在做什么”而不是“他们是谁”来判断。

## 07 · 时间对齐：不同速度，相同动作过程

还有一个很有意思的问题：**人动作比机器人快很多。**如果人用 1 秒抓起玩具，机器人可能要 4 秒；你直接让同一个模型学习同样长度的 action chunk，它看到的 temporal structure 完全不同。作者因此把人类动作“慢下来”：人类数据的 1 秒轨迹和机器人数据的 4 秒轨迹都采样成 **100 个未来动作点**。人类原始视频大约 30Hz，机器人约 50Hz，但 action chunk 都统一成 100 个位置。([arXiv][2]) 这再次说明 cross-embodiment 不只是“坐标对应”，还包括**时间尺度对应**。一个真正的动作表示最好描述“任务过程”，而不是死绑某具身体的执行速度。

## 08 · 共享 ACT 与两种动作输出

现在才来到模型本身，而它反而没有特别复杂。作者基于 **ACT**，也就是 Action Chunking with Transformers。核心是一个共享视觉编码器和 Transformer，然后接两个非常浅的输出头。一个输出 **pose-space action**，也就是未来 hand/end-effector trajectory；另一个输出机器人真正执行的 **joint-space action**。人类数据只有 pose supervision，因为 Aria 只知道手怎么动，不知道 ViperX 的关节应该转多少；机器人数据则同时有 end-effector pose 和 joint action。所以训练时，人类和机器人共同监督 pose head，只有机器人监督 joint head，但前面绝大多数 Transformer 参数全部共享。([arXiv][2])

这一招是整篇最值得真正理解的地方：**人类数据虽然不能直接告诉机器人“第 3 个关节转几度”，却可以逼共享网络学会一个更好的动作表示，而机器人少量 joint supervision 再负责把这个表示翻译成真正电机动作。**你可以想象一个人类老师和机器人老师共同教一个学生。人类老师无法说“ViperX elbow joint = 0.42 rad”，但能提供大量“看到这种局面以后，手应该往这里走”的经验；机器人老师数据少，却能告诉模型“这种末端运动具体对应哪些 joint commands”。共享 backbone 就在两种老师之间搭桥。

## 09 · 让人类数据进入低层控制表示

这也是 EgoMimic 和当时很多 Human-to-Robot 方法最关键的区别。过去一个常见路线是：**人类数据只训练高层 planner，机器人数据训练低层 controller。**例如人类视频告诉高层“下一步手应该去袋子这里”，低层机器人 policy 再根据这个目标真正控制关节。问题是，无论人类视频多到一百万小时，真正低层 policy 还是只吃那几百小时机器人数据，所以最终性能天花板可能依旧由 robot dataset 决定。EgoMimic 反过来说：**不要把 human data 隔离在高层，让它直接参与训练低层 shared representation。**论文和 MimicPlay 的对比就是为了证明这一点。([arXiv][2])

## 10 · 三个真实长程操作任务

实验选了三个很能体现动作难度的真实任务。第一个 Continuous Object-in-Bowl：机器人在 40 秒里反复把约 6cm 的小玩具抓进碗里，再拿起碗把玩具倒出来继续做，物体和碗的位置都随机变化，主要考精确抓取和持续执行。第二个 Laundry：双手把随机位置、随机旋转的 T-shirt 折起来，先右袖、再左袖、最后整体对折。第三个 Groceries：左手抓住柔软购物袋的提手把袋口撑开，同时右手连续把三包薯片放进去。([arXiv][2]) 这些任务挑得很聪明，因为它们不是“手移动到某个静态目标”这么简单，而会暴露低层动作精度、双手协调、柔性物体和长程误差累积。

## 11 · 任务得分与成功率的提升

结果非常直观。Object-in-Bowl 上，ACT 得 39 分，MimicPlay 71，只有机器人数据的 EgoMimic 是 68，而完整 EgoMimic 达到 **128**；Laundry 的完整成功率从 ACT 的 **55% 提到 88%**；Groceries 则从 ACT 的 **22% 提到 30%**，其中最难的抓住袋子提手、把袋子打开这一阶段，从 54% 提到 **70%**。作者总结，在三个任务中，相对任务得分提高约 34%–228%，绝对成功率提升约 8–33 个百分点。更重要的是，和“同样架构但完全不加人类数据”的 EgoMimic 比，加入 hand data 仍带来 10%–88% 的得分提升，说明效果不是单纯来自改了 ACT 架构。([arXiv][2])

## 12 · 新颜色与新场景能否迁移

我觉得这篇真正漂亮的实验不是 in-domain，而是**只在人类数据里出现过的信息能不能传给机器人。**作者让机器人折训练时没见过颜色的 T-shirt。只用 robot demonstrations 的 ACT 成功率掉到 **25%**，完整 EgoMimic 仍保持 **85%**。更极端的是 Object-in-Bowl：他们收了一些来自一个全新背景、全新光照场景的人类数据，但**没有在那个场景收任何机器人数据**。结果 EgoMimic 真机器人拿到新场景后可以达到 63 分，而使用同样 human information、但采用“human high-level planner + robot low-level policy”的 MimicPlay 只有 4 分。([arXiv][2]) 这说明某种程度上，human data 学到的新视觉分布真的进入了机器人控制网络，而不是停留在一个高层模块里。

## 13 · 同样一小时，怎样收数据更有效

然后是最有长期意义的 scaling 实验。作者固定已有 **2 小时 robot data**，再问：我现在多花 1 小时，应该继续遥控机器人，还是直接让人戴眼镜做？结果“2h robot + 1h human”的 EgoMimic 达到 **128 分**，而“3h robot”的 ACT 只有 **74 分**。原因当然不只是人类数据天生质量更高，还因为一小时 human data 能得到约 1400 次 Object-in-Bowl demonstration，而一小时 robot data 只有 135 次。([arXiv][2]) 这就是论文标题里 **Scaling Imitation Learning via Egocentric Video** 真正的含义：它不是单纯证明“人类视频有帮助”，而是问**每投入一小时数据采集预算，哪种 embodiment 产生的学习价值更高？**

## 14 · 连接后来的跨身体学习路线

这个结果如果放到你后来读过的 Human-to-Robot Transfer、GEN-1、S1 那些论文之前看，会特别有历史感。EgoMimic 还处在一个比较早的阶段，它没有 10 万小时数据，没有一个 5B VLA，也没有展示 one-shot physical prompting。它做的是一个更加基础的证明：**human embodiment data 不必只是高层语义，不必只是视觉预训练；只要把动作、视觉和坐标分布认真对齐，它可以直接参与 low-level imitation learning。**后来 PI 的 Human-to-Robot Transfer 会进一步说：foundation model 足够多样以后，human→robot transfer 甚至会“emerge”，不必如此依赖手工 alignment；GEN 系列会把人类 physical interaction 扩到几十万小时；Skild S1 更进一步把一段 human demonstration 直接当 inference-time prompt。但这些更激进的路线，其实都建立在同一个基本直觉上：**人类每天产生的身体经验，比机器人自己收的数据大得多，必须想办法让机器人吃进去。**

## 15 · 与 EgoExo-WM 对照：学行动与学动态

和你刚刚读的 EgoExo-WM 比，这两篇也刚好形成一个非常漂亮的对照。EgoMimic 问的是：**人的第一视角动作经验能不能直接训练 robot policy？**EgoExo-WM 问的是：**第三人称人类视频能不能先转成第一视角经验，再训练 world model？**一个把 human experience 用来学习“怎么行动”，另一个用来学习“行动以后世界怎样变化”。未来真正的 embodied foundation model 很可能两边都会做：看人类视频学 policy，同时也学 dynamics。

## 16 · 局限：任务、传感器与机器人形态

当然 EgoMimic 还有很明显的局限。首先它训练的是**每个任务一个 policy**，还不是一个统一 generalist；Object-in-Bowl、Laundry、Groceries 之间并不是一个模型靠语言 prompt 切换。第二，它的人类数据也不是随便 YouTube 抓来的，而是戴 Project Aria、带 SLAM 和 3D hand tracking 的高质量数据，所以离真正“被动收集互联网级 human data”还有距离。第三，机器人本身是专门设计得比较接近人的上半身，而且还和人用了相同相机，这帮助很大；换到形态差异巨大的机械臂，迁移未必这么容易。第四，人类数据没有 gripper open/close 信息，因此抓取闭合仍然只能靠机器人数据学习。论文自己也明确说，下一步才希望探索“完全只在人类数据里出现的新行为”能不能真正迁移，比如机器人没练过折裤子，只看人折裤子以后是否会。([arXiv][2])

## 17 · 用一分钟讲给朋友听

如果现在合上论文，用费曼法一分钟给朋友讲，我会这样说：**机器人 imitation learning 最大的问题是机器人数据太难收，而人做同样事情快得多。EgoMimic 让人戴一副 Project Aria 眼镜，一边正常用手做任务，一边记录第一视角 RGB、头部 SLAM 和双手 3D 轨迹；机器人也装同样眼镜，尽量让两边视觉一致。然后把人的手轨迹和机器人的 end-effector 轨迹都转换到相同的相机坐标系，分别做动作归一化，再把人手和机械臂从图像里遮掉，减少 domain gap。训练时同一个 ACT Transformer 同时吃 human 和 robot data：两边共同学习未来 hand/end-effector pose，只有机器人数据额外监督真正的 joint actions。这样，人类数据虽然不知道机器人关节怎么转，仍然可以直接改善控制网络的共享表示。最终它在放玩具、折衣服、装购物袋三个真实任务上明显超过只用机器人数据的 ACT，而且能把只在人类数据里见过的新衣服颜色和新场景迁移给机器人；最关键的是，同样多花一小时采数据，1 小时人类数据能收 1400 次示范，比一小时机器人遥操作的 135 次高一个数量级，也带来更大的性能收益。**([arXiv][2])

所以这篇我建议最后只记一句话：**不要把人类视频只当作“机器人看过的辅助教材”，而要把人本身当成另一种 embodiment，把人的动作经验直接放进机器人的 imitation-learning 数据池。** 📖 更技术一点就是：**Human hand data + domain alignment + shared pose representation + robot joint supervision → unified low-level policy。**如果后来 Human-to-Robot、GEN、S1 代表的是“人类身体经验开始规模化进入 robot foundation models”，那么 EgoMimic 就像这条路线里非常关键的一块早期基石：**它先证明了一件最基本的事——人的第一视角动作数据，真的可以直接让机器人的手变得更会动。**

[1]: https://arxiv.org/abs/2410.24221 "[2410.24221] EgoMimic: Scaling Imitation Learning via Egocentric Video"

[2]: https://arxiv.org/html/2410.24221v1 "EgoMimic: Scaling Imitation Learning via Egocentric Video"
