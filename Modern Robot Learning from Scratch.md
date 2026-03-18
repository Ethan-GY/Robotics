## 发展历程：
通过运动学方程函数映射输入和输出，轨迹-关节角度

RL：不在建模，而是从尝试奖励中学习轨迹，不过有缺陷

behavior cloning，也就是imitation learning，没有奖励，更像输入对应输出的监督学习

为什么具身智能不能仿照llm训练出智能基础模型？有open x- embodiment这个大数据集。VLA基于VLM加了动作输出层，可以完成叠衣服这种任务。 感知-理解-行动



Classic Robotics 

S101型号的机械臂

explicit modeling：根据关节角度输出位置

Dynamic-based Robotics 

robust差（一个组件坏了，整套失效），无法对齐模态，只有近似而不精确（忽略了摩擦力和形变等），没法从data出发



modern robotics

learning-based/implicit modeling

端到端接受观测并输出，可以对齐模态，不需要建模环境，只需要data



Robot imitation learning

强化学习在探索阶段非常费时，而且会做出危险的事情。奖励函数很难定义中间值

所以用imitation learning规避，offline，有人类专家设计轨迹（安全）

首先需要感知目标的相对位置

Input：o（观测的画面400*640*3+当前关节角度6），Output：a（下一步的关节角度6），建立function map observations to actions（动作概率分布）





## Deep Generative Modeling
训练预测概率分布接近真实概率分布



KL散度：![image](https://cdn.nlark.com/yuque/__latex/39da04cc843be83f0d7dffbbd98c95bc.svg)，px是权重，衡量x的重要性

衡量两个分布的相似度，越相似，KL越小。先用取值对应的p做比值，再套log（因为log可以以1为界区分正负，并且单调）

根据DGM生成目标对象，用于IL时就是依据概率选取动作



Energy-based model

![](https://cdn.nlark.com/yuque/0/2026/png/59026724/1773497485242-1f84e7b9-d4ad-4b82-ad98-8d8df7619233.png)

能量最低的地方，概率最大



还有Auto-regressive



重点是Variational Auto-Encoder（VAE ）

![](https://cdn.nlark.com/yuque/0/2026/png/59026724/1773542943868-17343e95-8847-4bc6-877a-49eb6f381ca8.png)

![](https://cdn.nlark.com/yuque/0/2026/png/59026724/1773545735550-82fc45db-febd-481a-b7a1-0940303b6f51.png)

![](https://cdn.nlark.com/yuque/0/2026/png/59026724/1773545841894-c9eb683f-b01b-48d2-bf27-3f1a0558f2cd.png)

要得到目标图像分布（概率分布），首先需要背后的隐藏因素z（latent variable），也就需要得到Latent Space（维度就是num of latent variables），各个样本在latent sapce组成的集群区域就是对应类别图像的概率分布区域，也就是为每张图像预测两个值（均值和方差）

在Robotics，input就是关节角度6+图像28*28

用encoder：![image](https://cdn.nlark.com/yuque/__latex/8eb01ee7248226db8d9526ce09279c16.svg)把图像根据latent variables转化进latent space，相当于用一个projector作映射

![](https://cdn.nlark.com/yuque/0/2026/png/59026724/1773545631558-9488d2ec-dd10-4b35-909f-20253e035e41.png)

![](https://cdn.nlark.com/yuque/0/2026/png/59026724/1773545655040-d2facbc3-1366-4e78-91fc-96e828fa38b4.png)

 再用decoder：![image](https://cdn.nlark.com/yuque/__latex/433a7f274d8312297f81af0588b59848.svg)从latent space中采样生成对象，并根据概率分布的均值和方差，对生成图像加入bias，得到不同的输出

![](https://cdn.nlark.com/yuque/0/2026/png/59026724/1773545803154-c82c782f-76f6-4ceb-a2f6-f9e0ba77290a.png)

 objective function: ![image](https://cdn.nlark.com/yuque/__latex/c4d27ff3cb5137c89d2cd4ae29fc718f.svg)

![](https://cdn.nlark.com/yuque/0/2025/png/59026724/1764385284538-b51ecd9f-fbb6-422c-83c4-5ab3d6102cec.png)

![](https://cdn.nlark.com/yuque/0/2025/png/59026724/1764384474664-6d412f93-be84-4bac-b17f-9a0931156c5b.png)

![](https://cdn.nlark.com/yuque/0/2025/png/59026724/1764384671733-1a158d76-e799-432e-9e9f-c34789f03caa.png)

rec：重建图片 接近 原始图片像素 （计算MSE）

KL：encoder输出的图片生成分布 接近 latent_space	里的标准多元正态分布 （数学推导）

![](https://cdn.nlark.com/yuque/0/2025/png/59026724/1764384616701-c2572204-a013-446a-97b9-28c393f518aa.png)

用log是trick，因为不影响单调性



## Transformer
Encoder：先将token embedding，然后拼接positional encoding。然后将向量转化成Queries，Keys and Values vectors，都是learnable的，训练出来Q的目标是找“能修饰自己”的K。用Q，K计算相关性，softmax得到attention scores，再乘以V得到context vectors，不断经过各层循环迭代更新矩阵参数。eg.input 是 the red car

![](https://cdn.nlark.com/yuque/0/2026/png/59026724/1773627081905-adab190f-8d4e-4869-83ad-c137784be777.png)

> 输入源句子的词向量，进入 Encoder 第一层；
>
> 第一层 Encoder：用当前的 W_Q/W_K/W_V（初始是随机值，训练中不断更新），把词向量转成 Q=K=V；
>
> 计算自注意力（得到 “加权后的 V”），再经过残差 + 归一化，得到第一层的输出 x₁（这是第一层的 K/V）；
>
> 第二层 Encoder：用 x₁作为输入，重新通过 W_Q/W_K/W_V 计算新的 Q=K=V，再算自注意力，得到 x₂；
>
> 直到第六层，输出最终的 x₆（这就是 Encoder 给 Decoder 的 K/V）；
>
> 反向传播时，会更新所有层的 W_Q/W_K/W_V（让 Q/K/V 更精准）。
>

Decoder：同样先embedding input（和encoder一样，结合position，然后进行self attention）。eg. input 是 <bos>car。然后通过cross attention，将encoder的context vector（用K，V）和decoder的embedding（用Q）进行计算，将attention score乘decoder embedding的V得到context vectors。

![](https://cdn.nlark.com/yuque/0/2026/png/59026724/1773629046904-cf2da031-1c17-462e-89a9-b7fc6f14a81e.png)![](https://cdn.nlark.com/yuque/0/2026/png/59026724/1773629075125-28559635-2363-4497-8e3d-a795c2e53868.png)

encoder和decoder一起训练，训练目标是decoder里预测词和真实词的交叉熵损失，确定矩阵W

最后通过projection，用softmax不断预测下个概率最大的token，直到<eos>



In Robotics field：

input: pictures-angle, instruction

Vit：trun pictures into patches(sequence), 同样拼接position一起传入transformer，计算attention score。不过最重要的cls token，也就是第一个pixel，经过attention矩阵相乘得到context vector，它也包括了其他像素块的信息。再经过mlp+softmax进行分类



至此，很容易理解为什么smolvla中有kv，qkv之分，前者的q来自decoder并用cross attention。

![](https://cdn.nlark.com/yuque/0/2026/png/59026724/1773640638428-0e48811c-8ec2-4fd2-ad83-02e226180fdf.png)

也可以理解groont的架构了

![](https://cdn.nlark.com/yuque/0/2026/png/59026724/1773640828002-c513b30c-c4b1-43a3-b292-1564ad34e261.png)



## ACT（Action Chunking Transformers）
目标：基于当前robot state（以SO101为例，6个joint angles），输出next time stamp的动作序列（先获取distributions of joint angles）

Architecture1:

![](https://cdn.nlark.com/yuque/0/2026/png/59026724/1773664334026-ae69815d-4db0-4e83-aecd-4291eabe824d.png)

问题在于只能接收proprierceptive data（本体感觉数据，也就是关节角度），而没有image。这会有个bug：只会倒牛奶的这个动作，但是不知道杯子在哪里。所以需要改进。



Architecture2:

![](https://cdn.nlark.com/yuque/0/2026/png/59026724/1773664588698-d60fb50b-518b-4af9-a853-6fdf114b7a95.png)

把图像也传入进来，但还是有bug：生成的预测是下一个时间步的action，不是一个action sequence，并且往后生成不断基于前面的observation，这需要架构上进一步的改进。

![](https://cdn.nlark.com/yuque/0/2026/png/59026724/1773665813369-24d268d9-af28-4f9c-84e6-2721ac08614c.png)

这个架构依然不满足action的递归，需要用transformer代替mlp

首先是encoder：

![](https://cdn.nlark.com/yuque/0/2026/png/59026724/1773667179282-42ff0b09-b67e-4668-a7e1-42050844379f.png)

transformer可以理解actions之间的attention

对input：cls，joints，action拆分成token并计算attention，然后计算context vectors，重点使用cls的，然后将其project to均值和方差，得到latent variables

input这个observed state和action pair就像前面不同人写不同类型的hello，相当于不同的人在不同state下执行不同的action，然后映射到latent space里，学习一种动作执行的策略。不能用回归，是因为目标不是要取所有可能动作的均值，而是基于state的action策略，也就是完成同一任务的不同方法

然后是decoder：负责做出最终的预测

![](https://cdn.nlark.com/yuque/0/2026/png/59026724/1773670079816-770cda38-4121-417d-b4ba-3a2acc1dc196.png)

为了理解杯子所处的3d位置信息，需要融合overview视角的摄像头和手掌上的摄像头，所以需要加个encoder在前面：这个encoder用来fusing objectives。

先用cnn对图像预处理提取特征

![](https://cdn.nlark.com/yuque/0/2026/png/59026724/1773671096134-4ae155fb-93cd-49d7-b19f-5a5448a5672d.png)

![](https://cdn.nlark.com/yuque/0/2026/png/59026724/1773670544613-3aa168f4-e858-436b-8843-b7e2db7da098.png)![](https://cdn.nlark.com/yuque/0/2026/png/59026724/1773670681446-92d89ffa-df4b-4cd1-9b8e-a05b8fd8ab6b.png)

然后用transformer的attention机制把所有信息融合。input是1个joints token，1个latent variable token，1200个像素token，然后相应得到1202个context vector。

![依旧是transformer架构](https://cdn.nlark.com/yuque/0/2026/png/59026724/1773671506774-a1ff47b3-87b0-49ac-892f-33190353a10f.png)

![比如这个就实现了智能，把夹爪和把手联系到一起](https://cdn.nlark.com/yuque/0/2026/png/59026724/1773671613153-89cce2ff-b260-420d-9b1d-5627331ec091.png)

然后再到decoder，Q很明显就是时间步（比如t+1到t+6，512维的，而且是fixed），K和V由前面1202维的context vector得到。依旧是cross attention，然后得到6个context vectors作为输出。再将每个512维的输出project到6个joints上 

![终极版：需要joints是为了理解current state（conditional），cls是为了融合全局信息，transformer架构是为了能输出action chunk并且建立actions之间的关联，第二个encoder的目的是跨模态的融合信息，position的目的是学习动作顺序](https://cdn.nlark.com/yuque/0/2026/png/59026724/1773672335289-b6d403f4-dba3-4ec0-b9d0-940d8e13e59d.png)





## ACT部署到SO101并进行训练
主要就是连接leader arm和follower arm，实现模仿学习，基于几个episode训练，用夹爪实现pick-and-place的策略，并可以泛化

可以参考huggingface上lerobot hackson的一些有趣的应用案例，比如可以用夹爪+筷子实现一些功能（这可以被封装成一个任务模块）

## Maniskill仿真
这是一个和isaac lab类似的适合入门的仿真环境。

将会把上一节课中真实世界中采集-训练-部署的流程，完全转移到仿真环境里

官方colab可以跑通一遍

![](https://cdn.nlark.com/yuque/0/2026/png/59026724/1773734650258-9ea4d823-4161-4967-b078-e6594c51e8b6.png)

多尝试不同的奖励函数和环境



然后sim2real 可以参考： github.com/StoneT2000/lerobot-sim2real

## 下一阶段学习指南：
![](https://cdn.nlark.com/yuque/0/2026/png/59026724/1773732501989-0ab8ddca-1f35-4c6f-8bab-6f0a27ab4955.png)

![](https://cdn.nlark.com/yuque/0/2026/png/59026724/1773732532199-72c94be8-61a8-4904-84a7-46c67582a084.png)

![](https://cdn.nlark.com/yuque/0/2026/png/59026724/1773732700216-731987a0-2d48-4e76-b645-718b1c4ca9cb.png)

![](https://cdn.nlark.com/yuque/0/2026/png/59026724/1773732709692-13217087-265c-4a3b-b1cd-2afbe6ec1850.png)

![](https://cdn.nlark.com/yuque/0/2026/png/59026724/1773732741494-7545d839-daf8-492a-ab9b-027f5c72c0c5.png)

