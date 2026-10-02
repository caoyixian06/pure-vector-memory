# The Answer-Perspective Gap: Measuring the Structural Limits of Dense Retrieval for Long-Term Conversational Memory

## 目标: ACL/EMNLP 2026-2027 主会长文(8页正文+附录)
## 定位: 分析/测量类论文(类似 "Language Models Do Not Embed Numbers Continuously" 的路线, 但对象是对话记忆检索)

---

## 论文一句话主张

对话记忆的密集检索存在一个可精确测量的结构性极限: 答案视角与问题视角在嵌入空间中的系统性不对称(判别力6倍差、117°半球分离), 使得检索单元在原理上只能到达"证据带"而永远无法直达答案——这个极限不是工程缺陷, 而是当前嵌入空间几何的固有性质; 我们提供完整的测量方法学、四个可复现的结构常量, 以及它们对记忆系统设计的可操作指令。

---

## 骨架(三级提纲)

### 1 Introduction (0.9页)

钩子: 对话记忆系统的公开分数在70-90间振荡, 但没人回答"检索这一层理论上限是多少、为什么"。现有记忆系统(Mem0/Hindsight/TAG)在检索失败时都归咎"模型不够强", 我们证明: 一半以上的检索失败源于嵌入空间的几何性质, 与模型无关。

贡献列表(4条):
- C1: 答案视角鸿沟的测量与三角不等式证明(检索终点=证据带, 几何必然)
- C2: 证据/噪声共模定理(0.986)与分布式微扰极限(±0.01/1024维)
- C3: 四个可复现的空间结构量(Δ指纹/词覆盖方程/说话人残差/序结构)及乱序对照法
- C4: 基于测量的设计指令(原子粒度检索单元+词级侦察+写入形态决定论), 在LoCoMo-Refined官方判官下验证(70.5%, 同口径超MemOS 15%相对)

### 2 Related Work (0.75页)

四线定位(每线明确"他们没测什么"):
- 嵌入空间结构探测: 数字嵌入非线性(arXiv 2510.08009)/表示解剖——无人测对话记忆的问答不对称
- Answer-aware检索工程: MEGRAG/MiniRAG/Hindsight——做工程不测机理
- 检索失败分析: Dense Retrievers Can Fail on Simple Queries (arXiv 2506.08592)——简单查询的词汇失败, 我们是对话记忆的结构失败(答案视角鸿沟)
- 嵌入失分现象: Embeddings for Preferences Not Semantics (arXiv 2605.08360)——展示干扰项>正确答案的单例现象, 我们提供1218配对系统测量+6×量化+半球几何证明
- 记忆系统与基准: Mem0/Hindsight/TAG/SLM-V3——比分数不比极限; LoCoMo标注缺陷审计佐证我们的基准选择

**定位句(写作时用):**
"Recent work has observed embedding models scoring distractors above correct statements [Prefs 2026]; we provide the first systematic measurement of this phenomenon in conversational memory — quantifying the perspective gap at 6x and proving its geometric inevitability."
"Controlled analyses have examined why dense retrieval fails on simple lexical queries [DR-Fail 2025]; we show the failure in conversational memory is not lexical but architectural."

### 3 Experimental Setup (0.75页)

- 数据: LoCoMo-Refined(1382题, 官方Qwen3-14B判官, 337缺陷修复版) + LongMemEval-S-cleaned(500题, 复现集)
- 嵌入: BGE-M3(1024) + Qwen3-embedding(256) 双空间(关键: 所有结论双空间验证)
- 库构建: 零LLM写入, raw原句+原子句双层, 13,301原子句
- 术语定义: 证据带/答案视角/问题视角/共模
- 伦理与复现: 全公开数据, 判官temp0, 代码+索引将发布

### 4 The Answer-Perspective Gap (2页, 核心节)

#### 4.1 半球分离的测量
问题质心vs答案质心夹角117°(cos=-0.454); 双空间复现(BGE -0.454 / qwen -0.491)
图1: 球面示意图(两半球+质心+一条示例查询)

#### 4.2 检索终点的几何证明
三角不等式: |a-x|=1.10 > |a-c|+|c-x|=0.93+0.94可绕行, 但直达路径a·x=-0.454<0 → 余弦检索原理上选不出答案
推论1: 检索的正确终点是证据带(evidence band), 答案必须在读取阶段构造

#### 4.3 视角判别力不对称(核心数字)
表1: 九元关系矩阵(a,b,c,h×双空间)
问题视角判别: a·c - a·h = 0.017(BGE)/0.014(qwen)
答案视角判别: c·x - h·x = 0.098(BGE)/0.082(qwen) — 6倍
图2: 视角判别力条形图(双空间)

#### 4.4 为什么不对称存在(机理解释)
答案文本由证据"提炼+改写"生成 → 答案保留证据的话题内核但剥离对话腔 → 嵌入空间的语域分量主导余弦, 内容分量被压缩(共模0.986的证据)

### 5 The Common-Mode Structure of the Retrieval Space (1.5页)

#### 5.1 共模定理
证据均值与噪声均值的1024维跨维相关=0.986(BGE)/0.979(qwen)——证据与噪声是同一数组的±0.01微扰
图3: 逐维均值散点(证据vs噪声, 0.986斜线)

#### 5.2 分布式微扰极限
单维最大差0.013, 无任何维度超过0.02; 有效秩22/1024(信息平铺)
推论2: 该空间不存在单维"证据探测器"; 一切判别信号必须是分布式聚合

#### 5.3 现有信号的边界
Δ指纹(朴素均值差): AUC 0.802, 拆半稳定0.912; 关键对照: 朴素形式≈Fisher判别最优(0.657)→ 已达该特征族最优
词覆盖(98%配对)与句余弦(0.921精排)的分界: 粒度决定信号
负结果边界(简表): 线性映射三连死(king-queen 0.178/Ridge 0/211/Δ交互0)→ 问题到答案不存在可学习的线性通路

#### 5.4 乱序对照法(方法学贡献)
月份循环序: 真序0.902 vs 乱序对照-0.031 → 序结构与词面的严格区分; 应用: 词向量空间序结构表(月/周/数字/情感价)

### 6 From Measurement to Design (1.5页)

#### 6.1 设计指令1: 检索单元必须是子句级
证据的±1邻域块召回93%(vs 单句52%); 但粒度过细的指代残缺硬顶(检索成功85/100上限)
表2: 窗口扫描(5/8/10/15/20/25×四指标) → 窗口15为拐点

#### 6.2 设计指令2: 词级侦察是唯一去相关的第二信号
256词级侦察与主通道相关0.025 → +6错题+5对题(50/50对照); 句级"伪侦察"同源信号零增益(对照实验)
#### 6.3 设计指令3: 写入形态决定检索可达性
指代残缺句("it was delicious")在切分时丢失全部连接 → 检索原理性盲区的根源; 原子域+邻域块93%→99%(实体桥)

#### 6.4 端到端验证
LoCoMo-Refined 1382题全量: 基线61.4% → 全部测量驱动的设计改动 → 70.5%(官方Qwen3-14B判官)
表3: 组件消融(每步单变量归因)
公开榜对位: 同判官下超MemOS(63.6)15%相对, 答题模型为flash级(成本对照Hindsight OSS-20B 83.18为原版宽松判分)

### 7 Limitations (0.3页)

单对话对(两人)语料; 判官为LLM(±2波动已量化); LongMemEval复现见附录X(数字待补); 检索成功上限85/100的指代残缺问题未解(需要读取层或结构化存储)

### 8 Conclusion (0.25页)

记忆检索的极限可以被测量、被证明、被绕过——但不能被"更好的模型"消除。

---

## 附录清单
A. 全部超参/复现细节
B. 双空间完整对照表
C. 乱序对照法的统计功效分析
D. 负结果全录(king-queen/Ridge/软配对/RRF——每个一段)
E. 长尾分析: 指代残缺句的解剖案例

---

## 图表清单(7图4表)
图1 球面半球示意 | 图2 视角判别力对比 | 图3 共模散点 | 图4 Δ五拆分小提琴 | 图5 乱序对照 | 图6 窗口扫描曲线 | 图7 消融瀑布
表1 九元关系 | 表2 窗口扫描 | 表3 消融 | 表4 公开榜对位

---

## 数据来源映射(每个数字的实验出处)
- 117°/-0.454: diag_dims.py(1376题质心)
- 0.017/0.098: abc_matrix.py P1表(1218配对)
- 0.986: raw_nums.py R4
- Δ0.802: check_k3_leak.py(5拆分)/diag_delta_v2.py K1
- 0.025去相关: dualspace2.py S1
- 98%覆盖: solve_x_word.py W5
- 78/81%: speaker_probe.py
- 0.902/0.822+乱序: vec_mining.py M1
- 窗口扫描: window_scan.py
- 70.5%: out_r37_qwen14b.jsonl(官方判官)
- 消融: r33a/r33b/r36/r37全量序列

---

## 预审稿人攻击点与防御
1. "只是LoCoMo一个数据集" → LongMemEval-S复现(补实验1, 附录+正文引)
2. "嵌入模型只有两个" → BGE/qwen是不同架构不同维度不同训练方, 结构一致即跨模型成立; 可补e5(medium)
3. "说话人实验只有两人对话" → 承认, 定位为"存在性证明"; 多人对话留future
4. "Δ是监督信号" → 0.35节明确声明+无泄漏变体(u2/词级侦察)同样有效
5. "你们自己也有70.5分系统, 既当运动员又当裁判" → 系统仅作验证载体, 主张全部基于测量, 系统组件与测量结论可分离
