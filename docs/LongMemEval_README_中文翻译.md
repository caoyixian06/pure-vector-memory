# LongMemEval:面向聊天助手长期交互记忆的基准测试(ICLR 2025)

> 原仓库:https://github.com/xiaowu0162/LongMemEval

🖋 作者:Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, Dong Yu

我们提出 LongMemEval——一个全面、高难度、可扩展的基准,用于测试聊天助手的长期记忆能力。

[![官网](https://img.shields.io/badge/🌐-网站-red)](https://xiaowu0162.github.io/LongMemEval.io/)
[![论文](https://img.shields.io/badge/📄-论文(ICLR_2025)-blue)](https://arxiv.org/abs/2410.10813)
[![数据](https://img.shields.io/badge/🤗-数据-green)](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)

## ⚠️ 新闻

- [2026/05] 请关注 LongMemEval-V2:智能体(Agentic)场景下的长期记忆。
- [2025/09] 我们进一步清洗了历史会话,以防止对答案正确性的干扰。更新后的基准见[这里](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned),变更日志见[这里](https://github.com/xiaowu0162/LongMemEval/blob/main/assets/changelogs.md)。你也可以通过[这个 Google Drive 链接](https://drive.google.com/drive/folders/1b7PrlaUg3B1nZTFnS8o-Y7w1UqSd4q0q)访问该文件。
- [2025/02] LongMemEval 被 ICLR 2025 接收。
- [2024/10] 基准发布。

## 🧠 LongMemEval 概览

我们发布了 500 个高质量问题,用于测试五项核心长期记忆能力:

- 信息提取(Information Extraction)
- 多会话推理(Multi-Session Reasoning)
- 知识更新(Knowledge Updates)
- 时间推理(Temporal Reasoning)
- 拒答/弃权(Abstention)

受"大海捞针"(needle-in-a-haystack)测试的启发,我们设计了一条属性可控(attribute-controlled)的流水线,为每个问题编译出连贯、可扩展、带时间戳的聊天历史。LongMemEval 要求聊天系统在线解析动态交互并据此记忆,并在全部交互会话结束后回答问题。

## 🛠️ 环境搭建

### 数据

LongMemEval 数据集已在 Hugging Face 正式发布。请下载并解压到 `data/` 文件夹:

```
mkdir -p data/
cd data/
wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json
wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_m_cleaned.json
cd ..
```

### 环境

我们建议为该项目使用 conda 环境。可按以下步骤搭建:

#### 仅做评测

如果你只需要对自己的系统产出计算指标,可以安装这个最小依赖集,即可运行 `src/evaluation/evaluate_qa.py`:

```
conda create -n longmemeval-lite python=3.9
conda activate longmemeval-lite
pip install -r requirements-lite.txt
```

#### 完整支持

如果你还想运行论文中介绍的记忆系统,请改为搭建以下环境:

```
conda create -n longmemeval python=3.9
conda activate longmemeval
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-full.txt
```

我们已在装有 CUDA 12.1 的 Linux 机器上测试过该环境。如果你使用其他平台,可能需要修改依赖项。

## 📜 数据集格式

数据包内含三个文件:

- `longmemeval_s.json`:论文中提出的 LongMemEval_S。将全部聊天历史拼接后,对 Llama 3 而言大约消耗 115k token(约 40 个历史会话)。
- `longmemeval_m.json`:论文中提出的 LongMemEval_M。每个聊天历史大约包含 500 个会话。
- `longmemeval_oracle.json`:使用 oracle 检索的 LongMemEval。历史中只包含证据会话。

每个文件内有 500 个评测实例,每个实例包含以下字段:

- `question_id`:每个问题的唯一 id。
- `question_type`:以下之一:`single-session-user`(单会话-用户)、`single-session-assistant`(单会话-助手)、`single-session-preference`(单会话-偏好)、`temporal-reasoning`(时间推理)、`knowledge-update`(知识更新)、`multi-session`(多会话)。如果 `question_id` 以 `_abs` 结尾,则该题为`拒答(abstention)`题。
- `question`:问题内容。
- `answer`:期望模型给出的答案。
- `question_date`:提问日期。
- `haystack_session_ids`:历史会话 id 列表(`longmemeval_s.json` 和 `longmemeval_m.json` 中按时间戳排序;`longmemeval_oracle.json` 中不排序)。
- `haystack_dates`:历史会话时间戳列表。
- `haystack_sessions`:用户-助手聊天历史会话的实际内容列表。每个会话是一个回合(turn)列表;每个回合是格式为 `{"role": user/assistant, "content": 消息内容}` 的对话。对于包含所需证据的回合,会额外提供字段 `has_answer: true`。该标签用于回合级(turn-level)记忆召回准确率评测。
- `answer_session_ids`:代表证据会话的会话 id 列表。用于会话级(session-level)记忆召回准确率评测。

## 📊 测试你的系统

要在 LongMemEval 上测试,你可以直接把带时间戳的历史喂给你的聊天系统,收集输出,再用我们提供的评测脚本打分。为此,请把输出保存为 `jsonl` 格式,每行包含两个字段:`question_id` 和 `hypothesis`。然后通过以下命令运行评测脚本:

```
export OPENAI_API_KEY=YOUR_API_KEY
export OPENAI_ORGANIZATION=YOUR_ORGANIZATION     # 如果你的 key 只属于一个组织,可省略
cd src/evaluation
python3 evaluate_qa.py gpt-4o your_hypothesis_file ../../data/longmemeval_oracle.json
```

运行该脚本会把评测日志保存到名为 `[your_hypothesis_file].log` 的文件中。文件里每一行都会新增一个字段 `autoeval_label`。虽然 `evaluate_qa.py` 已经会报告平均分,你也可以用以下命令从日志聚合分数:

```
(假设你已在 src/evaluation 目录下)

python3 print_qa_metrics.py gpt-4o your_hypothesis_file.log ../../data/longmemeval_oracle.json
```

## 💬 构建自定义聊天历史

LongMemEval 支持为一个问题实例编译任意长度的聊天历史,方便你在 `LongMemEval_M` 之上继续扩展难度。

### 下载语料

请从此[链接](https://drive.google.com/file/d/1Akg0ePkjbwzYTJ5uHdqOZ1ZyIPGVuqjv/view?usp=sharing)下载压缩数据并解压到 `data/custom_history` 下。发布的数据包含三部分:

- `1_attr_bg/data_1_attr_bg.json`:用户属性与背景。
- `2_questions`:问题、答案、证据陈述,以及证据会话。
- `5_filler_sess/data_5_filler_sess.json`:来自 ShareGPT 和 UltraChat 的填充会话。
- `6_session_cache/data_6_session_cache.json`:基于从背景中抽取的事实模拟出的用户会话。

### 复现 LongMemEval 的历史编译

运行 `python sample_haystack_and_timestamp.py task n_questions min_n_haystack_filler max_n_haystack_filler enforce_json_length` 即可复现 LongMemEval 的历史编译流程。

- `task` 是任务名。发布数据中的命名与论文不同,对应关系如下:

  | 数据中的名称 | 官方名称 |
  | --- | --- |
  | single_hop | single-session-user |
  | implicit_preference_v2 | single-session-preference |
  | assistant_previnfo | single-session-assistant |
  | two_hop | multi-session |
  | multi_session_synthesis | multi-session |
  | temp_reasoning_implicit | temporal-reasoning |
  | temp_reasoning_explicit | temporal-reasoning |
  | knowledge_update | knowledge-update |

- `n_questions` 是使用的最大问题数。
- `min_n_haystack_filler` 和 `max_n_haystack_filler` 限制历史中包含的会话数量。`longmemeval_s` 用 80,`longmemeval_m` 用 500。
- `enforce_json_length` 用于限制聊天历史的长度。`longmemeval_s` 用 115000;`longmemeval_m` 不使用该限制,可设为一个很大的数。

### 构建你自己的历史

要构建自定义聊天历史,可按 `2_questions` 和 `6_session_cache` 中的格式创建问题与证据会话,然后用类似命令运行 `sample_haystack_and_timestamp.py`。

## 🚀 运行记忆系统实验

我们在 `src/retrieval` 和 `src/generation` 文件夹下提供了记忆检索与检索增强问答的实验代码。

### 准备

如果你想用 OpenAI 模型作为阅读器(reader),请在 `src/generation/run_generation.sh` 第 33、34 行填入你的 OpenAI organization ID 和 key。

如果你想测试开放权重的阅读器 LLM,我们通过 `vllm` 本地启动的 OpenAI API 模拟器来支持。用以下命令启动服务:

```
cd src/utils
bash serve_vllm.sh GPU MODEL PORT TP_SIZE
```

- `GPU`:要使用的 GPU 列表,逗号分隔。
- `MODEL`:要使用模型的别名。可在 `serve_vllm.sh` 中查看或配置。
- `PORT`:服务监听端口,默认 8001。若修改端口,请确保同步修改 `src/generation/run_generation.sh` 第 38 行。
- `TP_SIZE`:张量并行大小,必须小于等于 GPU 参数中指定的 GPU 数量。
- 如果因显存限制需要限定最大 token 数,可使用:

```
bash serve_vllm_with_maxlen.sh GPU MODEL MAXLEN PORT TP_SIZE
```

### 长上下文生成

要运行把完整历史提供给模型的长上下文生成基线,可使用:

```
cd src/generation
bash run_generation.sh DATA_FILE MODEL full-history-session TOPK [HISTORY_FORMAT] [USERONLY] [READING_METHOD]
```

- `DATA_FILE`:已发布 json 文件之一的路径。注意 `longmemeval_s.json` 和 `longmemeval_oracle.json` 设计上可放入 128k 上下文的模型,而 `longmemeval_m.json` 太长,不适合长上下文测试。
- `MODEL`:要使用模型的别名。可在 `run_generation.sh` 中查看或配置。
- `TOPK`:提供给阅读器的最大历史会话数。建议设一个很大的数(如 1000)以确保包含全部历史会话。
- `HISTORY_FORMAT`:呈现历史的格式,可选 `json` 或 `nl`。推荐 `json`。
- `USERONLY`:是否从提示中去掉助手侧消息。推荐 `false`。
- `READING_METHOD`:阅读方式,可选 `direct`、`con` 或 `con-separate`。推荐 `con`,即让模型先提取有用信息、再基于信息推理。

日志文件会生成在 `generation_logs/` 目录下。之后按上文"测试你的系统"一节操作即可评测问答正确性。

### 记忆检索

按以下说明在 LongMemEval 上运行记忆索引与检索。

#### 基线检索

用以下命令运行基线检索:

```
cd src/retrieval
bash run_retrieval.sh IN_FILE RETRIEVER GRANULARITY
```

- `IN_FILE`:输入路径。
- `RETRIEVER`:`flat-bm25`、`flat-contriever`、`flat-stella`(Stella V5 1.5B)或 `flat-gte`(gte-Qwen2-7B-instruct)。对 `flat-stella`,代码需要你从原仓库手动下载模型。
- `GRANULARITY`:记忆索引的取值粒度,支持 `turn`(回合)或 `session`(会话)。

注意:对稠密向量模型,我们支持多 GPU 检索。默认代码会使用所有可用 GPU。

脚本会把检索结果输出到 `retrieval_logs/` 下并打印评测指标。也可以用以下命令从日志打印指标:

```
python3 src/evaluation/print_retrieval_metrics.py log_file
```

另注:评测检索时我们总是跳过 30 个拒答(abstention)实例,因为这些实例通常指代不存在的事件,没有可用的标准答案位置。

#### 索引扩展

要运行带键扩展(key expansion)的实验,请从此[链接](https://drive.google.com/file/d/1TB1DgVbqGwJ0QV1Z1Xo1Z1Xo1Z1Xo1Z1X/view)下载已发布的键扩展输出并放到 `LongMemEval/index_expansion_logs/` 目录下,然后运行:

```
cd src/retrieval
bash run_retrieval.sh IN_FILE RETRIEVER GRANULARITY EXPANSION_TYPE JOIN_MODE CACHE
```

- `EXPANSION_TYPE`:支持 `session-summ`、`session-keyphrase`、`session-userfact`、`turn-keyphrase`、`turn-userfact`。
- `JOIN_MODE`:支持三种模式:
  - `separate`:以扩展内容为键,新增一个(键, 值)对。
  - `merge`:把扩展内容与原键合并得到新键。
  - `replace`:用扩展内容替换原键。
- `CACHE`:与 `EXPANSION_TYPE` 对应的缓存文件路径。

我们在 `src/index_expansion` 下发布了离线生成键扩展输出的代码,供参考。

#### 时间感知查询扩展

我们提供了通过以下方式缩小检索搜索空间的实现:从会话中抽取带时间戳的事件、从查询中推断时间范围、再用该范围收窄搜索空间。首先从此[链接](https://drive.google.com/file/d/1vGfT3ZvQ3p0Z1Xo1Z1Xo1Z1Xo1Z1X/view)下载已抽取的带时间戳事件并解压到 `LongMemEval/index_expansion_logs/` 下,然后运行:

```
cd src/index_expansion
python3 temp_query_search_pruning.py TIMESTAMP_EVENT_FILE RETRIEVAL_LOG GRANULARITY
```

- `TIMESTAMP_EVENT_FILE`:你下载的带时间戳事件文件。
- `RETRIEVAL_LOG`:任意检索实验的输出。
- `GRANULARITY`:`session` 或 `turn`,须与另外两个参数保持一致。论文中使用 `session` 作为粒度。

### 检索增强生成

使用检索到的记忆进行问答,可用以下命令:

```
cd src/generation
bash run_generation.sh RETRIEVAL_LOG_FILE MODEL EXP TOPK [HISTORY_FORMAT] [USERONLY] [READING_METHOD]
```

- `RETRIEVAL_LOG_FILE`:应为检索步骤的输出。具体来说,该步骤依赖检索阶段添加的 `retrieval_results` 字段。
- `EXP` 应为 `[RETRIEVER]-[GRANULARITY]` 形式,如 `flat-stella-session`。
- 其余参数同上文介绍。

## 引用

如果本工作对你有用,请引用:

```
@article{wu2024longmemeval,
      title={LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory},
      author={Di Wu and Hongwei Wang and Wenhao Yu and Yuwei Zhang and Kai-Wei Chang and Dong Yu},
      year={2024},
      eprint={2410.10813},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.10813},
}
```
