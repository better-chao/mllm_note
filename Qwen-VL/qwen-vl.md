
# Qwen-VL 技术报告详解

## 一、论文基本信息

**标题**: Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond

**作者**: Jinze Bai, Shuai Bai, Shusheng Yang等 (阿里巴巴集团)

**发表时间**: 2023年8月 (arXiv:2308.12966v3)

**开源地址**: https://github.com/QwenLM/Qwen-VL

---

## 二、研究背景与动机

### 2.1 现有问题

1. **开源LVLM训练不足**: 当前开源大规模视觉-语言模型(LVLMs)普遍存在训练和优化不充分的问题，远落后于专有模型(如GPT-4V、Claude)

2. **粗粒度理解局限**: 大多数开源LVLMs只能进行粗粒度的图像感知，缺乏细粒度理解能力，如**物体定位(grounding)**和**文本阅读**

3. **实际应用受限**: 由于缺乏精细化的视觉理解能力，现有模型难以有效地在复杂真实场景中辅助用户

### 2.2 研究目标

开发一个**通用且高性能**的视觉-语言基础模型，具备：
- 图像描述和问答
- **视觉定位(Visual Grounding)**
- **文本阅读(OCR)**
- **多语言支持**(英文+中文为主)
- **多图像交互对话**

---

## 三、模型架构设计

### 3.1 整体架构

Qwen-VL采用**三组件架构**，总参数量**9.6B**：

```
[图像输入] 
    ↓
[Visual Encoder (1.9B)]  ← ViT-bigG (OpenCLIP预训练)
    ↓
[Position-aware VL Adapter (0.08B)]  ← Cross-Attention压缩
    ↓
[Large Language Model (7.7B)]  ← Qwen-7B
    ↓
[文本输出/边界框输出]
```

### 3.2 Visual Encoder (视觉编码器)

**设计细节**:
- **架构**: Vision Transformer (ViT)
- **初始化**: OpenCLIP的ViT-bigG预训练权重
- **参数量**: 1.9B
- **分辨率**: 
  - Stage 1 (预训练): 224×224
  - Stage 2 (多任务预训练): **448×448** (提升信息密度)
  - Stage 3 (SFT): 448×448
- **Patch size**: 14×14
- **输出序列长度**: 
  - 224分辨率: (224/14)² = 256
  - 448分辨率: (448/14)² = 1024

**关键改进**: 
- 在Stage 2提升分辨率到448×448，减少下采样信息损失
- 实验对比了Window Attention vs Global Attention，最终选择**Global Attention**（虽然计算量大，但收敛性能更好）

### 3.3 Position-aware Vision-Language Adapter

**核心创新**: 引入位置感知机制

**架构设计**:
```python
# 伪代码示意
class PositionAwareAdapter:
    def __init__(self):
        self.learnable_queries = nn.Embedding(256, hidden_dim)  # 256个可学习query
        self.cross_attention = CrossAttention()
        self.pos_encoding_2d = AbsolutePositionEncoding2D()  # 2D位置编码
    
    def forward(self, image_features):
        # image_features: [batch, 1024, dim] for 448x448 images
        
        # 添加2D位置编码到query-key对
        query = self.learnable_queries.weight
        key = image_features + self.pos_encoding_2d
        
        # Cross-Attention压缩: 1024 → 256
        compressed_features = self.cross_attention(query, key, image_features)
        
        return compressed_features  # [batch, 256, dim]
```

**设计动机**:
1. **效率问题**: 直接输入1024长度的视觉特征序列会导致LLM计算开销过大
2. **信息保留**: 单层Cross-Attention压缩到256长度
3. **位置保持**: **2D绝对位置编码**注入到cross-attention的query-key对中，缓解压缩过程中位置信息的损失

**消融实验** (Appendix E.2):
- 测试了64、144、256、400四种query数量
- **256**是最优选择：
  - 64太少，信息损失严重
  - 400太多，收敛困难且计算量大
  - 256在性能和效率间达到最佳平衡

### 3.4 Large Language Model

- **基座模型**: Qwen-7B (7.7B参数)
- **冻结策略**:
  - Stage 1: **冻结LLM**，只训练Vision Encoder和Adapter
  - Stage 2: **解冻LLM**，端到端训练全部参数
  - Stage 3: **冻结Vision Encoder**，只训练LLM和Adapter

### 3.5 Input-Output Interface (输入输出接口)

**图像输入格式**:
```
<img>image_path.jpg</img>
```

**边界框格式设计**:

这是Qwen-VL的**重要创新**之一：

```
<ref>描述内容</ref><box>(x_topleft, y_topleft),(x_bottomright, y_bottomright)</box>
```

**关键特点**:
1. **归一化坐标**: 坐标归一化到[0, 1000)范围
2. **字符串化表示**: 将坐标直接转换为字符串，通过LLM的tokenizer处理，**不需要额外的位置词汇表**
3. **特殊token标记**:
   - `<box>`, `</box>`: 标记边界框字符串
   - `<ref>`, `</ref>`: 标记边界框所指代的对象
   - `<img>`, `</img>`: 标记图像特征序列

**示例**:
```
<img>image.jpg</img>Generate the caption in English with grounding: 
Beautiful shot of <ref>bees</ref><box>(661,612),(833,812)</box> 
gathering nectars from <ref>an apricot flower</ref><box>(224,13),(399,313)</box>
```

---

## 四、训练流程

### 4.1 三阶段训练Pipeline

```
Stage 1: Pre-training (50k steps, ~1.5B samples)
    ↓
Stage 2: Multi-task Pre-training (19k steps)
    ↓
Stage 3: Supervised Fine-tuning (8k steps, 350k samples)
```

### 4.2 Stage 1: 预训练 (Pre-training)

**训练目标**: 建立基础的图像-文本对齐能力

**数据规模**: 
- **原始数据**: 50亿图像-文本对
- **清洗后**: 14亿对 (保留率28%)
- **语言分布**: 77.3%英文 + 22.7%中文

**数据来源** (表2):
| 数据集 | 原始量 | 清洗后 | 保留率 |
|--------|--------|--------|--------|
| LAION-en | 2B | 280M | 14% |
| LAION-COCO | 600M | 300M | 50% |
| DataComp | 1.4B | 300M | 21% |
| Coyo-700M | 700M | 200M | 28% |
| CC12M | 12M | 8M | 66% |
| CC3M | 3M | 3M | 100% |
| SBU | 1M | 0.8M | 80% |
| COCO Caption | 0.6M | 0.6M | 100% |
| LAION-zh | 108M | 105M | 97% |
| In-house Data | 220M | 220M | 100% |

**数据清洗流程** (Appendix A.1):
1. 移除宽高比过大的图像
2. 移除过小的图像
3. 基于CLIP score过滤(数据集特定阈值)
4. 移除包含非英文/非中文字符的文本
5. 移除包含emoji的文本
6. 移除文本长度过短或过长的样本
7. 清理HTML标签
8. 清理特定不规则模式

**训练配置**:
- **图像分辨率**: 224×224
- **Batch size**: 30,720
- **优化器**: AdamW (β₁=0.9, β₂=0.98, eps=1e-6)
- **学习率**: 
  - Peak: 2e⁻⁴
  - Min: 1e⁻⁶
  - Schedule: Cosine decay + 500步warm-up
- **ViT学习率衰减**: Layer-wise decay, factor=0.95
- **Weight decay**: 0.05
- **梯度裁剪**: 1.0
- **训练步数**: 50,000步
- **参数冻结**: **冻结LLM**，只训练ViT和Adapter
- **Loss**: Next-token prediction (文本token)

**收敛曲线分析** (图6):
- 训练loss稳定下降
- Zero-shot VQA性能在波动中上升
- Flickr30K CIDEr从62提升到76

### 4.3 Stage 2: 多任务预训练 (Multi-task Pre-training)

**训练目标**: 引入**细粒度视觉理解能力**(Grounding、OCR)和**交错图文数据**

**数据组成** (表3):

| 任务 | 样本量 | 数据集 |
|------|--------|--------|
| Captioning | 19.7M | LAION, DataComp, Coyo, CC12M/3M, SBU, COCO |
| VQA | 3.6M | GQA, VGQA, VQAv2, DVQA, OCR-VQA, DocVQA, TextVQA, ChartQA, AI2D |
| **Grounding** | 3.5M | **GRIT** |
| **Ref Grounding** | 8.7M | GRIT, Visual Genome, RefCOCO, RefCOCO+, RefCOCOg |
| **Grounded Cap.** | 8.7M | GRIT, Visual Genome, RefCOCO, RefCOCO+, RefCOCOg |
| **OCR** | 24.8M | **SynthDoG-en/zh**, Common Crawl PDF & HTML |
| Pure-text | 7.8M | In-house Data |

**关键数据构建**:

1. **OCR数据生成** (Appendix A.4):
   - **SynthDoG**: 使用COCO图像作为背景，合成英文(41种字体)和中文(11种字体)文本
   - **PDF数据**: 使用PyMuPDF渲染Common Crawl的PDF，提取文本和边界框
   - **HTML数据**: 使用Puppeteer渲染网页，提取文本和边界框

2. **Grounding数据处理**:
   - **GRIT数据清洗**: 发现递归嵌套的grounding box标注，使用贪心算法清洗，确保每张图像包含最多box且无递归嵌套

3. **交错图文数据**:
   - 将相同任务的数据打包成长度为2048的序列
   - 支持多图像输入

**训练配置变化**:
- **图像分辨率**: 224×224 → **448×448** ⚡
- **ViT序列长度**: 256 → 1024
- **LLM序列长度**: 512 → 2048
- **Batch size**: 4,096
- **学习率**: 
  - Peak: 5e⁻⁵ (降低)
  - Min: 1e⁻⁵
- **训练步数**: 19,000步
- **参数冻结**: **解冻LLM**，端到端训练
- **模型并行**: 2路并行 (ViT和LLM)

**数据格式示例** (Box B.1):
```
# Captioning
<img>cc3m/01581435.jpg</img>Generate the caption in English: 
the beautiful flowers for design.<eos>

# Grounded Captioning
<img>coyo700m/1.jpg</img>Generate the caption in English with grounding: 
Beautiful shot of <ref>bees</ref><box>(661,612),(833,812)</box><box>(120,555),(265,770)</box> 
gathering nectars from <ref>an apricot flower</ref><box>(224,13),(399,313)</box><eos>

# OCR
<img>synthdog/1.jpg</img>OCR with grounding: 
<ref>It is managed</ref><quad>(568,121),(625,131),(624,182),(567,172)</quad>...<eos>
```

### 4.4 Stage 3: 监督微调 (Supervised Fine-tuning)

**训练目标**: 提升**指令跟随能力**和**多轮对话能力**

**数据规模**: 350K样本

**数据来源**:
1. **LLM自我指令生成**的多模态对话数据
2. **人工标注**的对话数据
3. **策略拼接**构建的对话数据
4. **多模态+纯文本**混合对话数据(保持通用对话能力)

**关键能力注入**:
- **多图像理解**: 图像前添加"Picture id:"前缀
- **定位能力**: grounding和多图像理解的泛化
- **多轮对话**: ChatML格式

**ChatML格式** (Appendix B.2):
```
<im_start>user
Picture 1: <img>vg/VG_100K_2/649.jpg</img>What is the sign in the picture?<im_end>
<im_start>assistant
The sign is a road closure with an orange rhombus.<im_end>
<im_start>user
How is the weather in the picture?<im_end>
<im_start>assistant
The shape of the road closure sign is an orange rhombus.<im_end>
```

**Loss计算**: 只对**assistant的回复**和**特殊token**计算loss，不对role名称和问题计算loss

**训练配置**:
- **图像分辨率**: 448×448
- **Batch size**: 128
- **学习率**: 
  - Peak: 1e⁻⁵
  - Min: 1e⁻⁶
- **训练步数**: 8,000步
- **Warm-up**: 3,000步
- **参数冻结**: **冻结ViT**，训练LLM和Adapter
- **ViT学习率衰减**: 0 (完全冻结)

---

## 五、实验结果

### 5.1 图像描述和通用VQA (表4)

**Image Captioning**:
| 模型 | Nocaps (0-shot) | Flickr30K (0-shot) |
|------|-----------------|---------------------|
| Flamingo-80B | - | 67.2 |
| BLIP-2 (Vicuna-13B) | 103.9 | 71.6 |
| InstructBLIP (Vicuna-13B) | 121.9 | 82.8 |
| **Qwen-VL** | **121.4** | **85.8** ⚡ |
| Qwen-VL-Chat | 120.2 | 81.0 |
| SOTA (PALI-17B) | 127.0 | 84.5 |

**关键发现**: Qwen-VL在Flickr30K上达到**85.8 CIDEr**，超越Flamingo-80B (67.2)，甚至超越SOTA

**General VQA**:
| 模型 | VQAv2 | OKVQA | GQA | SciQA-Img | VizWiz |
|------|-------|-------|-----|-----------|--------|
| BLIP-2 (Vicuna-13B) | 65.0 | 45.9 | 32.3 | 61.0 | 19.6 |
| InstructBLIP (Vicuna-13B) | - | - | 49.5 | 63.1 | 33.4 |
| Shikra (Vicuna-13B) | 77.36 | 47.16 | - | - | - |
| **Qwen-VL** | **79.5** | **58.6** | **59.3** | **67.1** | **35.2** |
| Qwen-VL-Chat | 78.2 | 56.6 | 57.5 | 68.2 | 38.9 |

**性能亮点**:
- VQAv2: 79.5% (大幅超越BLIP-2的65.0%)
- OKVQA: 58.6% (需要外部知识)
- GQA: 59.3% (场景理解和推理)

### 5.2 文本导向VQA (表5)

| 模型 | TextVQA | DocVQA | ChartQA | AI2D | OCR-VQA |
|------|---------|--------|---------|------|---------|
| BLIP-2 (Vicuna-13B) | 42.4 | - | - | - | - |
| InstructBLIP (Vicuna-13B) | 50.7 | - | - | - | - |
| mPLUG-DocOwl (LLaMA-7B) | 52.6 | 62.2 | 57.4 | - | - |
| Pix2Struct-Large (1.3B) | - | 76.6 | 58.6 | 42.1 | 71.3 |
| **Qwen-VL** | **63.8** | **65.1** | **65.7** | **62.3** | **75.7** |
| Qwen-VL-Chat | 61.5 | 62.6 | 66.3 | 57.7 | 70.5 |

**关键亮点**:
- TextVQA: **63.8%** vs InstructBLIP的50.7% (+13.1%)
- 在所有文本导向任务上全面领先开源模型

### 5.3 视觉定位 (表6)

**RefCOCO系列**:
| 模型 | RefCOCO val | RefCOCO testA | RefCOCO testB |
|------|-------------|---------------|---------------|
| Shikra-13B | 87.83 | 91.11 | 81.81 |
| **Qwen-VL-7B** | **89.36** | **92.26** | **85.34** |
| Qwen-VL-Chat | 88.55 | 92.27 | 84.51 |
| G-DINO-L (SOTA) | 90.56 | 93.19 | 88.24 |

**GRIT (RefExp)**:
- Qwen-VL: **78.22%**
- Shikra-13B: 69.03%
- 提升9.2个百分点

**性能分析**:
- 在所有grounding任务上**显著超越**同等规模的generalist模型
- 接近专门的grounding SOTA模型(G-DINO-L)

### 5.4 Few-shot Learning (图4)

**测试基准**: OKVQA, VizWiz, TextVQA, Flickr30k

**对比模型**: Flamingo-9B/80B, OpenFlamingo-9B, IDEFICS-9B/80B

**关键发现**:
- Qwen-VL (9.6B) 的few-shot性能**超越Flamingo-9B**
- **接近Flamingo-80B**的性能
- 验证了模型的**in-context learning能力**

### 5.5 指令跟随能力 (表7)

| 模型 | TouchStone-En | TouchStone-Cn | SEED-Bench-All | MME-Perception | MME-Cognition |
|------|---------------|---------------|----------------|----------------|---------------|
| MiniGPT4 | 531.7 | - | 42.8 | 581.67 | 144.29 |
| InstructBLIP | 552.4 | - | 53.4 | 1212.82 | 291.79 |
| LLaVA | 602.7 | - | 33.5 | 502.82 | 214.64 |
| mPLUG-Owl | 605.4 | - | 34.0 | 967.34 | 276.07 |
| **Qwen-VL-Chat** | **645.2** | **401.2** | **58.2** | **1487.58** | **360.71** |

**性能优势**:
- TouchStone英文: **645.2** (第2名mPLUG-Owl: 605.4)
- TouchStone中文: **401.2** (大幅领先)
- MME-Perception: **1487.58** (感知能力)
- MME-Cognition: **360.71** (认知能力)

**细分能力优势** (TouchStone):
- 文本识别(Text Recognition)
- 图表分析(Chart Analysis)
- 定位能力(Localization)

---

## 六、关键技术创新点总结

### 6.1 架构创新

1. **Position-aware Adapter**
   - 2D绝对位置编码注入cross-attention
   - 缓解特征压缩过程中的位置信息损失
   - 256个learnable queries达到性能与效率平衡

2. **边界框字符串化表示**
   - 不需要额外位置词汇表
   - 直接通过LLM tokenizer处理
   - 归一化到[0, 1000)范围

### 6.2 训练策略创新

1. **三阶段渐进式训练**
   - Stage 1: 冻结LLM，建立基础对齐
   - Stage 2: 解冻LLM，提升分辨率，引入细粒度任务
   - Stage 3: 冻结ViT，增强对话能力

2. **分辨率渐进提升**
   - 224×224 → 448×448
   - 减少信息损失，提升细粒度理解

3. **纯文本数据混合**
   - 在Stage 2和Stage 3引入纯文本数据
   - 防止catastrophic forgetting
   - 纯文本能力不降反升(表11)

### 6.3 数据创新

1. **大规模OCR数据合成**
   - SynthDoG: 24.8M样本
   - PDF/HTML渲染 + 自动标注
   - 支持英文和中文

2. **Grounded Caption数据**
   - 8.7M image-caption-box三元组
   - 同时训练grounding和caption能力

3. **严格的数据清洗**
   - 50亿 → 14亿 (保留率28%)
   - 多语言、多任务、高质量

### 6.4 能力创新

1. **多图像交互**: Picture id机制
2. **细粒度定位**: 边界框输入输出
3. **中英双语**: 22.7%中文数据
4. **Few-shot能力**: 接近80B模型
5. **OCR能力**: TextVQA 63.8%

---

## 七、消融实验与分析

### 7.1 Learnable Queries数量 (图7, Appendix E.2)

**实验设置**: 64, 144, 256, 400

**结果**:
- **初始loss**: queries越少，初始loss越低(因为更简单)
- **收敛性能**: 256最优
  - 64: 信息损失严重，收敛后性能差
  - 400: 收敛困难，训练不稳定
  - 256: 性能与效率最佳平衡

### 7.2 Window Attention vs Global Attention (图8, 表10, Appendix E.3)

**测试配置**:
- 448×448 + Window Attention: 9s/iter
- 448×448 + Global Attention: 10s/iter
- 896×896 + Window Attention: 25s/iter
- 896×896 + Global Attention: 60s/iter

**结果**:
- Window Attention虽然快，但**loss显著更高**
- **最终选择**: 448×448 + Global Attention
  - 训练速度可接受(10s/iter)
  - 收敛性能最优
  - 896×896过慢(60s/iter)，不采用

## 7.3 纯文本能力

为了研究多模态训练对纯文本能力的影响,论文展示了Qwen-VL与开源LLM在纯文本任务上的性能对比(Table 11)。

**初始化说明**:
- Qwen-VL使用Qwen-7B的**中间检查点**(intermediate checkpoint)作为LLM初始化
- 之所以没有使用Qwen-7B的最终发布版本,是因为Qwen-VL和Qwen-7B在非常相似的时期开发

**纯文本基准测试结果**:

| 模型 | MMLU | CMMLU | C-Eval |
|------|------|-------|--------|
| LLaMA-7B | 35.1 | 26.8 | - |
| LLaMA2-7B | 46.8 | 31.8 | 32.5 |
| Baichuan-7B | 42.3 | 44.4 | 42.8 |
| Baichuan2-7B | 54.2 | 57.1 | 54.0 |
| ChatGLM2-6B | 47.9 | 48.8 | 51.7 |
| InternLM-7B | 51.0 | 51.8 | 52.8 |
| **Qwen-7B (最终版)** | 58.2 | 62.2 | 63.5 |
| **Qwen-7B (中间版,用作Qwen-VL初始化)** | 49.9 | - | 48.5 |
| **Qwen-VL** | **50.7** | **49.5** | **51.1** |

**关键发现**:

1. **防止灾难性遗忘**: 在多任务训练(Stage 2)和监督微调(Stage 3)阶段,Qwen-VL不仅使用视觉-语言数据,还混入了**纯文本数据**进行训练,目的是防止文本理解能力的灾难性遗忘(catastrophic forgetting)

2. **能力保持与提升**: 对比结果表明,Qwen-VL在纯文本能力上:
   - **没有退化**: 相比初始化的中间检查点,各项指标均有提升
   - MMLU: 49.9 → 50.7 (+0.8)
   - CMMLU: → 49.5 (新增)
   - C-Eval: 48.5 → 51.1 (+2.6)

3. **与纯文本LLM相当**: 由于Qwen-7B提供了良好的LLM初始化,Qwen-VL在纯文本任务上的表现与许多纯文本LLM相当,甚至超过了LLaMA2-7B、Baichuan-7B等模型

**混合训练策略的有效性**:
- 通过在视觉-语言训练中混入纯文本数据,Qwen-VL成功保留了LLM的原生文本理解能力
- 这种策略证明了多模态模型可以在获得视觉能力的同时,不牺牲文本能力

---

## 八、数据集详细信息

### 8.1 图像-文本对数据清洗

论文使用的网络爬取图像-文本对数据集包括:
- LAION-en, LAION-zh (Schuhmann et al., 2022a)
- LAION-COCO (Schuhmann et al., 2022b)
- DataComp (Gadre et al., 2023)
- Coyo (Byeon et al., 2022)

**清洗步骤**:
1. 移除图像宽高比过大的样本
2. 移除图像尺寸过小的样本
3. 移除CLIP分数过低的样本(数据集特定阈值)
4. 移除包含非英文或非中文字符的文本
5. 移除包含emoji字符的文本
6. 移除文本长度过短或过长的样本
7. 清理文本中的HTML标签部分
8. 清理具有特定不规则模式的文本

对于学术caption数据集(CC12M, SBU等),还移除了包含特殊标签的样本,并选择最长的文本作为标注。

### 8.2 OCR数据生成

**合成OCR数据**:
- 使用**Synthdog** (Kim et al., 2022)生成
- 背景图像: COCO train2017 和 unlabeled2017作为自然场景背景
- 字体选择: 41种英文字体 + 11种中文字体
- 坐标标注: 生成文本的四边形坐标作为训练标签

**PDF数据处理**(使用PyMuPDF):
1. 提取每页的所有文本及其边界框
2. 渲染每页并保存为图像文件
3. 移除过小的图像
4. 移除字符数量过多或过少的图像
5. 移除包含"Latin Extended-A/B"块中Unicode字符的图像
6. 移除包含"Private Use Area (PUA)"块中Unicode字符的图像

**HTML网页处理**(使用Puppeteer):
流程与PDF类似,但使用Puppeteer替代PyMuPDF来渲染HTML页面并获取真实标注。

---

## 九、训练收敛性分析

### 9.1 预训练阶段收敛曲线(Figure 6)

**训练配置**:
- 混合精度: BFloat16
- Batch size: 30720
- 学习率: 2e⁻⁴
- 训练轮数: 1 epoch(所有图像只训练一次)

**关键观察**:
1. **训练损失**: 随着训练图像数量增加,损失稳定下降
2. **Zero-shot Caption能力**(Flickr30K): 从62 CIDEr提升至76 CIDEr
3. **Zero-shot VQA能力**(VQAv2): 从48%提升至约56%,尽管Stage 1没有添加VQA数据

这表明大规模图像-文本对的预训练为模型建立了良好的视觉-语言对齐基础。

### 9.2 Learnable Queries数量消融实验(Figure 7)

**实验设置**: 使用ViT-L/14,输入分辨率224×224,输出序列长度256

**测试的Query数量**: 64, 144, 256, 400

**结果**:
- **训练初期**(前50步): Query越少,初始损失越低
- **收敛阶段**(1k-5k步): Query过多或过少都会导致收敛变慢
- **最终选择**: 256个queries
  - 原因: Stage 2使用448×448分辨率,ViT输出序列长度为1024,太少的queries会导致信息丢失

### 9.3 Window Attention vs Global Attention对比(Table 10 & Figure 8)

**训练速度对比**:

| 输入分辨率 & Attention类型 | 训练速度 |
|--------------------------|---------|
| 448×448, Global Attention | 10s/iter |
| 448×448, Window Attention | 9s/iter |
| 896×896, Global Attention | 60s/iter |
| 896×896, Window Attention | 25s/iter |

**损失对比**:
- 使用Window Attention时,模型损失显著更高
- 448×448分辨率下,两种方案训练速度相近

**最终决策**: 
- Qwen-VL在Vision Transformer中使用**Global Attention**
- 不使用896×896分辨率,因为训练速度过慢(即使使用Window Attention也需要2.5倍时间)

---

## 十、总结与未来工作

### 10.1 Qwen-VL的核心贡献

1. **全面的视觉-语言能力**: 集成了图像描述、视觉问答、OCR、文档理解和视觉定位能力
2. **SOTA性能**: 在多个基准测试上取得同等规模通用模型的最佳性能
3. **多语言支持**: 天然支持英文、中文和多语言指令
4. **细粒度理解**: 通过高分辨率输入(448×448)和细粒度语料,实现出色的定位和文本识别能力
5. **多图像对话**: 支持任意交错的图像-文本数据作为输入
6. **开源贡献**: 所有模型公开发布,促进多模态研究发展

### 10.2 未来发展方向

论文提出了三个主要的未来研究方向:

1. **模态扩展**:
   - 将Qwen-VL与更多模态集成,如语音和视频
   - 构建真正的多模态统一模型

2. **规模提升**:
   - 通过扩大模型规模、训练数据和更高分辨率
   - 使其能够处理更复杂和更精细的多模态数据关系

3. **生成能力增强**:
   - 扩展Qwen-VL的多模态生成能力
   - 特别是生成高保真图像和流畅语音

---

## 十一、技术亮点总结

### 11.1 架构设计亮点

1. **Position-aware Adapter**:
   - 使用2D绝对位置编码的cross-attention机制
   - 在压缩视觉特征时保留位置信息,对细粒度理解至关重要

2. **Bounding Box字符串化表示**:
   - 将坐标归一化到[0, 1000)范围
   - 使用字符串格式表示,无需额外的位置词汇表
   - 通过特殊token(<box>, </box>, <ref>, </ref>)标识定位信息

3. **简洁高效的架构**:
   - 总参数9.6B: ViT-bigG(1.9B) + Adapter(0.08B) + Qwen-7B(7.7B)
   - 256个learnable queries将视觉序列压缩到固定长度

### 11.2 训练策略亮点

1. **三阶段渐进式训练**:
   - Stage 1: 大规模图像-文本对预训练(1.4B样本)
   - Stage 2: 多任务预训练,引入细粒度标注
   - Stage 3: 指令微调(350K对话数据)

2. **分辨率逐步提升**:
   - Stage 1: 224×224(冻结LLM)
   - Stage 2: 448×448(解冻LLM)
   - 逐步提升避免训练不稳定

3. **纯文本数据混合训练**:
   - 在Stage 2和Stage 3混入纯文本数据
   - 成功防止灾难性遗忘,保持LLM原生能力

### 11.3 数据策略亮点

1. **大规模数据清洗**: 从5B原始数据清洗至1.4B高质量样本(保留率28%)
2. **合成OCR数据**: 24.8M样本,覆盖英文和中文
3. **细粒度定位数据**: GRIT、Visual Genome、RefCOCO系列等
4. **ChatML格式**: 使用标准对话格式,支持多轮对话

---

## 十二、Qwen-VL vs 竞品对比

### 12.1 与其他开源模型对比

**优势领域**:
1. **OCR和文档理解**: 在TextVQA、DocVQA等任务上显著超越BLIP-2、InstructBLIP
2. **细粒度定位**: 在RefCOCO系列上接近专用SOTA模型
3. **中英双语能力**: TouchStone中文评分401.2,远超其他模型
4. **Few-shot学习**: 性能接近参数量10倍的Flamingo-80B

**与Specialist SOTA的差距**:
- 在某些任务上仍有差距(如Caption的CIDEr、Grounding的准确率)
- 但Qwen-VL作为通用模型,在广度上具有显著优势

### 12.2 技术创新点对比

| 模型 | 视觉编码器 | LLM基座 | 定位能力 | OCR能力 | 多语言 |
|------|----------|---------|---------|---------|--------|
| BLIP-2 | EVA-CLIP | Vicuna-13B | ✗ | 弱 | 英文为主 |
| InstructBLIP | EVA-CLIP | Vicuna-13B | ✗ | 弱 | 英文为主 |
| Kosmos-2 | CLIP | Decoder | ✓ | 中等 | 多语言 |
| Shikra | CLIP | Vicuna-13B | ✓ | 弱 | 英文为主 |
| **Qwen-VL** | ViT-bigG | Qwen-7B | ✓ | **强** | **中英双语** |

---

## 结论

Qwen-VL系列模型通过精心设计的架构、渐进式训练策略和大规模多任务数据,实现了在同等规模通用模型中的领先性能。其在细粒度视觉理解(OCR、定位)、多语言支持和对话能力方面的优势,使其成为视觉-语言研究和应用的重要基础模型。

**论文页数**: 24页  
**发表信息**: arXiv:2308.12966v3 [cs.CV] 13 Oct 2023  
**作者机构**: Alibaba Group  
**代码和模型**: https://github.com/QwenLM/Qwen-VL




