

## 👤 User

分析一下qween vl的主要架构,训练数据,训练loss,评估的任务指标,另外结合这个论文的项目代码给出分析
# Qwen-VL 详细技术分析报告

## 一、模型架构详解

### 1.1 整体架构设计

Qwen-VL采用**三组件架构**,总参数量为**9.6B**:

| 组件 | 参数量 | 初始化来源 |
|------|--------|-----------|
| Visual Encoder (ViT) | 1.9B | OpenCLIP ViT-bigG |
| VL Adapter | 0.08B | 随机初始化 |
| Large Language Model | 7.7B | Qwen-7B |

### 1.2 核心组件分析

#### **Visual Encoder (视觉编码器)**
- 架构:Vision Transformer (ViT)
- 初始化:OpenCLIP的ViT-bigG预训练权重
- 输入处理:
  - Stage 1:224×224分辨率
  - Stage 2-3:448×448分辨率(提升细粒度理解)
  - Patch stride:14
  - 输出序列长度:(448/14)² = 1024

#### **Position-aware Vision-Language Adapter (位置感知适配器)**
这是Qwen-VL的**关键创新**:

```
输入:ViT输出的1024长度特征序列
核心机制:单层Cross-Attention
- Query:256个可学习的embedding向量
- Key/Value:来自ViT的图像特征
- 位置编码:2D绝对位置编码融入query-key对
输出:压缩后的256长度特征序列
```

**设计亮点**:
1. **特征压缩**:1024→256,提升效率
2. **位置保留**:通过2D位置编码保留空间信息,支持细粒度理解
3. **消融实验**:论文测试了64/144/256/400个查询向量,256为最优

#### **Large Language Model**
- 基座:Qwen-7B(7.7B参数)
- 特殊token扩展:
  - 图像标记:`<img>` `</img>`
  - 边界框:`<box>` `</box>`
  - 引用标记:`<ref>` `</ref>`
  - 对话标记:`<im_start>` `<im_end>`

### 1.3 输入输出接口

#### **边界框表示创新**
```
归一化坐标:[0, 1000)范围
格式:
- 输入/输出:"<box>(Xtopleft,Ytopleft),(Xbottomright,Ybottomright)</box>"
- 关联文本:"<ref>描述文本</ref><box>...</box>"
```

**优势**:
- 无需额外位置词汇表
- 直接tokenization为文本
- 支持中英文grounding

---

## 二、训练数据详解

### 2.1 三阶段训练数据

#### **Stage 1: 预训练(1.4B样本)**

数据清洗策略:
1. 过滤过大宽高比图像
2. 移除过小图像
3. 数据集特定的CLIP分数过滤
4. 移除非英文/中文字符
5. 清理HTML标签和不规则模式

| 数据源 | 原始量 | 清洗后 | 保留率 | 语言 |
|--------|--------|--------|--------|------|
| LAION-en | 2B | 280M | 14% | 英文 |
| LAION-COCO | 600M | 300M | 50% | 英文 |
| DataComp | 1.4B | 300M | 21% | 英文 |
| Coyo | 700M | 200M | 28% | 英文 |
| CC12M | 12M | 8M | 66% | 英文 |
| CC3M | 3M | 3M | 100% | 英文 |
| SBU | 1M | 0.8M | 80% | 英文 |
| COCO Caption | 0.6M | 0.6M | 100% | 英文 |
| LAION-zh | 108M | 105M | 97% | 中文 |
| In-house Data | 220M | 220M | 100% | 中文 |
| **总计** | **5B** | **1.4B** | **28%** | 77.3%英文/22.7%中文 |

#### **Stage 2: 多任务预训练**

7大任务并行训练:

| 任务类型 | 样本数 | 数据集 |
|---------|--------|-------|
| **Captioning** | 19.7M | LAION-en/zh, DataComp, Coyo, CC12M/3M, SBU, COCO |
| **VQA** | 3.6M | GQA, VGQA, VQAv2, DVQA, OCR-VQA, DocVQA, TextVQA, ChartQA, AI2D |
| **Grounding** | 3.5M | GRIT (生成名词/短语grounded captions) |
| **Ref Grounding** | 8.7M | GRIT, Visual Genome, RefCOCO系列 |
| **Grounded Cap** | 8.7M | 同Ref Grounding |
| **OCR** | 24.8M | **SynthDoG-en/zh**, Common Crawl PDF/HTML |
| **纯文本** | 7.8M | In-house数据(防止灾难性遗忘) |

**OCR数据生成**:
- 背景:COCO train2017/unlabeled2017
- 字体:41种英文 + 11种中文
- 工具:SynthDoG(Kim et al., 2022)
- 坐标标注:四边形坐标

#### **Stage 3: 指令微调(350K样本)**

数据构成:
1. **LLM自指令生成**:caption/对话数据
2. **人工标注**:定位+多图理解对话
3. **模型生成**:结合策略拼接
4. **纯文本对话**:保持通用对话能力

### 2.2 数据格式

**Multi-task预训练格式**:
```
Image Captioning:
<img>path</img>Generate the caption in English: [caption]<eos>

VQA:
<img>path</img> [question] Answer: [answer]<eos>

Caption with Grounding:
<img>path</img>Generate the caption in English with grounding:
<ref>object</ref><box>(x1,y1),(x2,y2)</box>...<eos>

OCR:
<img>path</img>OCR with grounding: 
<ref>text</ref><quad>(x1,y1),(x2,y2),(x3,y3),(x4,y4)</quad>...<eos>
```

**SFT对话格式(ChatML)**:
```
<im_start>user
Picture 1: <img>path</img>问题内容<im_end>
<im_start>assistant
回答内容<im_end>
```

---

## 三、训练Loss与优化策略

### 3.1 训练目标

**统一Loss**:标准语言建模交叉熵损失
```
Loss = CrossEntropy(text_tokens)
```

**关键设计**:
- 仅对文本token计算loss
- 图像特征序列不参与loss计算
- SFT阶段:仅监督回答和特殊token,不监督角色名和问题

### 3.2 训练超参数

#### **Stage 1: 预训练**
```python
Optimizer: AdamW (β1=0.9, β2=0.98, eps=1e-6)
Learning Rate: 
  - Peak: 2e-4
  - Min: 1e-6
  - Schedule: Cosine decay
  - Warmup: 500 steps
  - ViT layer-wise decay: 0.95

Batch Size: 30720
Training Steps: 50,000 (~1.5B样本)
Weight Decay: 0.05
Gradient Clip: 1.0
Precision: BFloat16
Trainable: ViT + VL Adapter (LLM冻结)
```

#### **Stage 2: 多任务预训练**
```python
Learning Rate:
  - Peak: 5e-5
  - Min: 1e-5
  - Warmup: 400 steps

Batch Size: 4096
Training Steps: 19,000
Model Parallelism: 2 (ViT + LLM)
Trainable: 全模型解冻
Resolution: 448×448
```

#### **Stage 3: SFT**
```python
Learning Rate:
  - Peak: 1e-5
  - Min: 1e-6
  - Warmup: 3,000 steps

Batch Size: 128
Training Steps: 8,000
Trainable: LLM + VL Adapter (ViT冻结)
Data: 350K对话样本
```

### 3.3 收敛曲线分析

论文Figure 6显示:
- **Loss稳定下降**:从3.0降至1.8
- **Caption能力提升**:Flickr CIDEr从62→76
- **Zero-shot VQA涌现**:VQAv2从48%→56%(无VQA训练数据!)

---

## 四、评估任务与指标体系

### 4.1 标准Benchmark评估

#### **图像描述(Image Captioning)**

| Benchmark | Metric | Qwen-VL | SOTA对比 |
|-----------|--------|---------|----------|
| Nocaps (val) | CIDEr↑ | 121.4 | InstructBLIP:121.9 |
| **Flickr30K** | CIDEr↑ | **85.8** | Flamingo-80B:67.2 |

#### **通用VQA**

| Benchmark | Metric | Qwen-VL | Qwen-VL-Chat | Specialist SOTA |
|-----------|--------|---------|--------------|-----------------|
| **VQAv2** | Acc↑ | **79.5** | 78.2 | PALI-X-55B:86.1 |
| **OKVQA** | Acc↑ | **58.6** | 56.6 | PALI-X-55B:66.1 |
| **GQA** | EM↑ | **59.3** | 57.5 | CFR:72.1 |
| ScienceQA-Img | Acc↑ | 67.1 | **68.2** | LLaVA+GPT4:92.5 |
| VizWiz | VQA Score↑ | 35.2 | **38.9** | PALI-X-55B:70.9 |

#### **文本导向VQA**

| Benchmark | Metric | Qwen-VL | 对比模型最佳 |
|-----------|--------|---------|-------------|
| **TextVQA** | VQA Score↑ | **63.8** | InstructBLIP:50.7 |
| DocVQA | ANLS↑ | 65.1 | Pix2Struct-Large:76.6 |
| **ChartQA** | Relaxed EM↑ | **65.7** | Pix2Struct-Large:58.6 |
| **AI2D** | EM↑ | **62.3** | Pix2Struct-Large:42.1 |
| **OCR-VQA** | EM↑ | **75.7** | Pix2Struct-Large:71.3 |

**关键发现**:Qwen-VL在448分辨率下超越1024分辨率的Pix2Struct

#### **Referring Expression Comprehension**

在RefCOCO/RefCOCO+/RefCOCOg/GRIT上**全面SOTA**(generalist模型):

| Dataset | Metric | Qwen-VL-7B | Shikra-13B |
|---------|--------|-----------|------------|
| RefCOCO val | Acc↑ | **89.36** | 87.83 |
| RefCOCO+ testB | Acc↑ | **77.21** | 74.41 |
| RefCOCOg test | Acc↑ | **85.48** | 83.16 |
| GRIT refexp | Acc↑ | **78.22** | 69.03 |

### 4.2 Few-shot学习

在OKVQA/VizWiz/TextVQA/Flickr30K上:
- **Qwen-VL-9B** ≈ **Flamingo-80B** (8.9x参数优势)
- 4-shot性能显著提升

### 4.3 指令遵循评估

#### **TouchStone**
| Model | 英文得分 | 中文得分 |
|-------|---------|---------|
| mPLUG-Owl | 605.4 | - |
| Qwen-VL-Chat | **645.2** | **401.2** |
| Qwen-VL-Chat-1.1 | **711.6** | **481.7** |

#### **SEED-Bench**
| Model | Image | Video | All |
|-------|-------|-------|-----|
| InstructBLIP | 58.8 | 38.1 | 53.4 |
| Qwen-VL | **62.3** | **39.1** | **56.3** |
| Qwen-VL-Chat | **65.4** | 37.8 | **58.2** |

#### **MME**
| Model | Perception | Cognition |
|-------|-----------|-----------|
| InstructBLIP | 1212.82 | 291.79 |
| Qwen-VL-Chat | **1487.58** | **360.71** |

---

## 五、代码实现分析

### 5.1 核心代码结构

根据finetune.py:312-315:
```python
if training_args.fix_vit and hasattr(model,'transformer') and hasattr(model.transformer,'visual'):
    model.transformer.visual.requires_grad_(False)
    if hasattr(model.transformer.visual,'attn_pool'):
        model.transformer.visual.attn_pool.requires_grad_(True)
```

**训练策略**:
- **Stage 1**: ViT + VL Adapter可训练,LLM冻结
- **Stage 2**: 全部解冻
- **Stage 3(SFT)**: LLM + VL Adapter可训练,ViT冻结(**VL Adapter中的attn_pool除外**)

### 5.2 数据处理(finetune.py:124-175)

**预处理流程**:
```python
1. 角色映射:
   roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}

2. Token ID使用:
   - im_start = tokenizer.im_start_id
   - im_end = tokenizer.im_end_id
   
3. Loss掩码策略:
   - System prompt: IGNORE_TOKEN_ID
   - User输入: IGNORE_TOKEN_ID
   - Assistant回答: 正常token计算loss
   - 特殊token(<im_end>等): 计算loss
```

### 5.3 LoRA/Q-LoRA支持

#### **LoRA配置**(finetune.py:54-63)
```python
lora_r: 64
lora_alpha: 16
lora_dropout: 0.05
lora_target_modules: ["c_attn", "attn.c_proj", "w1", "w2"]
```

#### **显存对比**(单A100-80G,batch_size=1):

| 方法 | 384 tokens | 2048 tokens |
|------|-----------|-------------|
| LoRA (Base) | 37.1G / 2.3s | 38.7G / 6.1s |
| LoRA (Chat) | 23.3G / 2.2s | 27.3G / 5.9s |
| Q-LoRA | 17.0G / 4.2s | 19.3G / 7.9s |

**关键差异**:
- **LoRA (Base)**:需训练embedding + output layer(新增special tokens)
- **LoRA (Chat)**:无需训练这些层(已包含ChatML tokens)
- **Q-LoRA**:4-bit量化,最省显存但速度较慢

### 5.4 评估脚本

根据README:
- 所有benchmark评估脚本位于`eval_mm/`
- 支持的评估:
  - Image Captioning: eval_mm/evaluate_caption.py
  - VQA: eval_mm/evaluate_vqa.py
  - Grounding: eval_mm/evaluate_grounding.py
  - Multiple Choice: eval_mm/evaluate_multiple_choice.py
  - MME: eval_mm/mme/eval.py
  - SEED-Bench: eval_mm/seed_bench/eval.py

---

## 六、核心创新点总结

### 6.1 架构创新
1. **Position-aware Adapter**:2D位置编码 + Cross-Attention压缩
2. **统一接口**:边界框作为文本序列,无需专用词汇表
3. **多分辨率训练**:224→448提升细粒度能力

### 6.2 训练创新
1. **三阶段训练**:预训练→多任务→SFT,逐步增强能力
2. **OCR数据合成**:SynthDoG生成24.8M高质量OCR样本
3. **防灾难性遗忘**:Stage 2/3混合纯文本数据

### 6.3 能力突破
1. **首个支持中文grounding的通用模型**
2. **Zero-shot grounding**:未见中文grounding数据也能泛化
3. **多图交错对话**:支持图像比较和多图理解
4. **细粒度识别**:448分辨率超越1024分辨率模型

---

## 七、性能Benchmark对比

### 综合能力雷达图(论文Figure 1)

在9大任务上与同规模generalist模型对比,Qwen-VL在**7/9任务上领先**:
- ✅ 领先:VQAv2, Flickr30K, RefCOCO, OKVQA, GQA, TextVQA, VizWiz
- ≈ 持平:Nocaps
- ❌ 落后:AI2D(但仍是generalist模型最佳)

### 与专用SOTA对比

| 任务 | Qwen-VL | Specialist SOTA | 差距 |
|------|---------|-----------------|------|
| Caption | 85.8 | 127.0 (PALI-17B) | -41.2 |
| VQAv2 | 79.5 | 86.1 (PALI-X-55B) | -6.6 |
| Grounding | 89.36 | 92.64 (UNINEXT-H) | -3.28 |
| OCR-VQA | 75.7 | 75.0 (PALI-X-55B) | **+0.7** |

**结论**:在OCR任务上已**超越专用模型**

---

## 八、代码库功能完整性

### 8.1 支持的功能

✅ **推理**:
- 🤗 Transformers集成
- 🤖 ModelScope集成
- 量化推理(Int4)
- Web UI Demo(web_demo_mm.py)
- OpenAI API兼容(openai_api.py)

✅ **微调**:
- Full-parameter finetuning
- LoRA (finetune/finetune_lora_ds.sh)
- Q-LoRA (finetune/finetune_qlora_ds.sh)
- DeepSpeed ZeRO-2/3支持

✅ **评估**:
- 完整benchmark评估脚本(eval_mm/)
- TouchStone评估(touchstone/)
- MME/SEED-Bench评估

### 8.2 量化性能(Int4)

| Metric | BF16 | Int4 | 差异 |
|--------|------|------|------|
| TouchStone-EN | 645.2 | 651.4 | +6.2 |
| TouchStone-CN | 401.2 | 386.6 | -14.6 |
| 推理速度(8192 tokens) | 24.32 tok/s | **34.34 tok/s** | +41% |
| 显存(生成8192 tokens) | 28.01GB | **17.23GB** | -38% |

**结论**:Int4几乎无损性能,显著提升效率

---

## 九、局限性与未来方向

### 9.1 当前局限
1. **分辨率上限**:448×448相比GPT-4V的高分辨率仍有差距
2. **专用任务差距**:在某些任务上与专用模型仍有6-40分差距
3. **中文能力**:虽领先开源模型,但部分任务弱于GPT-4V/Gemini

### 9.2 论文提出的未来方向
1. ✨ 多模态扩展:集成语音、视频
2. 📈 规模升级:更大模型、更多数据、更高分辨率
3. 🎨 生成能力:高保真图像生成、流畅语音生成

---

## 十、总结

Qwen-VL是一个**设计精巧、训练充分、能力全面**的开源视觉语言模型:

**核心优势**:
1. 🏆 同规模generalist模型性能领先
2. 🌏 原生中英双语支持
3. 🎯 细粒度理解(grounding + OCR)
4. 💡 架构简洁高效(9.6B参数)
5. 🛠️ 完整工具链(推理/微调/评估)

**适用场景**:
- ✅ 多语言图像理解
- ✅ 文档/图表/OCR任务
- ✅ 细粒度定位(grounding)
- ✅ 多图对话交互
- ✅ 资源受限部署(Int4量化)

**代码质量**:
- 完整可复现的训练流程
- 丰富的微调选项(Full/LoRA/Q-LoRA)
- 详细的评估脚本
- 清晰的数据格式文档

这是一个**生产级**的开源视觉语言模型,适合作为研究基线或商业应用的起点。

---

## 👤 User

既然他是一个多模态模型,能否介绍一下视觉部分怎么处理,文本输入怎么处理,之后又如何得到输出,三个阶段的损失函数分布是什么?

---

## 🤖 Claude

非常好的问题！让我详细解析Qwen-VL的**完整数据流**和**三阶段损失函数**。

---

# Qwen-VL 数据流与损失函数详解

## 一、完整数据流架构

### 1.1 总体流程图

```
┌─────────────┐         ┌──────────────┐
│  Input Image│         │  Input Text  │
│ (448×448×3) │         │  "这是什么?" │
└──────┬──────┘         └──────┬───────┘
       │                       │
       ▼                       ▼
┌─────────────────────────────────────────┐
│         STEP 1: 特征提取阶段              │
└─────────────────────────────────────────┘
       │                       │
       ▼                       ▼
┌──────────────┐         ┌──────────────┐
│   ViT-bigG   │         │  Tokenizer   │
│   Encoder    │         │  (Qwen-7B)   │
│   (1.9B)     │         │              │
└──────┬───────┘         └──────┬───────┘
       │                        │
       │                        ▼
       │                 ┌──────────────┐
       │                 │ Text Tokens  │
       │                 │ [101, 256,..] │
       │                 └──────┬───────┘
       ▼                        │
┌──────────────┐                │
│ Image Patches│                │
│ (32×32×1536) │                │
│ = 1024 tokens│                │
└──────┬───────┘                │
       │                        │
       ▼                        │
┌─────────────────────────────────────────┐
│      STEP 2: 视觉特征压缩阶段             │
└─────────────────────────────────────────┘
       │                        │
       ▼                        │
┌──────────────────────┐        │
│  VL Adapter          │        │
│  (Cross-Attention)   │        │
│  ┌────────────────┐  │        │
│  │ Query: 256个   │  │        │
│  │ 可学习向量      │  │        │
│  └────────────────┘  │        │
│  ┌────────────────┐  │        │
│  │ Key/Value:     │  │        │
│  │ 1024个patch特征│  │        │
│  └────────────────┘  │        │
│  ┌────────────────┐  │        │
│  │ 2D Position    │  │        │
│  │ Encoding       │  │        │
│  └────────────────┘  │        │
└──────┬───────────────┘        │
       │                        │
       ▼                        │
┌──────────────┐                │
│ 压缩后的图像  │                │
│ 特征 (256维)  │                │
└──────┬───────┘                │
       │                        │
       ▼                        ▼
┌─────────────────────────────────────────┐
│      STEP 3: 特征拼接与标记阶段           │
└─────────────────────────────────────────┘
       │                        │
       └────────┬───────────────┘
                ▼
   ┌────────────────────────────┐
   │  Sequence Construction:    │
   │  <img> [256个image tokens] │
   │  </img> [text tokens]      │
   └────────┬───────────────────┘
            │
            ▼
┌─────────────────────────────────────────┐
│      STEP 4: LLM生成阶段                 │
└─────────────────────────────────────────┘
            │
            ▼
   ┌────────────────┐
   │   Qwen-7B LLM  │
   │   (7.7B params)│
   │                │
   │  Transformer   │
   │  Decoder Layers│
   └────────┬───────┘
            │
            ▼
   ┌────────────────┐
   │  Output Logits │
   │  (vocab_size)  │
   └────────┬───────┘
            │
            ▼
   ┌────────────────┐
   │  Generated     │
   │  Text:         │
   │  "这是一只拉布 │
   │   拉多犬。"     │
   └────────────────┘
```

---

## 二、视觉部分详细处理流程

### 2.1 图像预处理

```python
# 伪代码示意
def preprocess_image(image_path):
    """
    输入: PIL Image 或图像路径
    输出: tensor [3, 448, 448]
    """
    # 1. 加载图像
    image = load_image(image_path)
    
    # 2. Resize到固定分辨率
    # Stage 1: 224×224
    # Stage 2/3: 448×448
    image = resize(image, size=(448, 448))
    
    # 3. 归一化 (ImageNet统计量)
    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]
    image = normalize(image, mean, std)
    
    return image  # shape: [3, 448, 448]
```

### 2.2 ViT编码过程

```python
class VisionTransformer:
    """
    输入: [B, 3, 448, 448]
    输出: [B, 1024, 1536]  # 1024个patch, 每个1536维
    """
    def __init__(self):
        self.patch_size = 14
        self.patch_embed = Conv2d(3, 1536, kernel_size=14, stride=14)
        self.pos_embed = Parameter(torch.zeros(1, 1024, 1536))
        self.transformer_blocks = ModuleList([...])  # 48层Transformer
        
    def forward(self, x):
        # x: [B, 3, 448, 448]
        
        # 1. Patch Embedding
        # (448/14) × (448/14) = 32 × 32 = 1024 patches
        x = self.patch_embed(x)  # [B, 1536, 32, 32]
        x = x.flatten(2).transpose(1, 2)  # [B, 1024, 1536]
        
        # 2. 加入位置编码
        x = x + self.pos_embed
        
        # 3. 经过Transformer层
        for block in self.transformer_blocks:
            x = block(x)  # Self-Attention + FFN
            
        return x  # [B, 1024, 1536]
```

**关键细节**:
- **Patch数量**: (448/14)² = 1024个patches
- **特征维度**: 1536 (ViT-bigG的hidden size)
- **无CLS token**: 直接使用所有patch tokens

### 2.3 VL Adapter压缩

```python
class PositionAwareVLAdapter(nn.Module):
    """
    输入: [B, 1024, 1536] (ViT输出)
    输出: [B, 256, 4096]  (LLM输入维度)
    """
    def __init__(self):
        self.num_queries = 256
        # 可学习的query embeddings
        self.query_tokens = Parameter(torch.randn(1, 256, 4096))
        
        # Cross-Attention层
        self.cross_attn = MultiheadAttention(
            embed_dim=4096,
            num_heads=32,
            kdim=1536,  # ViT特征维度
            vdim=1536
        )
        
        # 2D位置编码
        self.position_encoding_2d = build_2d_sincos_position_embedding(
            h=32, w=32, embed_dim=1536
        )
        
    def forward(self, image_features):
        # image_features: [B, 1024, 1536]
        B = image_features.size(0)
        
        # 1. 加入2D位置编码到Key
        pos_enc = self.position_encoding_2d  # [1024, 1536]
        keys = image_features + pos_enc.unsqueeze(0)
        
        # 2. Query: 可学习向量
        queries = self.query_tokens.expand(B, -1, -1)  # [B, 256, 4096]
        
        # 3. Cross-Attention
        # Q: [B, 256, 4096]
        # K, V: [B, 1024, 1536] (自动投影到4096维)
        output, _ = self.cross_attn(
            query=queries.transpose(0, 1),      # [256, B, 4096]
            key=keys.transpose(0, 1),           # [1024, B, 1536]
            value=image_features.transpose(0, 1)  # [1024, B, 1536]
        )
        
        output = output.transpose(0, 1)  # [B, 256, 4096]
        return output
```

**核心机制**:
1. **Query**: 256个可学习向量(类似DETR的object queries)
2. **Key/Value**: 来自ViT的1024个patch特征
3. **位置信息**: 通过2D位置编码注入到Key中
4. **输出**: 256个压缩后的视觉token,维度对齐LLM(4096)

---

## 三、文本部分处理流程

### 3.1 Tokenization

```python
class QwenTokenizer:
    """
    基于Qwen-7B的tokenizer
    词汇表大小: ~151,851
    """
    def __init__(self):
        # 特殊token
        self.img_start_id = 151857  # <img>
        self.img_end_id = 151858    # </img>
        self.im_start_id = 151644   # <im_start>
        self.im_end_id = 151645     # <im_end>
        self.box_start_id = 151859  # <box>
        self.box_end_id = 151860    # </box>
        self.ref_start_id = 151861  # <ref>
        self.ref_end_id = 151862    # </ref>
        
    def encode(self, text):
        """
        输入: "这是什么?"
        输出: [101, 256, 389, 102]
        """
        return self.tokenize(text)
    
    def from_list_format(self, content_list):
        """
        输入: [
            {'image': 'path/to/img.jpg'},
            {'text': '这是什么?'}
        ]
        输出: "<img>path/to/img.jpg</img>这是什么?"
        """
        text = ""
        for item in content_list:
            if 'image' in item:
                text += f"<img>{item['image']}</img>"
            elif 'text' in item:
                text += item['text']
        return text
```

### 3.2 文本Embedding

```python
def text_embedding(text_tokens):
    """
    输入: [101, 256, 389, 102]  # token IDs
    输出: [B, seq_len, 4096]
    """
    # 1. Token Embedding
    token_embeds = embedding_layer(text_tokens)  # [B, seq_len, 4096]
    
    # 2. 位置编码 (RoPE)
    # Qwen使用RoPE (Rotary Position Embedding)
    # 在Attention层中动态计算，此处不显式添加
    
    return token_embeds
```

---

## 四、多模态融合机制

### 4.1 序列构造

```python
def construct_multimodal_sequence(image_features, text_tokens):
    """
    输入:
      - image_features: [B, 256, 4096] (来自VL Adapter)
      - text_tokens: [B, text_len]
    
    输出:
      - input_embeds: [B, 256 + 2 + text_len, 4096]
    """
    B = image_features.size(0)
    
    # 1. 获取特殊token的embedding
    img_start_embed = embedding_layer(torch.tensor([151857]))  # <img>
    img_end_embed = embedding_layer(torch.tensor([151858]))    # </img>
    
    # 2. 获取文本embedding
    text_embeds = embedding_layer(text_tokens)  # [B, text_len, 4096]
    
    # 3. 拼接序列
    # 格式: <img> [256个image tokens] </img> [text tokens]
    input_embeds = torch.cat([
        img_start_embed.expand(B, 1, -1),  # [B, 1, 4096]
        image_features,                     # [B, 256, 4096]
        img_end_embed.expand(B, 1, -1),    # [B, 1, 4096]
        text_embeds                         # [B, text_len, 4096]
    ], dim=1)
    
    # 总长度: 1 + 256 + 1 + text_len = 258 + text_len
    return input_embeds
```

**序列示例**:
```
输入: 图像 + "这是什么?"

Token序列:
┌───────┬──────────────────────┬────────┬───────────────┐
│ <img> │ [256个视觉token]      │ </img> │ 这 是 什 么 ? │
└───────┴──────────────────────┴────────┴───────────────┘
  特殊   │    来自VL Adapter     │  特殊  │   文本tokens   │
  token  │    的图像特征          │  token │                │
         └─────── 258 tokens ─────┘

Embedding维度:
每个位置: [4096维向量]
```

### 4.2 LLM处理

```python
class QwenLMWithVision(nn.Module):
    def forward(self, input_embeds):
        """
        输入: [B, seq_len, 4096]
        输出: [B, seq_len, vocab_size]
        """
        # 1. 经过Transformer Decoder层
        hidden_states = input_embeds
        
        for layer in self.transformer_layers:  # 32层
            # Self-Attention (带RoPE位置编码)
            hidden_states = layer.self_attn(hidden_states)
            # Feed-Forward
            hidden_states = layer.ffn(hidden_states)
        
        # 2. 输出投影
        logits = self.lm_head(hidden_states)  # [B, seq_len, 151851]
        
        return logits
```

**关键点**:
- **图像token参与Attention**: 视觉特征与文本token通过Self-Attention交互
- **位置编码**: RoPE为所有token(包括图像)提供位置信息
- **Causal Mask**: 训练时使用因果掩码,确保自回归生成

---

## 五、三阶段损失函数详解

### 5.1 Stage 1: 预训练阶段

#### **损失函数**

```python
def stage1_loss(logits, labels, image_mask):
    """
    仅对文本token计算损失
    
    参数:
      - logits: [B, seq_len, vocab_size]
      - labels: [B, seq_len]  # ground truth tokens
      - image_mask: [B, seq_len]  # 标记哪些位置是图像token
    
    返回:
      - loss: 标量
    """
    # 1. 构造损失掩码
    # 图像token位置: IGNORE (-100)
    # 文本token位置: 正常计算
    loss_mask = (image_mask == 0)  # False for image, True for text
    
    # 2. 将图像token位置的label设为-100
    masked_labels = labels.clone()
    masked_labels[~loss_mask] = -100
    
    # 3. 计算Cross-Entropy Loss
    # shift prediction: 预测下一个token
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = masked_labels[:, 1:].contiguous()
    
    loss_fct = CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    
    return loss
```

#### **损失计算示例**

```
序列:
┌───────┬────────────────┬────────┬─────┬───┬───┬───┬───┐
│ <img> │ [256 img toks] │ </img> │ 这  │ 是 │ 一 │ 只 │ 狗 │
└───────┴────────────────┴────────┴─────┴───┴───┴───┴───┘

Labels (要预测的下一个token):
┌────────────────┬────────┬─────┬───┬───┬───┬───┬─────┐
│ [256 img toks] │ </img> │ 这  │ 是 │ 一 │ 只 │ 狗 │ EOS │
└────────────────┴────────┴─────┴───┴───┴───┴───┴─────┘

Loss Mask (哪些位置计算损失):
┌────────────────┬────────┬─────┬───┬───┬───┬───┬─────┐
│    IGNORE      │ IGNORE │  ✓  │ ✓ │ ✓ │ ✓ │ ✓ │  ✓  │
└────────────────┴────────┴─────┴───┴───┴───┴───┴─────┘
                           └────── 仅这些位置计算loss ─────┘

实际Loss:
loss = CE(pred_这, label_这) + CE(pred_是, label_是) + ... 
       + CE(pred_EOS, label_EOS)
```

**训练策略**:
- **冻结**: LLM全部参数
- **训练**: ViT + VL Adapter
- **数据**: 1.4B图文对
- **目标**: 让视觉特征对齐到LLM的语义空间

---

### 5.2 Stage 2: 多任务预训练

#### **损失函数 (本质相同,但数据格式多样)**

```python
def stage2_loss(logits, labels, task_type):
    """
    不同任务使用相同的loss函数,但label构造不同
    """
    # 基础loss仍是Cross-Entropy
    loss = CrossEntropyLoss(ignore_index=-100)
    
    # 根据任务类型,label格式不同:
    if task_type == "caption":
        # <img>...</img>Generate the caption: [caption]<eos>
        # 仅对caption部分计算loss
        
    elif task_type == "vqa":
        # <img>...</img>[question] Answer: [answer]<eos>
        # 仅对answer部分计算loss
        
    elif task_type == "grounding":
        # <img>...</img>Generate with grounding: 
        # <ref>text</ref><box>(x,y),(x,y)</box>...<eos>
        # 对整个grounded caption计算loss
        
    elif task_type == "ocr":
        # <img>...</img>OCR: <ref>text</ref><quad>...</quad><eos>
        # 对OCR文本和坐标都计算loss
        
    return loss(logits, labels)
```

#### **各任务的Loss计算**

**Caption任务**:
```
输入序列:
<img>[256 toks]</img>Generate the caption in English:

目标序列 (计算loss):
                                                     A woman playing with her dog.<eos>
                                                     └────── 仅此部分参与loss ──────┘
```

**Grounding任务**:
```
输入序列:
<img>[256 toks]</img>Generate with grounding:

目标序列 (计算loss):
                                            <ref>woman</ref><box>(100,200),(300,400)</box>
                                            └─────────── 文本+坐标都参与loss ───────────┘
```

**VQA任务**:
```
输入序列:
<img>[256 toks]</img>What is in the image? Answer:

目标序列:
                                                        A dog.<eos>
                                                        └─ loss ─┘
```

**训练策略**:
- **解冻**: 全模型参数可训练
- **分辨率**: 448×448
- **数据**: 70M+ 多任务样本
- **目标**: 增强多任务能力,引入grounding和OCR

---

### 5.3 Stage 3: 指令微调

#### **损失函数 (ChatML格式)**

```python
def stage3_sft_loss(logits, labels, role_tokens):
    """
    仅对assistant回答计算损失
    
    ChatML格式:
    <im_start>user\n{user_input}<im_end>\n
    <im_start>assistant\n{assistant_output}<im_end>\n
    """
    masked_labels = labels.clone()
    
    # 1. System prompt → IGNORE
    # 2. User input → IGNORE
    # 3. Assistant output → 计算loss
    # 4. 特殊token (<im_end>等) → 计算loss
    
    for i, role in enumerate(role_tokens):
        if role == "user" or role == "system":
            # 找到这段的起始和结束
            start_idx = find_role_start(i)
            end_idx = find_role_end(i)
            # 除了<im_end>,其余都IGNORE
            masked_labels[start_idx:end_idx-1] = -100
            
        elif role == "assistant":
            # 保留这段用于计算loss
            # 但role token本身要IGNORE
            role_token_idx = find_role_token_idx(i)
            masked_labels[role_token_idx] = -100
    
    loss = CrossEntropyLoss(ignore_index=-100)
    return loss(logits, masked_labels)
```

#### **SFT Loss计算示例**

```
完整序列:
<im_start>system
You are a helpful assistant.<im_end>
<im_start>user
Picture 1: <img>[256 toks]</img>
这是什么?<im_end>
<im_start>assistant
这是一只拉布拉多犬。<im_end>

Loss Mask:
<im_start>system                              ← IGNORE
You are a helpful assistant.                  ← IGNORE
<im_end>                                       ← ✓ 计算loss (特殊token)

<im_start>user                                 ← IGNORE
Picture 1: <img>[256 toks]</img>这是什么?      ← IGNORE
<im_end>                                       ← ✓ 计算loss

<im_start>assistant                            ← IGNORE (role token)
这是一只拉布拉多犬。                             ← ✓✓✓ 计算loss
<im_end>                                       ← ✓ 计算loss

实际Loss仅来自:
1. 特殊token: <im_end> (3次)
2. Assistant回答: "这是一只拉布拉多犬。"
```

**代码实现** (来自finetune.py:124-175):

```python
def preprocess(sources, tokenizer, max_len):
    im_start = tokenizer.im_start_id
    im_end = tokenizer.im_end_id
    
    for source in sources:
        input_id, target = [], []
        
        # System prompt
        system = [im_start] + _system + system_content + [im_end]
        input_id += system
        target += [im_start] + [IGNORE_TOKEN_ID]*(len(system)-3) + [im_end]
        
        # User/Assistant turns
        for sentence in source:
            role = sentence["from"]
            _input_id = tokenizer(role).input_ids + \
                        tokenizer(sentence["value"]).input_ids + [im_end]
            input_id += _input_id
            
            if role == 'user':
                # User输入: 全部IGNORE,除了<im_end>
                _target = [im_start] + [IGNORE_TOKEN_ID]*(len(_input_id)-3) + [im_end]
            elif role == 'assistant':
                # Assistant回答: 计算loss,但role token要IGNORE
                _target = [im_start] + [IGNORE_TOKEN_ID]*len(role_tokens) + \
                          _input_id[len(role_tokens)+1:-2] + [im_end]
            
            target += _target
    
    return input_id, target
```

**训练策略**:
- **冻结**: ViT
- **训练**: LLM + VL Adapter
- **数据**: 350K对话样本
- **目标**: 对齐人类偏好,提升对话能力

---

## 六、三阶段对比总结

| 维度 | Stage 1 | Stage 2 | Stage 3 |
|------|---------|---------|---------|
| **名称** | 预训练 | 多任务预训练 | 指令微调 |
| **分辨率** | 224×224 | 448×448 | 448×448 |
| **可训练参数** | ViT + Adapter | 全部 | LLM + Adapter |
| **数据量** | 1.4B | 70M+ | 350K |
| **数据类型** | 图文对 | 7种任务 | 对话数据 |
| **Loss范围** | 文本caption | 任务答案 | Assistant回答 |
| **Loss函数** | CrossEntropy | CrossEntropy | CrossEntropy |
| **特殊之处** | 图像token不计算loss | 不同任务label格式不同 | user/system部分IGNORE |
| **训练步数** | 50K | 19K | 8K |
| **Batch Size** | 30720 | 4096 | 128 |
| **Peak LR** | 2e-4 | 5e-5 | 1e-5 |

---

## 七、完整推理过程示例

让我用一个完整例子串联整个流程:

### 输入
```python
image = "dog.jpg"  # 一张拉布拉多的照片
query = "这是什么品种的狗?"
```

### Step 1: 视觉编码
```
dog.jpg (原始图像)
  ↓ resize
[3, 448, 448] (归一化tensor)
  ↓ ViT Encoder
[1024, 1536] (1024个patch特征)
  ↓ VL Adapter (Cross-Attention + 2D PE)
[256, 4096] (压缩后的视觉token)
```

### Step 2: 文本编码
```
"这是什么品种的狗?"
  ↓ Tokenizer
[151857, 28450, 25608, 31893, 102, 151858]
# <img>   这     是    什么   狗    </img>
  ↓ Embedding
[6, 4096] (6个token的embedding)
```

### Step 3: 序列拼接
```
[1 + 256 + 1 + 6, 4096]
= [264, 4096]

详细结构:
Position 0:     <img> embedding
Position 1-256: 256个视觉token (来自VL Adapter)
Position 257:   </img> embedding
Position 258:   "这" embedding
Position 259:   "是" embedding
Position 260:   "什么" embedding
Position 261:   "品种" embedding
Position 262:   "的" embedding
Position 263:   "狗" embedding
```

### Step 4: LLM生成
```
[264, 4096] (输入embeddings)
  ↓ 32层Transformer Decoder
[264, 4096] (hidden states)
  ↓ LM Head
[264, 151851] (logits over vocabulary)
  ↓ Sampling (最后一个位置)
Token ID: 45892 ("这")
  ↓ 自回归继续生成
"这是一只拉布拉多犬。"
```

### Step 5: 输出
```
完整输出序列:
<img>[256 toks]</img>这是什么品种的狗?这是一只拉布拉多犬。<eos>
                                      └──────────────┘
                                      生成的回答部分
```

---

## 八、关键设计思想总结

### 8.1 为什么图像token不计算loss?

**原因**:
1. **无监督信号**: 图像patch没有"正确答案"
2. **对齐目标**: 让视觉特征学会预测文本,而非重建图像
3. **效率**: 避免在高维视觉空间计算无意义的loss

### 8.2 为什么需要VL Adapter?

**原因**:
1. **维度压缩**: 1024→256,降低计算成本
2. **特征对齐**: ViT特征(1536维) → LLM空间(4096维)
3. **位置保留**: 2D位置编码保留空间信息,支持grounding

### 8.3 为什么三阶段训练?

**原因**:
1. **Stage 1**: 大规模弱监督数据建立视觉-语言映射
2. **Stage 2**: 高质量任务数据增强细粒度能力
3. **Stage 3**: 对话数据对齐人类偏好

### 8.4 统一Loss的优势

**优势**:
- **简洁**: 所有任务共享同一loss函数
- **灵活**: 通过数据格式控制不同任务
- **可扩展**: 添加新任务无需修改模型架构




