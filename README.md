 #MIMII 工业声音异常检测（SincNet+Transformer)
 
本项目基于MIMII数据集，在AI的辅助下，构建了多任务声学检测模型。不仅能识别机器类型（Fan，Pump,Slider,Valve),还可以判读声音是否异常

 ##核心特性
 
**SincNet** ：在孔学长的建议下（虽然在课上错说成了SinCNN)，采用了SincConv滤波器，原理好像是函数y=sin(x)/x乘某个数将其进行变换，然后用这些函数提取特征，这样抗噪能力更强,常用于音频领域

**Transformer Attention**：同样在孔学长的建议下，引入了Transformer自注意力机制(说实话，昨天专业导论上刚提了一下Transformer，说是当下热门的模型），可以精准捕捉异常信号

**Focal Loss与动态阈值**：为了应对不平衡的工业数据，自动搜索最佳报警阈值

**Linux极速加速**（其实这个是在和ai对话中了解的，无意中发现Linux系统相对于Windows系统有着加速的效果，于是就想不辜负那么麻烦的系统，让ai加入了这个）：支持多进程加载，GPU混合精度（AMP）（这个是我强烈要求AI让其充分利用GPU所得出的，导致我第一次爆显存）及余弦退火策略


 ##实验结果（Epoch13最优模型）
 
本项目在5w+条全量MIMII音频上进行20个Epoch多任务训练，最终各项核心指标如下：

**机器分类准确率**：**87.11%**

**机器分类F1（M-F1）**：**87.49%**

**异常检测AUC（A-AUC）**：**0.8573**

**异常检测F1-Score(A-F1)**:**0.5510**(包含动态阈值优化）

（至今不知道孔老先生当初怎么做到的93%）


 ##目录结构
```text
homework.2.ProMax
├── dataset.py
├── inference.py
├── model.py
├── requirements.txt
├── train.py
├── utils.py
└── weights
    └── best_sincnet_transformer.pkl


