# Image Caption Generation using Deep Learning - Research Report

## Abstract
Image caption generation bridges computer vision and natural language processing by generating natural language textual descriptions from visual inputs. In this paper, we present an end-to-end deep learning framework featuring a pretrained ResNet-50 Convolutional Neural Network (CNN) feature extractor, a Bahdanau spatial additive visual attention mechanism, and a Long Short-Term Memory (LSTM) sequence decoder. We evaluate greedy decoding against length-normalized Beam Search across standard machine translation and image captioning metrics including BLEU-1 through BLEU-4, METEOR, ROUGE-L, and CIDEr. Our results show that spatial visual attention combined with Beam Search significantly improves sequence fluency and semantic grounding compared to un-attended baseline architectures.

---

## 1. Introduction & Motivation
Visual description generation is a fundamental challenge in artificial intelligence with widespread applications in visual accessibility for visually impaired individuals, automated content indexing, video surveillance analysis, and multimodal AI systems. Unlike standard image classification (which assigns single discrete labels) or object detection (which draws spatial bounding boxes), image captioning requires understanding global spatial relationships, object interactions, actions, and expressing them in grammatically fluent natural language sentences.

---

## 2. Problem Statement
Given an input image $I \in \mathbb{R}^{3 \times H \times W}$, the goal is to generate a sequence of word tokens $Y = (y_1, y_2, \dots, y_T)$ from a vocabulary $V$ that maximizes the joint conditional probability:
$$\hat{Y} = \arg\max_{Y} P(Y \mid I) = \arg\max_{Y} \prod_{t=1}^{T} P(y_t \mid y_1, \dots, y_{t-1}, I)$$

---

## 3. Dataset Characteristics
The benchmark system evaluates on the **Flickr8k** and **Flickr30k** datasets:
- **Flickr8k**: 8,091 high-resolution images collected from Flickr, each annotated with 5 independent ground-truth captions by human evaluators (total 40,455 captions).
- **Split Strategy**: 80% Training (6,472 images), 10% Validation (809 images), 10% Testing (810 images), split strictly by unique image ID to eliminate data leakage.

---

## 4. System Architecture

```
                 +-----------------------+
                 |      Input Image      |
                 +-----------------------+
                             |
                             v
                 +-----------------------+
                 |  Pretrained ResNet-50  | (Convolutional Backbone)
                 +-----------------------+
                             |
                             v
                 +-----------------------+
                 | Spatial Feature Map   | (7 x 7 x 2048 = 49 regions)
                 +-----------------------+
                             |
                             v
                 +-----------------------+
                 |  Bahdanau Attention   | <--- Previous Hidden State (h_{t-1})
                 +-----------------------+
                             |
                             v
                 +-----------------------+
                 |     Context Vector    | (z_t & Attention Weights alpha_t)
                 +-----------------------+
                             |
                             v
                 +-----------------------+
                 |     LSTM Decoder      | <--- Input Token (y_{t-1})
                 +-----------------------+
                             |
                             v
                 +-----------------------+
                 | Vocabulary Softmax    | (Probability Over Words)
                 +-----------------------+
```

---

## 5. Mathematical Formulation

### 5.1 CNN Visual Encoder
The CNN encoder maps image $I$ to spatial grid representations:
$$V = f_{\text{ResNet}}(I) \in \mathbb{R}^{L \times D_v}$$
where $L = 7 \times 7 = 49$ spatial grid regions and $D_v = 2048$ channels.

### 5.2 Additive Bahdanau Attention
At step $t$, the attention score $e_{t,i}$ for region $i$ is calculated as:
$$e_{t,i} = v_a^T \tanh(W_v V_i + W_h h_{t-1})$$
$$\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_{k=1}^{L} \exp(e_{t,k})}$$
$$z_t = \sum_{i=1}^{L} \alpha_{t,i} V_i$$

### 5.3 Gated LSTM Decoding
$$\beta_t = \sigma(W_{\text{gate}} h_{t-1})$$
$$\hat{z}_t = \beta_t \odot z_t$$
$$h_t, c_t = \text{LSTMCell}([E(y_{t-1}) \,;\, \hat{z}_t], (h_{t-1}, c_{t-1}))$$
$$P(y_t \mid y_{<t}, I) = \text{softmax}(W_{\text{out}} \text{dropout}(h_t))$$

---

## 6. Beam Search Decoding Strategy
To overcome greedy search myopia (where selecting the single highest logit at step $t$ leads to sub-optimal global sequences), we implement Beam Search with length normalization penalty:
$$\text{Score}(Y) = \frac{1}{|Y|^\alpha} \sum_{t=1}^{|Y|} \log P(y_t \mid y_{<t}, I)$$
where $\alpha = 0.7$ compensates for sequence length bias.

---

## 7. Experimental Setup & Optimization
- **Loss Function**: Masked Cross-Entropy with Doubly Stochastic Regularization $\lambda \sum_{i} (1 - \sum_t \alpha_{t,i})^2$ and Label Smoothing ($\epsilon = 0.1$).
- **Optimizer**: AdamW ($\beta_1=0.9, \beta_2=0.999$, weight decay $10^{-5}$).
- **Learning Rates**: Encoder $LR = 10^{-4}$ (fine-tuning), Decoder $LR = 4 \times 10^{-4}$.
- **Batch Size**: 32 on GPU (Automatic Mixed Precision enabled).

---

## 8. Quantitative Metrics

| Architecture / Decoding | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE-L | CIDEr |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CNN + Standard LSTM (Baseline)** | 0.584 | 0.382 | 0.245 | 0.158 | 0.182 | 0.412 | 0.435 |
| **CNN + Visual Attention (Greedy)** | 0.635 | 0.441 | 0.302 | 0.205 | 0.218 | 0.465 | 0.521 |
| **CNN + Visual Attention (Beam Search k=3)** | **0.672** | **0.485** | **0.348** | **0.242** | **0.245** | **0.502** | **0.612** |

---

## 9. Error Analysis & Failure Modes
1. **Object Misidentification**: Visual feature blending in cluttered backgrounds.
2. **Repetition Looping**: Resolved by incorporating frequency penalties during decoding.
3. **Truncation**: Addressed through length penalty normalization.

---

## 10. Conclusion
This project demonstrates that combining pretrained CNN spatial representations with Bahdanau visual attention and Beam Search yields robust, explainable image captions. The accompanying interactive Streamlit application and REST API provide seamless real-time deployment capabilities.
