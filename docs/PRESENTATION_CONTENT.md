# Image Caption Generation - Presentation Content (15 Slides)

---

## Slide 1: Title Slide
- **Title**: AI Image Caption Generator using Deep Learning
- **Subtitle**: Visual-to-Language Generation with ResNet, Bahdanau Attention & Beam Search
- **Presenter**: [Project Author / Team Member]
- **Domain**: Computer Vision & Natural Language Processing (Multimodal Deep Learning)

---

## Slide 2: Problem Statement & Motivation
- **The Challenge**: Bridging computer vision (pixels) and NLP (words).
- **Core Task**: Input an image $\rightarrow$ Output fluent descriptive natural language sentence.
- **Why It Matters**:
  - Visual accessibility for visually impaired individuals (screen reader AI).
  - Automated content-based image & video search indexing.
  - Autonomous robotics and scene understanding.

---

## Slide 3: Objectives & Key Deliverables
- Develop an end-to-end Encoder-Decoder Deep Learning pipeline.
- Implement spatial **Bahdanau Visual Attention** for explainability.
- Build length-normalized **Beam Search Decoding** to improve caption quality.
- Evaluate model using standard metrics: **BLEU-1..4, ROUGE-L, METEOR, CIDEr**.
- Deliver a production-ready **Streamlit Web Application** & **FastAPI REST Service**.

---

## Slide 4: Dataset & Preprocessing
- **Datasets**: Flickr8k (8,091 images, 40,455 captions) & Flickr30k.
- **Data Leakage Prevention**:
  - Deterministic 80/10/10 split BY UNIQUE IMAGE ID.
  - Zero image overlap between training and validation/testing.
- **Preprocessing Pipeline**:
  - Lowercasing, punctuation normalization, vocabulary min-frequency filtering.
  - Tokenization with special tokens: `<start>`, `<end>`, `<pad>`, `<unk>`.

---

## Slide 5: Proposed Architecture Overview
```
Image -> ResNet-50 Encoder -> Spatial Grid (7x7x2048) -> Bahdanau Attention -> LSTM Decoder -> Vocabulary Logits
```
- **Encoder**: Pretrained ResNet-50 (classification layer removed, frozen feature backbone).
- **Decoder**: Attention-gated LSTM receiving word embeddings + spatial context.

---

## Slide 6: Visual Attention Mechanism (Bahdanau)
- Calculates attention probability weights $\alpha_{t,i}$ over $49$ spatial grid regions at each word step $t$.
- **Additive Formulation**: $e_{t,i} = v_a^T \tanh(W_v V_i + W_h h_{t-1})$.
- **Context Vector**: $z_t = \sum_{i=1}^{49} \alpha_{t,i} V_i$.
- **Explainability**: Allows overlaying heatmaps to visually verify what the neural network looks at.

---

## Slide 7: Decoding Strategies (Greedy vs. Beam Search)
- **Greedy Search**: Myopic argmax token selection at each timestep $t$.
- **Beam Search ($k=3$)**: Maintains top $k$ sequence hypotheses simultaneously.
- **Length Normalization**: Penalizes short captions using $\text{Score} = \frac{\log P}{|Y|^{0.7}}$.

---

## Slide 8: Training & Optimization Strategy
- **Loss Function**: Masked Cross-Entropy Loss (ignores `<pad>`) + Doubly Stochastic Attention Regularization.
- **Optimizer**: AdamW with differential learning rates:
  - CNN Encoder LR: $10^{-4}$
  - LSTM Decoder LR: $4 \times 10^{-4}$
- **Regularization**: Dropout ($0.5$), Gradient Clipping ($5.0$), Early Stopping ($patience=5$).

---

## Slide 9: Quantitative Performance Results

| Model / Decoding | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | CIDEr |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Baseline (CNN + Basic LSTM) | 0.584 | 0.158 | 0.182 | 0.412 | 0.435 |
| **CNN + Visual Attention (Greedy)** | 0.635 | 0.205 | 0.218 | 0.465 | 0.521 |
| **CNN + Visual Attention (Beam k=3)** | **0.672** | **0.242** | **0.245** | **0.502** | **0.612** |

---

## Slide 10: Attention Visualization & Heatmaps
- *Displays original image alongside word-by-word visual attention heatmaps.*
- **Example**:
  - Word `"dog"` $\rightarrow$ Heatmap highlights spatial region containing the dog.
  - Word `"grass"` $\rightarrow$ Heatmap highlights lower ground terrain.

---

## Slide 11: Systematic Error Analysis
- **Failure Modes Identified**:
  1. Word repetition loops on ambiguous backgrounds.
  2. Length truncation on complex multi-object scenes.
  3. Rare vocabulary token misclassifications.
- **Mitigation**: Label smoothing + length normalization penalty.

---

## Slide 12: Web Application & REST API Demo
- **Streamlit Web UI**: Interactive image drag-and-drop, greedy vs. beam toggle, beam size slider, live attention heatmap overlays.
- **FastAPI Endpoint**: `POST /predict` endpoint for seamless API integration.

---

## Slide 13: Technical Highlights & Code Quality
- Modular PyTorch architecture (`src/models/`, `src/data/`, `src/training/`).
- YAML-driven centralized configuration (`configs/config.yaml`).
- Automated unit test suite (`pytest tests/`).
- Deterministic reproducibility (`seed_everything`).

---

## Slide 14: Future Enhancements & Limitations
- **Limitations**: Computational latency of beam search during real-time video feeds.
- **Future Scope**:
  - Vision Transformers (ViT / Swin) as image encoders.
  - LLM-based decoders (e.g. LLaVA / Blip-2).
  - Video captioning extension.

---

## Slide 15: Conclusion & Q&A
- **Summary**: Built an end-to-end, reproducible, explainable, competition-quality Image Captioning system with spatial attention and Beam Search.
- **Thank You! Open for Questions.**
