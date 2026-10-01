# Image Caption Generation - Viva & Interview Preparation Guide (45 Q&A)

This document contains 45 essential viva voce and technical interview questions with clear, direct explanations.

---

### Part 1: Computer Vision & Pretrained Encoders

#### Q1: What is the primary role of the CNN Encoder in Image Captioning?
**Answer**: The CNN encoder acts as a spatial visual feature extractor. Instead of producing class logits, it extracts a 3D feature tensor $H' \times W' \times D$ (e.g. $7 \times 7 \times 2048$ in ResNet-50) representing localized visual concepts across grid regions.

#### Q2: Why use a pretrained ResNet-50 instead of training a CNN from scratch?
**Answer**: Transfer learning. ResNet-50 trained on ImageNet has already learned low-level edges, textures, motifs, and high-level object representations across 1.2M images. Training from scratch on Flickr8k (8k images) would cause severe overfitting.

#### Q3: Why do we remove the final classification FC layer and global average pooling layer?
**Answer**: Global average pooling collapses spatial dimensions ($7 \times 7 \to 1 \times 1$), destroying location information. To enable spatial visual attention over specific regions (e.g. "where is the dog?"), we must retain the spatial grid feature map.

#### Q4: What does `FREEZE_ENCODER = True` mean during initial training?
**Answer**: It freezes CNN weights (`requires_grad = False`) so gradients only flow through the decoder. This speeds up training and prevents huge decoder gradients from destroying pretrained CNN representations during early epochs.

#### Q5: What is the spatial resolution of ResNet-50 features for a 224x224 input image?
**Answer**: $7 \times 7$ grid ($49$ spatial locations), each with a $2048$-dimensional feature vector.

---

### Part 2: Attention Mechanism

#### Q6: Why do we need an Attention Mechanism in Image Captioning?
**Answer**: Without attention, the decoder relies on a single static global image vector for every word. Visual attention dynamically refocuses on relevant image sub-regions at each time step (e.g., looking at ground when predicting "grass", and looking at animal when predicting "dog").

#### Q7: What is the difference between Bahdanau (Additive) Attention and Luong (Dot-Product) Attention?
**Answer**:
- **Bahdanau (Additive)**: Uses a non-linear hidden layer: $e_{i} = v_a^T \tanh(W_v V_i + W_h h_{t-1})$.
- **Luong (Dot-Product)**: Uses direct vector products: $e_i = h_t^T W V_i$ or $h_t^T V_i$. Additive attention performs better when encoder and decoder state dimensions differ.

#### Q8: What are attention weights $\alpha_{t,i}$?
**Answer**: A softmax probability distribution over the $49$ spatial grid regions at step $t$, such that $\sum_{i=1}^{49} \alpha_{t,i} = 1.0$.

#### Q9: What is the Context Vector $z_t$?
**Answer**: The weighted sum of spatial feature vectors using attention weights: $z_t = \sum_{i=1}^{L} \alpha_{t,i} V_i$.

#### Q10: What is Doubly Stochastic Attention Regularization?
**Answer**: A loss term $\lambda \sum_{i=1}^{L} (1 - \sum_{t=1}^T \alpha_{t,i})^2$ that encourages the model to attend to every spatial grid location equally across the sequence, preventing the model from ignoring parts of the image.

---

### Part 3: Sequence Modeling & Decoder

#### Q11: Why use LSTM instead of standard Vanilla RNN?
**Answer**: Vanilla RNNs suffer from vanishing and exploding gradients over long sequences. LSTMs use memory cells and gating mechanisms (Input, Forget, Output gates) to maintain long-term context without gradient decay.

#### Q12: What inputs does the LSTM cell receive at decoding step $t$?
**Answer**: The concatenated vector of the previous word embedding $E(y_{t-1})$ and the visual context vector $z_t$.

#### Q13: What are the special tokens used in vocabulary construction?
**Answer**:
- `<start>`: Marks sequence beginning.
- `<end>`: Marks sequence termination.
- `<pad>`: Pads sequences to uniform batch length.
- `<unk>`: Replaces out-of-vocabulary words.

#### Q14: What is Teacher Forcing and why is it used during training?
**Answer**: Teacher forcing feeds the ground-truth token $y_{t-1}$ as input to step $t$ instead of the model's own predicted token $\hat{y}_{t-1}$. It stabilizes early training and accelerates convergence.

#### Q15: What is the risk of 100% Teacher Forcing during training?
**Answer**: Exposure bias. During inference, ground-truth tokens are unavailable. If the model relies exclusively on ground truth during training, a single mistake during testing can cause error accumulation.

---

### Part 4: Inference & Beam Search

#### Q16: How does Greedy Decoding work?
**Answer**: At each step $t$, greedy decoding selects the word token with the absolute maximum softmax probability: $\hat{y}_t = \arg\max_w P(w \mid y_{<t}, I)$.

#### Q17: What is the main drawback of Greedy Decoding?
**Answer**: It is myopic (greedy local choices may lead to low probability sequences globally).

#### Q18: How does Beam Search solve greedy search drawbacks?
**Answer**: Beam search keeps track of $k$ top candidate sequence hypotheses simultaneously at each step, ranking them by cumulative log probability.

#### Q19: What is Length Normalization Penalty in Beam Search?
**Answer**: Because cumulative log-probabilities decrease with length ($\sum \log P < 0$), beam search naturally favors short captions. Dividing by $|Y|^\alpha$ ($\alpha \approx 0.7$) penalizes short sequences fairly.

#### Q20: When does Beam Search terminate generation?
**Answer**: When a candidate generates `<end>` or reaches `max_length`.

---

### Part 5: Evaluation Metrics

#### Q21: What is BLEU score?
**Answer**: Bilingual Evaluation Understudy. Measures modified n-gram precision between generated caption and ground truth references.

#### Q22: What is the difference between BLEU-1, BLEU-2, BLEU-3, and BLEU-4?
**Answer**:
- BLEU-1: Unigram precision (lexical accuracy).
- BLEU-2: Bigram precision.
- BLEU-3: Trigram precision.
- BLEU-4: 4-gram precision (grammatical fluency).

#### Q23: Why is Brevity Penalty necessary in BLEU?
**Answer**: Prevents short predictions (e.g. generating just "a dog") from achieving artificially high 100% precision scores.

#### Q24: What is ROUGE-L metric?
**Answer**: Recall-Oriented Understudy for Gisting Evaluation based on Longest Common Subsequence (LCS). Measures sentence structure similarity.

#### Q25: What is METEOR metric?
**Answer**: Metric for Evaluation of Translation with Explicit ORdering. Computes harmonic mean of precision and recall with explicit support for stemming and synonym matching.

#### Q26: What is CIDEr metric and why is it preferred for Image Captioning?
**Answer**: Consensus-based Image Description Evaluation. Uses TF-IDF weighting on n-grams to down-weight frequent generic words (like "a", "is") and reward specific visual descriptions.

---

### Part 6: Training & Data Processing

#### Q27: How do you prevent Data Leakage between splits?
**Answer**: Split dataset by unique Image ID rather than by caption records, ensuring all 5 captions of an image belong strictly to one split (train, val, or test).

#### Q28: Why do we ignore `<pad>` tokens in Cross Entropy Loss?
**Answer**: Padding tokens are artificial structural artifacts. Including them in loss computation would skew gradients toward predicting `<pad>`.

#### Q29: What is Label Smoothing?
**Answer**: Softens one-hot target vectors by distributing $\epsilon$ probability across non-target words. Prevents model overconfidence.

#### Q30: What is Gradient Clipping?
**Answer**: Clips gradient norm to a maximum threshold (e.g. $5.0$) to prevent exploding gradients in recurrent networks.

---

### Part 7: Modern Architecture Extensions

#### Q31: How can a Transformer Decoder replace the LSTM Decoder?
**Answer**: Uses Multi-Head Cross-Attention over visual feature maps and Self-Attention with causal masking over token sequences, allowing parallel sequence processing.

#### Q32: Why do Transformers require Positional Encoding?
**Answer**: Unlike LSTMs (which process tokens sequentially step-by-step), Transformers process all tokens concurrently and lack inherent sequence order without positional embeddings.

---

### Part 8: Practical Engineering & Deployment

#### Q33: Why use Automatic Mixed Precision (AMP)?
**Answer**: Performs matrix multiplications in FP16 while maintaining FP32 master weights, reducing GPU memory footprint by ~50% and boosting throughput.

#### Q34: What is the purpose of FastAPI backend in this project?
**Answer**: Provides a lightweight REST API (`POST /predict`) enabling external web applications, mobile apps, or microservices to request image caption predictions programmatically.

#### Q35: How does Streamlit render attention heatmaps interactively?
**Answer**: Takes the spatial attention matrix $\alpha$ $(49 \to 7 \times 7)$, resizes it to input image dimensions via bilinear interpolation, applies Gaussian smoothing, and overlays a jet colormap on the original image.

---

### Part 9: Failure Analysis & Tradeoffs

#### Q36: Why might a caption repeat words ("dog dog dog")?
**Answer**: Unbalanced hidden state weights or lack of repetition penalty during beam decoding.

#### Q37: How do you fix word repetition loops?
**Answer**: Add frequency penalty to beam candidate log-prob scoring or enforce coverage tracking.

#### Q38: Why do multiple reference captions matter during evaluation?
**Answer**: A single image can be correctly described in many valid ways. Evaluating against 5 references captures linguistic variance.

#### Q39: What is the main bottleneck of Beam Search compared to Greedy?
**Answer**: Beam search with size $k$ requires $k$ forward evaluations at every timestep, increasing computation time linearly by $k$.

#### Q40: What is the difference between frozen CNN vs fine-tuned CNN?
**Answer**: Frozen CNN uses static ImageNet features. Fine-tuning unfreezes higher conv layers (ResNet block 4) to adapt visual filters specifically to captioning tasks.

#### Q41: What is the effect of setting `min_freq` in vocabulary building?
**Answer**: Replaces rare words with `<unk>`, reducing vocabulary size, saving embedding parameters, and preventing overfitting on rare tokens.

#### Q42: What metric correlates best with human judgment for image captions?
**Answer**: CIDEr and SPICE.

#### Q43: How does batch collation handle variable-length captions?
**Answer**: Pads captions in a batch to `max_len` using `<pad>` token and generates a lengths vector for masked loss calculation.

#### Q44: Can this model run on CPU?
**Answer**: Yes. Device detection falls back to CPU seamlessly if CUDA GPU is unavailable.

#### Q45: How would you extend this project to video captioning?
**Answer**: Replace 2D CNN with 3D CNN (e.g. C3D/I3D) or Video Swin Transformer to extract spatial-temporal feature volumes across video frames.
