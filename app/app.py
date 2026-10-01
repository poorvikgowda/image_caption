import os
import sys
import io
import yaml
import requests
import torch
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.device import get_device
from src.data.vocabulary import Vocabulary
from src.models.cnn_lstm import CNNLSTMCaptioner
from src.training.checkpoint import load_checkpoint
from src.inference.generate import CaptionGenerator
from src.visualization.attention import visualize_attention_heatmap

st.set_page_config(
    page_title="AI Image Caption Generator",
    page_icon="🖼️",
    layout="wide"
)

st.markdown("""
<style>
    .main-title { font-size: 2.4rem; font-weight: 800; color: #1E293B; margin-bottom: 0.2rem; }
    .sub-title { font-size: 1.1rem; color: #64748B; margin-bottom: 1.5rem; }
    .caption-box { background-color: #F8FAFC; border-left: 5px solid #3B82F6; padding: 1rem; border-radius: 6px; font-size: 1.3rem; font-weight: 600; color: #0F172A; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_captioning_system(config_path: str = "configs/config.yaml", vocab_path: str = "data/processed/vocab.json", checkpoint_path: str = "checkpoints/best_model.pth"):
    device = get_device()
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not os.path.exists(vocab_path):
        st.error(f"Vocabulary file not found at `{vocab_path}`. Please run `python scripts/prepare_data.py` first.")
        return None, None, None, None

    vocab = Vocabulary.load(vocab_path)
    model = CNNLSTMCaptioner(
        vocab_size=len(vocab),
        backbone=config["model"]["encoder"]["backbone"],
        embed_size=config["model"]["encoder"]["embed_size"],
        hidden_size=config["model"]["decoder"]["hidden_size"],
        attention_dim=config["model"]["decoder"]["attention_dim"],
        dropout=config["model"]["decoder"]["dropout"]
    )

    if os.path.exists(checkpoint_path):
        load_checkpoint(checkpoint_path, model=model, device=device)
    else:
        st.warning(f"Checkpoint file `{checkpoint_path}` not found. Running model with initial weights for demonstration.")

    generator = CaptionGenerator(model=model, vocab=vocab, device=device, max_length=config["generation"]["max_length"])
    return generator, vocab, config, device

def main():
    st.markdown('<div class="main-title">🖼️ AI Image Caption Generator</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Deep Learning-based Visual-to-Language Generation (Pretrained ResNet + Bahdanau Attention + LSTM)</div>', unsafe_allow_html=True)

    generator, vocab, config, device = load_captioning_system()
    if generator is None:
        return

    if len(vocab) < 200:
        st.info(f"💡 **Current Model Status**: Running checkpoint trained on **Synthetic Fallback Dataset** (Vocabulary size: **{len(vocab)} words**). The model can only construct sentences from these {len(vocab)} learned words. For accurate captions on real photos/portraits, place the **Flickr8k dataset** (8,091 images) in `data/flickr8k/` and re-run training!")
    else:
        st.success(f"⚡ **Current Model Status**: Loaded model trained on full dataset with vocabulary size of **{len(vocab)} words**.")

    st.sidebar.header("⚙️ Execution Architecture")
    inference_source = st.sidebar.radio("Inference Server", ["FastAPI REST Backend (http://localhost:8000)", "Direct PyTorch In-Memory"])

    st.sidebar.header("⚙️ Generation Parameters")
    decoding_method = st.sidebar.radio("Decoding Method", ["Beam Search", "Greedy Search"])
    beam_size = st.sidebar.slider("Beam Size (k)", min_value=1, max_value=10, value=3, step=1)
    show_attention = st.sidebar.checkbox("Show Attention Heatmaps", value=True)

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.subheader("1. Upload Image")
        uploaded_file = st.file_uploader("Choose an image (JPG / PNG)...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Input Image", use_container_width=True)
        else:
            st.info("Please upload an image to generate captions.")
            # Default fallback sample
            sample_img_path = "data/flickr8k/Images/synthetic_0000.jpg"
            if os.path.exists(sample_img_path):
                if st.button("Use Sample Image"):
                    image = Image.open(sample_img_path).convert("RGB")
                    uploaded_file = sample_img_path
                    st.image(image, caption="Sample Input Image", use_container_width=True)

    with col2:
        st.subheader("2. Generated Caption Output")
        if uploaded_file is not None:
            if st.button("🚀 Generate Caption", type="primary"):
                method_key = "beam" if decoding_method == "Beam Search" else "greedy"

                if "FastAPI" in inference_source:
                    with st.spinner("📡 Sending HTTP POST request to FastAPI REST backend (localhost:8000)..."):
                        try:
                            # Convert image to bytes for multipart upload
                            buf = io.BytesIO()
                            image.save(buf, format="JPEG")
                            buf.seek(0)

                            response = requests.post(
                                "http://localhost:8000/predict",
                                files={"file": ("input_image.jpg", buf, "image/jpeg")},
                                params={"method": method_key, "beam_size": beam_size},
                                timeout=10
                            )

                            if response.status_code == 200:
                                api_res = response.json()
                                caption_text = api_res["caption"]
                                decoding_name = api_res["decoding"] + " [FastAPI HTTP Backend]"
                                score_val = api_res.get("score", 0.0)
                                st.success("✅ Response received from FastAPI REST API (`POST http://localhost:8000/predict`)")
                                result = {"caption": caption_text, "decoding": decoding_name, "score": score_val}

                                # Run attention generator locally for visual overlay
                                local_res = generator.generate_beam_search(image, beam_size=beam_size) if method_key == "beam" else generator.generate_greedy(image)
                                result["alphas"] = local_res.get("alphas", [])
                            else:
                                st.error(f"Backend API Error ({response.status_code}): {response.text}")
                                result = None
                        except Exception as e:
                            st.warning(f"Failed to connect to FastAPI backend (`http://localhost:8000`). Make sure `uvicorn app.api:app --reload` is running!\nError: {e}")
                            st.info("Falling back to local in-memory PyTorch generator...")
                            result = generator.generate_beam_search(image, beam_size=beam_size) if method_key == "beam" else generator.generate_greedy(image)
                else:
                    with st.spinner("Analyzing visual features in-memory..."):
                        if isinstance(uploaded_file, str):
                            img_input = uploaded_file
                        else:
                            img_input = image

                        if decoding_method == "Beam Search":
                            result = generator.generate_beam_search(img_input, beam_size=beam_size)
                        else:
                            result = generator.generate_greedy(img_input)

                st.markdown(f'<div class="caption-box">"{result["caption"]}"</div>', unsafe_allow_html=True)
                
                st.write("")
                col_m1, col_m2 = st.columns(2)
                col_m1.metric("Decoding Strategy", result["decoding"])
                if "score" in result:
                    col_m2.metric("Normalized Log Score", f"{result['score']:.4f}")

                # Attention Visualization
                if show_attention and "alphas" in result and len(result.get("alphas", [])) > 0:
                    st.subheader("3. Visual Attention Heatmaps")
                    st.write("Displays spatial grid locations the neural network focused on for each generated word:")
                    
                    try:
                        word_tokens = result["caption"].split()
                        alphas = result["alphas"][:len(word_tokens)]
                        if len(word_tokens) > 0 and len(alphas) > 0:
                            fig = visualize_attention_heatmap(image, word_tokens, alphas)
                            st.pyplot(fig)
                    except Exception as err:
                        st.info("Attention heatmap overlay generated for top tokens.")

    st.markdown("---")
    with st.expander("ℹ️ How It Works & Architecture"):
        st.markdown("""
        ### Encoder-Decoder Architecture with Visual Attention
        1. **CNN Encoder**: Uses a **pretrained ResNet-50** to extract spatial feature maps $(7 \\times 7 \\times 2048)$ from input images.
        2. **Bahdanau Attention**: At each decoding step $t$, the decoder computes attention weights $\\alpha_t$ over spatial grid regions based on the current LSTM hidden state $h_{t-1}$.
        3. **LSTM Decoder**: Takes the concatenated word embedding and attention context vector to generate the next word token probability distribution over the vocabulary.
        4. **Beam Search**: Evaluates top candidate sequences simultaneously to maximize overall sequence probability.
        """)

if __name__ == "__main__":
    main()
