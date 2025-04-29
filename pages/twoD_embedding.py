# pages/2d_embedding.py
import streamlit as st
from _2d_wordembedding import draw_wordembedding_2d   # 假設你把函式放在 pages/_2d_wordembedding.py

st.set_page_config(page_title="2D Embedding 可視化", layout="wide")
st.title("📊 2D Word Embedding 可視化")

st.markdown(
    """
    在下面輸入多行句子（每行一句），
    按下「Generate」後會跑 Word2Vec + PCA，再用 Plotly 畫出 2D embedding。
    """
)

text = st.text_area("輸入句子（每行一句）", height=200)
sentences = [s.strip() for s in text.split("\n") if s.strip()]

if st.button("Generate 2D Embedding"):
    if not sentences:
        st.warning("至少輸入一句話！")
    else:
        fig = draw_wordembedding_2d(sentences)
        st.plotly_chart(fig, use_container_width=True)
