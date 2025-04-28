import warnings
warnings.filterwarnings("ignore")
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from sklearn.decomposition import PCA
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

def draw_wordembedding_2d(sentence_list):
    tokenized_sentences = [simple_preprocess(sentence) for sentence in sentence_list]
    model = Word2Vec(tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4)
    word_vectors = np.array([model.wv[word] for word in model.wv.index_to_key])
    pca = PCA(n_components=3)
    reduced_vectors = pca.fit_transform(word_vectors)

    color_map = {
    0: 'red',
    1: 'blue',
    2: 'green',
    3: 'purple',
    4: 'orange',
    5: 'cyan',
    6: 'magenta',
    }
    word_colors = []
    for word in model.wv.index_to_key:
        for i, sentence in enumerate(tokenized_sentences):
            if word in sentence:
                word_colors.append(color_map[i])
                break

    word_ids = [f"word-{i}" for i in range(len(model.wv.index_to_key))]

    scatter = go.Scatter(
        x=reduced_vectors[:, 0],
        y=reduced_vectors[:, 1],
        mode='markers+text',
        text=model.wv.index_to_key,
        textposition='top center',
        marker=dict(color=word_colors, size=8),
        customdata=word_colors,
        ids=word_ids,
        hovertemplate="Word: %{text}<br>Color: %{customdata}"
    )

    # Create line traces for each displayed sentence
    display_array = [True, True, False, False, False, True, True]  # Example display array
    # 設定只畫出上面七個句子中的其中四句

    # Create line traces for each sentence
    line_traces = []
    for i, sentence in enumerate(tokenized_sentences):
        if display_array[i]:
            line_vectors = [reduced_vectors[model.wv.key_to_index[word]] for word in sentence] # line_vectors包含了每個句子中，每個字的array
            line_trace = go.Scatter(
                x=[vector[0] for vector in line_vectors],
                y=[vector[1] for vector in line_vectors],
                mode='lines',
                line=dict(color=color_map[i], width=1, dash='solid'),
                showlegend=True,
                name=f"Sentence {i+1}",  # Customize the legend text
                hoverinfo='all'  # Disable line trace hover info
            )
            # Set different marker symbols for the start and end words
            line_traces.append(line_trace)

    fig = go.Figure(data=[scatter] + line_traces)

    # Set the plot title and axis labels
    fig.update_layout(
        xaxis_title="X",
        yaxis_title="Y",
        title="2D Visualization of Word Embeddings",
        width=1000,  # Custom width
        height=1000  # Custom height
    )

    return fig
