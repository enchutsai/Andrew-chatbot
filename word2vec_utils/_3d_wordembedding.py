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

    # Create a 3D scatter plot using Plotly
    scatter = go.Scatter3d(
        x=reduced_vectors[:, 0],
        y=reduced_vectors[:, 1],
        z=reduced_vectors[:, 2],
        mode='markers+text',
        text=model.wv.index_to_key,
        textposition='top center',
        marker=dict(color=word_colors,size=2)
    )

    fig = go.Figure(data=[scatter])

    # Set the plot title and axis labels
    fig.update_layout(
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        title="3D Visualization of Word Embeddings",
        width=1000,  # Custom width
        height=1000  # Custom height
    )

    # Show the plot
    fig.show()

    word_ids = [f"word-{i}" for i in range(len(model.wv.index_to_key))]

    scatter = go.Scatter3d(
        x=reduced_vectors[:, 0],
        y=reduced_vectors[:, 1],
        z=reduced_vectors[:, 2],
        mode='markers+text',
        text=model.wv.index_to_key,
        textposition='top center',
        marker=dict(color=word_colors,size=2),
        customdata=word_colors,
        ids=word_ids,
        hovertemplate="Word: %{text}<br>Color: %{customdata}"
    )

    # Create line traces for each displayed sentence

    # Create line traces for each sentence
    line_traces = []
    for i, sentence in enumerate(tokenized_sentences):
        line_vectors = [reduced_vectors[model.wv.key_to_index[word]] for word in sentence]
        line_trace = go.Scatter3d(
            x=[vector[0] for vector in line_vectors],
            y=[vector[1] for vector in line_vectors],
            z=[vector[2] for vector in line_vectors],
            mode='lines',
            line=dict(color=color_map[i], dash='solid'),
            showlegend=False,
            hoverinfo='none'  # Disable line trace hover info
            )
        line_traces.append(line_trace)


    fig = go.Figure(data=[scatter] + line_traces)

    # Set the plot title and axis labels
    fig.update_layout(
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        title="3D Visualization of Word Embeddings",
        width=1000,  # Custom width
        height=1000  # Custom height
    )

    # Show the plot
    return fig