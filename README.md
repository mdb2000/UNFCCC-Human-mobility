<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>

<h1> Human Mobility Analysis in the UNFCCC Documentation</h1>

<p>This repository contains the code used to analyze human mobility-related sentences extracted from over 9,000 official UNFCCC documents. The analysis involves Natural Language Processing (NLP) techniques, including word embeddings, clustering, and topic mining.</p>

<ul>
  <li>Here you can find the pinpoint database with the original PDF downloaded: <a href="https://journaliststudio.google.com/pinpoint/search?collection=b498f56be9c75e07&utm_source=collection_share_link">Pinpoint Database</a></li>
  <li>Here is a Google Drive Folder where you can download the documents: <a href="https://drive.google.com/drive/folders/1mAyZk5G5-eC081sFoHKMJJCQqEOGX5vM?usp=drive_link">Google Drive Folder</a></li>
</ul>

<p>Below is a step-by-step explanation of the Python code implemented for this project.</p>


<h2>1. Import Relevant Packages</h2>

<p>Start by importing the necessary Python packages for the analysis.</p>

<pre><code>
from bertopic import BERTopic
import re
from langdetect import detect
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
import nltk
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from collections import Counter
from nltk.tokenize import word_tokenize
from umap import UMAP
</code></pre>

<h2>2. Download Dataset Locally</h2>

<p>The dataset can be downloaded from the GitHub page as <code>df_final.csv</code>. Load the dataset using the following command:</p>

<pre><code>
# Load data
df = pd.read_csv('...\\df_final.csv')
</code></pre>

<h2>3. BERTopic Analysis</h2>

<h3>3.1 Initialize and Fit BERTopic Model</h3>
<p>First, initialize the BERTopic model, specifying the number of topics. Then, fit the model to the processed sentences from the dataset. The topics and probabilities are generated, along with hierarchical visualization.</p>

<pre><code>
# Initialize and fit the BERTopic model on the processed column of the dataset
topic_model = BERTopic(nr_topics=80)
topics, probs = topic_model.fit_transform(df['processed'].tolist())

#The following commands display different graphs and table useful for the comprehension
topic_model.get_topic_info()
topic_model.get_document_info(df['processed'].tolist())
hierarchical_topics = topic_model.hierarchical_topics(df['processed'].tolist())
topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
</code></pre>

<h3>3.2 Prepare Data for Plotting</h3>
<p>Generate embeddings using BERTopic and apply UMAP for dimensionality reduction, followed by clustering visualization.</p>

<pre><code>
# Prepare data for plotting
embeddings = topic_model._extract_embeddings(df['processed'], method="document")
umap_model = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit(embeddings)
df_umap = pd.DataFrame(umap_model.embedding_, columns=["x", "y"])
df_umap["topic"] = topics
</code></pre>

<h3>3.3 Visualize Topics</h3>
<p>Visualize topics, focusing on the top 15 topics, and represent outliers.</p>

<pre><code>
# Visualize topics
cmap = matplotlib.colors.ListedColormap(['#FF5722', '#03A9F4', '#4CAF50', '#80CBC4', '#673AB7', '#795548', '#E91E63', 
                                         '#212121', '#00BCD4', '#CDDC39', '#AED581', '#FFE082', '#BCAAA4', 
                                         '#B39DDB', '#F48FB1'])

fig, ax = plt.subplots(figsize=(15, 15))
scatter_outliers = ax.scatter(outliers['x'], outliers['y'], c="#808080", s=3, alpha=.3)
scatter = ax.scatter(non_outliers['x'], non_outliers['y'], c=non_outliers['topic'], s=4, alpha=.3, cmap=cmap)

centroids = to_plot.groupby("topic").mean().reset_index().iloc[1:]
for row in centroids.iterrows():
    topic = int(row[1]['topic'])
    topic_words = topic_model.get_topic(topic)
    if topic_words:
        text = f"{topic}: " + "_".join([x[0] for x in topic_words[:3]])
        ax.text(row[1]['x'], row[1]['y']*1.01, text, fontsize=fontsize, horizontalalignment='center')

ax.text(0.99, 0.01, f"BERTopic - Top {top_n} topics", transform=ax.transAxes, horizontalalignment="right", color="black")
plt.xticks([], [])
plt.yticks([], [])
plt.show()
topic_model.visualize_topics()
</code></pre>

<h2>4. BERT Word Embeddings and K-Means Clustering</h2>

<h3>4.1 BERT Embeddings</h3>
<p>Next, we extract the [CLS] token embeddings from the pre-trained BERT model for each sentence in the dataset.</p>

<pre><code>
# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to get BERT [CLS] token embedding for a sentence
def get_cls_sentence(sentence):
    input_ids = torch.tensor([tokenizer.encode(sentence, add_special_tokens=True, max_length=512, truncation=True)])
    with torch.no_grad():
        outputs = model(input_ids)
        cls_embedding = outputs[0][:, 0, :]
    return cls_embedding.flatten().numpy()

# Apply BERT to get sentence embeddings
df['sentence_embedding'] = df['Sentence'].apply(get_cls_sentence)
</code></pre>

<h3>4.2 K-Means Clustering</h3>
<p>Apply K-Means clustering with k=20 to group sentences based on their embeddings.</p>

<pre><code>
# Convert sentence embeddings to a numpy array for clustering
X = np.vstack(df['sentence_embedding'].values)

# Apply K-Means clustering with K=20
kmeans = KMeans(n_clusters=20, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X)
</code></pre>

<h2>5. Dimensionality Reduction with t-SNE</h2>

<p>Apply t-SNE to reduce the dimensionality of sentence embeddings to 2D and visualize the sentences in space.</p>

<pre><code>
# Apply t-SNE for dimensionality reduction
tsne_model = TSNE(n_components=2, random_state=42)
X_tsne = tsne_model.fit_transform(X)

# Plotting the sentences in 2D space
plt.figure(figsize=(10, 7))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], marker='o', alpha=0.7, s=1)
plt.title("2D Visualization of Sentences Using BERT Embeddings")
plt.xlabel('t-SNE feature 1')
plt.ylabel('t-SNE feature 2')
plt.show()
</code></pre>

<h3>5.1 Get Representative Words for Each Cluster</h3>
<p>Identify the most representative words for each cluster based on word frequency within the cluster.</p>

<pre><code>
# Function to get the most representative words for a cluster
def get_representative_words_for_cluster(cluster_num, top_n=5):
    words_in_cluster = list(itertools.chain(*df[df['cluster'] == cluster_num]['processed']))
    word_counts = Counter(words_in_cluster)
    most_common_words = [word for word, count in word_counts.most_common(top_n)]
    return ' '.join(most_common_words)

# Get representative words for each cluster
cluster_representative_words = [get_representative_words_for_cluster(i) for i in range(20)]
</code></pre>

<h3>5.2 Plot t-SNE Results with Cluster Centers</h3>
<p>Finally, plot the t-SNE results with the cluster centers annotated by the most representative words for each cluster.</p>

<pre><code>
# Plotting the t-SNE results with cluster centers annotated
plt.figure(figsize=(12, 8))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=df['cluster'], cmap='viridis', s=10, alpha=0.5)

# Annotate clusters using average t-SNE coordinates
for i in range(20):
    cluster_points = X_tsne[df['cluster'] == i]
    cluster_center = cluster_points.mean(axis=0)
    plt.text(cluster_center[0], cluster_center[1], cluster_representative_words[i], fontsize=7, ha='center', 
             bbox=dict(facecolor='white', alpha=0.5))

plt.title('t-SNE visualization with cluster centers annotated')
plt.xlabel('t-SNE feature 1')
plt.ylabel('t-SNE feature 2')

