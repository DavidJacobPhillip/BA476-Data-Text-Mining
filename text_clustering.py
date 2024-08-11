import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

file = 'imdb_top_1000.csv'
df = pd.read_csv(file)
print(df.columns)
print(df.head())
tfidf = TfidfVectorizer()
df = df[df['Overview'].notna()]
text = tfidf.fit_transform(df['Overview'])

def find_optimal_clusters(data, max_k):
    iters = range(2, max_k + 1, 1)
    sse = []
    for k in iters:
        sse.append(MiniBatchKMeans(n_clusters=k, init_size=1024, batch_size=2048, random_state=10).fit(data).inertia_)
        print('Fit {} clusters'.format(k))
    f, ax = plt.subplots(1, 1)
    ax.plot(iters, sse, marker='o')
    ax.set_xlabel('Cluster Centers')
    ax.set_xticks(iters)
    ax.set_xticklabels(iters)
    ax.set_ylabel('SSE')
    ax.set_title('SSE by Cluster Center Plot')
    plt.show()
# find_optimal_clusters(text, 20)

def find_optimal_clusters2(data, max_k):
    iters = range(2, max_k + 1, 1)
    sse = []
    for k in iters:
        sse.append(KMeans(n_clusters=k, random_state=10).fit(data).inertia_)
        print('Fit {} clusters'.format(k))
    f, ax = plt.subplots(1, 1)
    ax.plot(iters, sse, marker='o')
    ax.set_xlabel('Cluster Centers')
    ax.set_xticks(iters)
    ax.set_xticklabels(iters)
    ax.set_ylabel('SSE')
    ax.set_title('SSE by Cluster Center Plot')
    plt.show()
# find_optimal_clusters2(text, 20)

def get_top_keywords(data, clusters, labels, n_terms):
    df = pd.DataFrame(data.todense()).groupby(clusters).mean()
    df_terms = pd.DataFrame()
    for i, r in df.iterrows():
        print('\nCluster {}'.format(i))
        print(','.join([labels[t] for t in np.argsort(r)[-n_terms:]]))
        df_terms[i] = [labels[t] for t in np.argsort(r)[-n_terms:]]
    print(df_terms)
    save_file = 'kmeans_' + str(len(df)) + '_topics_keywords.csv'
    df_terms.to_csv(save_file, index=False)
clusters = KMeans(n_clusters=5, random_state=10).fit_predict(text)      #n_clusters is where you modify the value you think it should be based on the elbow
get_top_keywords(text, clusters, tfidf.get_feature_names(), 20)
df['kmeans_cluster'] = clusters
print(df['kmeans_cluster'])

df['kmeans_cluster'] = clusters
df.to_csv('data_with_kmeans', index=False)