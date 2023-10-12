import torch
from transformers import BertTokenizer, BertModel
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class WordSelector:
    def __init__(self, model_path="./config/bert-base-chinese"):
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertModel.from_pretrained(model_path)

    def create_importance_scores(self, words):
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1

        word_scores = [word_counts[word] for word in words]
        return word_scores

    def get_word_embeddings(self, words):
        word_embeddings = []
        for word in words:
            inputs = self.tokenizer(word, return_tensors="pt")
            outputs = self.model(**inputs)
            word_embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy())

        return word_embeddings

    def select_distinct_words(self, words, num_words=4):
        importance_scores = self.create_importance_scores(words)
        word_embeddings = self.get_word_embeddings(words)
        similarity_matrix = cosine_similarity(word_embeddings)

        kmeans = KMeans(n_clusters=num_words, random_state=0).fit(similarity_matrix)

        selected_words = []
        for cluster_id in set(kmeans.labels_):
            words_in_cluster = [word for i, word in enumerate(words) if kmeans.labels_[i] == cluster_id]
            scores_in_cluster = [score for i, score in enumerate(importance_scores) if kmeans.labels_[i] == cluster_id]

            most_important_word_idx = np.argmax(scores_in_cluster)
            selected_words.append(words_in_cluster[most_important_word_idx])

        return selected_words

if __name__ == "__main__":

    # 示例&测试
    selector = WordSelector()
    words = ['飞泉', '飞瀑', '瀑布', '锦屏', '翠屏', '玉台', '琼台', '瑶池', '长天', '中游', '钟山', '南游', '楚水', '玉关',
             '东皋']
    selected_words = selector.select_distinct_words(words)
    print("Selected words:", selected_words)
