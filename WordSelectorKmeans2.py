import torch
from transformers import BertTokenizer, BertModel
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class WordSelector2:
    def __init__(self, model_path="./config/bert-base-chinese"):
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertModel.from_pretrained(model_path)

    def get_word_embeddings(self, words):
        word_embeddings = []
        for word in words:
            inputs = self.tokenizer(word, return_tensors="pt")
            outputs = self.model(**inputs)
            word_embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy())
        return word_embeddings

    def select_distinct_words(self, words, num_words=4):
        importance_scores = np.random.rand(len(words))
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

    def process_keyword_lists(self, keyword_data):
        result = {}
        for image, words in keyword_data.items():
            selected_words = self.select_distinct_words(words)
            result[image] = selected_words
        return result


if __name__ == "__main__":

    # 示例&测试
    keyword_data = {
        "waterfall.jpg": ['丹青', '仙山', '山水', '林壑', '仙家', '国风', '仙宫', '仙境', '游仙', '飞泉', '画里', '东郭', '山川', '宿雨', '山亭'],
        "4.png": ['瑶琴', '抱琴', '歌罢', '貂蝉', '携琴', '长吟', '赏音', '青娥', '诗仙', '谪仙', '鸣琴', '长卿', '奏', '知音', '吟咏'],
        # ... 其他关键词数据
    }

    selector = WordSelector2()
    selected_keywords = selector.process_keyword_lists(keyword_data)
    print(selected_keywords)
