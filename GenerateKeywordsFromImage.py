# 专门用来生成关键词
import os
import torch
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
from transformers import BertTokenizer, T5ForConditionalGeneration
from transformers import GPT2LMHeadModel
from PIL import Image
from tqdm import tqdm
from WordSelectorKmeans import WordSelector
from WordSelectorKmeans2 import WordSelector2
import os
def cosine_similarity(x, y):
    return torch.sum(x * y) / (torch.sqrt(torch.sum(pow(x, 2))) * torch.sqrt(torch.sum(pow(y, 2))))
import sys

class GenerateKeywordsFromImage:

    def __init__(self,
                 lm_tokenizer: BertTokenizer,
                 lm_model,
                 clip_processor: ChineseCLIPProcessor,
                 clip: ChineseCLIPModel,  # clip预训练模型
                 image_root_path: str = None,
                 keyword_path: str = None,
                 keyword_dict_path: str = None,
                 dict_save_path: str = "./datasets/keywords_dict.pt",
                 top_k: int = 4,
                 keywordsNum : int = 15
                 ):

        self.clip_processor = clip_processor
        self.clip_model = clip
        self.image_root_path = image_root_path
        self.keyword_path = keyword_path
        self.keyword_dict_path = keyword_dict_path
        # self.keyword_dict_path = None # 设置为None表示需要生成新的关键词字典
        self.dict_save_path = dict_save_path
        self.keyword_dict = None
        self.top_k = top_k
        self.keywordsNum = keywordsNum
        self.lm_tokenizer = lm_tokenizer
        self.lm_model = lm_model

        assert self.keyword_path is not None or self.keyword_dict_path is not None, "用于CLIP模型对比的关键词不存在，需要提供关键词"

        if self.keyword_dict_path is None:
            self.keyword_dict = self._create_keyword_dict(self.keyword_path, self.dict_save_path)
        else:
            self.keyword_dict = torch.load(self.keyword_dict_path)

    def _create_keyword_dict(self, keyword_root=None, save_root="./datasets/keywords_dict.pt"):
        assert self.keyword_path is not None or keyword_root is not None, "请提供keyword.txt"

        root = keyword_root if keyword_root is not None else self.keyword_path
        keyword_name, text_feature = [], []
        with open(root, "r", encoding="utf-8") as f:
            data = f.readlines()
            for line in tqdm(data, total=len(data)):
                keyword_name.append(line.strip())
                feature = self.clip_processor(text=line.strip(), return_tensors="pt")
                text_feature.append(self.clip_model.get_text_features(**feature))
                # line 包含 "云根"。
                # keyword_name 包含一个元素 "云根"。
                # feature 包含通过 clip_processor 处理后的文本特征张量。
                # text_feature 包含一个文本特征张量，这个列表随后包含所有关键字的特征张量。

        text_feature = torch.cat(text_feature, dim=0)  # 将列表转换为一个矩阵
        text_feature = text_feature / text_feature.norm(p=2, dim=-1, keepdim=True)  # Normalize
        keyword_dict = {keyword_name[i]: text_feature[i:i + 1] for i in range(len(keyword_name))}
        # {
        #     '云根': tensor([[0.1234, 0.5678, ...]]),
        #     '存': tensor([[0.9876, 0.5432, ...]]),
        #     ...
        #      '肥': tensor([[0.2345, 0.6789, ...]])
        # }
        torch.save(keyword_dict, save_root)
        return keyword_dict

    def top_k_keywords(self, img_feature,keywordsNum = 15):
        text_features = torch.cat(list(self.keyword_dict.values()), dim=0)
        similar = torch.tensor([
            cosine_similarity(text_features[i], img_feature) for i in range(text_features.shape[0])])
        top_k = torch.topk(similar, k=keywordsNum)

        return top_k.indices.squeeze(0).tolist()

    def generate_(self, image_path,keywordsNum = 15):
        image = Image.open(image_path)
        image = self.clip_processor(images=image, return_tensors="pt")
        img_feature = self.clip_model.get_image_features(**image)
        img_feature = img_feature / img_feature.norm(p=2, dim=-1, keepdim=True)

        top_keyword_index = self.top_k_keywords(img_feature,keywordsNum)
        top_keywords = [list(self.keyword_dict.keys())[i] for i in top_keyword_index]
        prompt = "关键词：" + " ".join(top_keywords) + " [EOS] "

        input_ids = self.lm_tokenizer.encode(prompt)
        return top_keywords, input_ids

    def get_topk_keywords_final(self,keys,num_words):
        selector = WordSelector()
        selected_words = selector.select_distinct_words(keys, num_words)
        return selected_words
    def get_topk_keywords_final2(self,keys,num_words):
        selector = WordSelector2()
        selected_words = selector.select_distinct_words(keys, num_words)
        return selected_words

    def create_prompt_song(self, top_keywords):
        keywords_string = " ".join(top_keywords)
        prompt = f"我想写一首古诗，使用这些关键词： {keywords_string}，使用的朝代风格是Song"
        return prompt


    def create_prompt_ming(self, top_keywords):
        keywords_string = " ".join(top_keywords)
        prompt = f"我想写一首古诗，使用这些关键词： {keywords_string}，使用的朝代风格是Ming"
        return prompt


    def create_prompt_tang(self, top_keywords):
        keywords_string = " ".join(top_keywords)
        prompt = f"我想写一首古诗，使用这些关键词： {keywords_string}，使用的朝代风格是Tang"
        return prompt


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Generate a poem from an image.')
    # add arguments
    parser.add_argument('--vocab_path', type=str, default="./config/t5_config/vocab.txt",
                        help='the path of the tokenizer vocab')
    parser.add_argument('--model_type', type=str, default="T5", help='choose language model, \'T5\' or \'GPT2\'')
    parser.add_argument('--model_path', type=str, default="./config/t5_config", help='the path of language model')
    parser.add_argument('--clip_path', type=str, default="./config/Chinese_CLIP",
                        help='the path of CLIP processor & model')
    parser.add_argument('--image_path', default="./images/1.png", help='the path of the image file')
    parser.add_argument('--image_root_path', default="./images", help='the path of the image file')

    parser.add_argument('--keyword_path', default="./datasets/keywords.txt", help='the path of poem keywords')
    parser.add_argument('--keyword_dict_path', type=str, default="./datasets/keywords_dict.pt",
                        help='the path to save keywords and its text encoder vector')
    parser.add_argument('--epochs', type=int, default=50, help='the number of epochs for training (default: 50)')
    parser.add_argument('--top_k', type=int, default=4, help='the number of top candidates to show (default: 5)')
    parser.add_argument('--keywordsNum', type=int, default=15, help='the number of keywords to show (default: 15)')
    args = parser.parse_args()
    # tokenizer = BertTokenizer(vocab_file=args.vocab_path, eos_token="[EOS]")
    processor = ChineseCLIPProcessor.from_pretrained(args.clip_path)
    clip_model = ChineseCLIPModel.from_pretrained(args.clip_path)
    tokenizer = BertTokenizer(vocab_file=args.vocab_path, eos_token="[EOS]")
    lm_model = None
    assert args.model_type == 'T5' or args.model_type == 'GPT2', "语言模型不支持!请使用T5或GPT2."
    if args.model_type == 'T5':
        lm_model = T5ForConditionalGeneration.from_pretrained(args.model_path)
    elif args.model_type == 'GPT2':
        lm_model = GPT2LMHeadModel.from_pretrained(args.model_path)

    generator = GenerateKeywordsFromImage(
        clip_processor=processor, clip=clip_model,
        lm_model = lm_model,
        keyword_path=args.keyword_path,
        keyword_dict_path=args.keyword_dict_path,
        dict_save_path=args.keyword_dict_path,
        top_k=args.top_k,
        keywordsNum=args.keywordsNum,
        lm_tokenizer = tokenizer,
        image_root_path = args.image_root_path

    )

    #
    # image_path = './images/2.png'
    # # for i in os.listdir(image_path):
    # #     if i.endswith(('.png', '.jpg', '.jpeg')):
    # #         image_path = os.path.join(image_path, i)
    # #         print(image_path)
    # keys, output_ids = generator.generate_(image_path)    # 示例词汇列表
    # print(keys)
    # print("selectors1")
    # selector = WordSelector()
    # selected_words = selector.select_distinct_words(keys, num_words=4)
    # print(selected_words)
    #
    # selector2 = WordSelector2()
    # selected_words2 = selector2.select_distinct_words(keys, num_words=4)
    # print(selected_words2)
    # print("--------------------------------------------------")

    # 打开一个文件来写入输出
    output_file = open('output.txt', 'w', encoding='utf-8')

    image_dir = generator.image_root_path
    for filename in os.listdir(image_dir):
        file_path = os.path.join(image_dir, filename)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            print("处理图片:", file_path, file=output_file)
            keys, output_ids = generator.generate_(file_path)  # 示例词汇列表
            print(keys, file=output_file)
            print("selectors1", file=output_file)
            selected_words = generator.get_topk_keywords_final(keys, generator.top_k)
            print(selected_words, file=output_file)
            print("--------------------------------------------------", file=output_file)
            selected_words2 = generator.get_topk_keywords_final2(keys, generator.top_k)
            print(selected_words2, file=output_file)

    # 关闭输出文件
    output_file.close()


