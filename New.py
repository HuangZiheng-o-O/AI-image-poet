import argparse
import os

import torch
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
from transformers import BertTokenizer, T5ForConditionalGeneration
from transformers import GPT2LMHeadModel
from PIL import Image
from tqdm import tqdm

from transformers import AutoConfig, AutoModel, AutoTokenizer
import logging
import os
import sys
import json
import torch
import numpy as np

def cosine_similarity(x, y):
    return torch.sum(x * y) / (torch.sqrt(torch.sum(pow(x, 2))) * torch.sqrt(torch.sum(pow(y, 2))))

# 定义从图像生成古诗的类
class GeneratePoemFromImage:
    def __init__(self, clip_processor, clip, keyword_path=None, keyword_dict_path=None, dict_save_path="./datasets/keywords_dict.pt", top_k=8):
        self.clip_processor = clip_processor
        self.clip_model = clip
        self.keyword_path = keyword_path
        self.keyword_dict_path = keyword_dict_path
        self.dict_save_path = dict_save_path
        self.keyword_dict = None
        self.tok_k = top_k

        if self.keyword_dict_path is None:
            self.keyword_dict = self._create_keyword_dict(self.keyword_path, self.dict_save_path)
        else:
            self.keyword_dict = torch.load(self.keyword_dict_path)

    def _create_keyword_dict(self, keyword_root=None, save_root="./datasets/keywords_dict.pt"):
        root = keyword_root if keyword_root is not None else self.keyword_path
        keyword_name, text_feature = [], []
        with open(root, "r", encoding="utf-8") as f:
            data = f.readlines()
            for line in data:
                keyword_name.append(line.strip())
                feature = self.clip_processor(text=line.strip(), return_tensors="pt")
                text_feature.append(self.clip_model.get_text_features(**feature))

        text_feature = torch.cat(text_feature, dim=0)  # 将列表转换为一个矩阵
        text_feature = text_feature / text_feature.norm(p=2, dim=-1, keepdim=True)  # Normalize
        keyword_dict = {keyword_name[i]: text_feature[i:i + 1] for i in range(len(keyword_name))}
        torch.save(keyword_dict, save_root)
        return keyword_dict

    def top_k_keywords(self, img_feature):
        text_features = torch.cat(list(self.keyword_dict.values()), dim=0)
        similar = torch.tensor([
            cosine_similarity(text_features[i], img_feature) for i in range(text_features.shape[0])])
        top_k = torch.topk(similar, k=self.tok_k)
        return top_k.indices.squeeze(0).tolist()

    def generate_(self, image_path):
        image = Image.open(image_path)
        image = self.clip_processor(images=image, return_tensors="pt")
        img_feature = self.clip_model.get_image_features(**image)
        img_feature = img_feature / img_feature.norm(p=2, dim=-1, keepdim=True)

        top_keyword_index = self.top_k_keywords(img_feature)
        top_keywords = [list(self.keyword_dict.keys())[i] for i in top_keyword_index]
        # prompt = "关键词：" + " ".join(top_keywords) + " [EOS] "
        #
        # input_ids = self.lm_tokenizer.encode(prompt)
        # Here, I would typically call your language model to generate the poem
        # Since I'm focusing on the CLIP functionality, this is left as a representative step
        return top_keywords# , input_ids


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
    file_path = r"model"

    parser = argparse.ArgumentParser(description='Generate a poem from an image.')
    parser.add_argument('--clip_path', type=str, default="./config/Chinese_CLIP", help='path of CLIP processor & model')
    parser.add_argument('--image_path', default=".\images\gaozhiyuan.png", help='path of the image file')
    parser.add_argument('--keyword_path', default="./datasets/keywords.txt", help='path of poem keywords')
    parser.add_argument('--keyword_dict_path', type=str, default="./datasets/keywords_dict.pt", help='path to save keywords and its text encoder vector')
    parser.add_argument('--top_k', type=int, default=5, help='number of top candidates to show')
    args = parser.parse_args()
    processor = ChineseCLIPProcessor.from_pretrained(args.clip_path)
    clip_model = ChineseCLIPModel.from_pretrained(args.clip_path)

    generator = GeneratePoemFromImage(
        clip_processor=processor,
        clip=clip_model,
        keyword_path=args.keyword_path,
        keyword_dict_path=args.keyword_dict_path,
        top_k=args.top_k
    )
    top_keywords = generator.generate_(args.image_path)

    # 载入Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(r"model", trust_remote_code=True)
    config = AutoConfig.from_pretrained(r"model", trust_remote_code=True, pre_seq_len=128)
    model = AutoModel.from_pretrained(r"model", config=config, trust_remote_code=True)

    prefix_state_dict = torch.load(os.path.join(r"checkpoint-3000", "pytorch_model.bin"))
    new_prefix_state_dict = {}
    for k, v in prefix_state_dict.items():
        if k.startswith("transformer.prefix_encoder."):
            new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
    model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)

    model = model.quantize(4)
    model = model.half().cuda()
    model.transformer.prefix_encoder.float()
    model = model.eval()



    print("现在是基于图片./images/gaozhiyuan.jpg生成的Tang朝风格的诗")
    response, history  = model.chat(tokenizer, generator.create_prompt_ming(top_keywords),
                                   history=[])
    print(response)
    print("现在是基于图片./images/gaozhiyuan.jpg生成的Ming朝风格的诗")
    response2, history2  = model.chat(tokenizer, generator.create_prompt_song(top_keywords),
                                   history=[])
    print(response2)