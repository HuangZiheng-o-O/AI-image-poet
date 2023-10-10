
#是为了调用img2poem_solution完成所有功能
import torch
from transformers import ChineseCLIPProcessor, ChineseCLIPModel, AutoTokenizer, AutoConfig, AutoModel
from transformers import BertTokenizer, T5ForConditionalGeneration
from transformers import GPT2LMHeadModel
from PIL import Image
from tqdm import tqdm
from WordSelectorKmeans import WordSelector
from WordSelectorKmeans2 import WordSelector2
from img2poem_solution import GenerateKeywordsFromImage
def cosine_similarity(x, y):
    return torch.sum(x * y) / (torch.sqrt(torch.sum(pow(x, 2))) * torch.sqrt(torch.sum(pow(y, 2))))
import sys
import os
from transformers import AutoConfig, AutoModel, AutoTokenizer





if __name__ == "__main__":

    file_path = r"model"
    processor = ChineseCLIPProcessor.from_pretrained("config/Chinese_CLIP")
    clip_model = ChineseCLIPModel.from_pretrained("config/Chinese_CLIP")
    # language_model = None
    # assert args.model_type == 'T5' or args.model_type == 'GPT2', "语言模型不支持!请使用T5或GPT2."
    # if args.model_type == 'T5':
    #     language_model = T5ForConditionalGeneration.from_pretrained(args.model_path)
    # elif args.model_type == 'GPT2':
    #     language_model = GPT2LMHeadModel.from_pretrained(args.model_path)

    generator = GenerateKeywordsFromImage(
        lm_model=T5ForConditionalGeneration.from_pretrained("config/t5_config"),
        lm_tokenizer=BertTokenizer(vocab_file="config/t5_config/vocab.txt", eos_token="[EOS]"),
        clip_processor=processor, clip=clip_model,
        keyword_path="datasets/keywords.txt",
        keyword_dict_path="./datasets/keywords_dict.pt",
        image_root_path  =r"images",
        dict_save_path="./datasets/keywords_dict.pt",
        keywordsNum=12,
        top_k=3
    )


    chatGLMtokenizer = AutoTokenizer.from_pretrained(r"model", trust_remote_code=True)
    config = AutoConfig.from_pretrained(r"model", trust_remote_code=True, pre_seq_len=128)
    model = AutoModel.from_pretrained(r"model", config=config, trust_remote_code=True)
    prefix_state_dict = torch.load(os.path.join("checkpoint-3000", "pytorch_model.bin"))
    new_prefix_state_dict = {}
    for k, v in prefix_state_dict.items():
        if k.startswith("transformer.prefix_encoder."):
            new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
    model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)

    # Comment out the following line if you don't use quantization
    model = model.quantize(4)
    model = model.half().cuda()
    model.transformer.prefix_encoder.float()
    model = model.eval()
    # response, history = model.chat(chatGLMtokenizer, "你好", history=[])


    # 打开一个文件来写入输出
    output_file = open('output.txt', 'w', encoding='utf-8')
    print("这一轮测试的参数是"
          "        keywordsNum=12,"
          "        top_k=3", file=output_file)
    image_dir = generator.image_root_path
    for filename in os.listdir(image_dir):
        file_path = os.path.join(image_dir, filename)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            print("=========================================================")
            print("=========================================================", file=output_file)
            print("处理图片:", file_path)
            print("处理图片:", file_path, file=output_file)
            keys, output_ids = generator.generate_(file_path)  # 示例词汇列表
            print(keys, file=output_file)
            #######################################################################
            print("现在是基于图片生成的Ming朝风格的诗1")
            selected_words = generator.get_topk_keywords_final(keys, generator.top_k)
            response, history = model.chat(chatGLMtokenizer, generator.create_prompt_ming(selected_words),
                                           history=[])
            print(selected_words, file=output_file)
            print(response)
            print(response,file_path, file = output_file)

            print("现在是基于图片生成的Ming朝风格的诗2")
            selected_words11 = generator.get_topk_keywords_final2(keys, generator.top_k)
            response11, history11 = model.chat(chatGLMtokenizer, generator.create_prompt_ming(selected_words11),
                                           history=[])
            print(selected_words11, file=output_file)
            print(response11)
            print(response11,file_path, file = output_file)
            print("--------------------------------------------------", file=output_file)
            print("现在是基于图片生成的Tang朝风格的诗1")
            selected_words2 = generator.get_topk_keywords_final(keys, generator.top_k)
            print(selected_words2, file=output_file)
            response2, history2 = model.chat(chatGLMtokenizer, generator.create_prompt_tang(selected_words2),
                                           history=[])
            print(response2)
            print(response2,file_path, file = output_file)
            print("现在是基于图片生成的Tang朝风格的诗2")
            selected_words22 = generator.get_topk_keywords_final2(keys, generator.top_k)
            print(selected_words22, file=output_file)
            response22, history22 = model.chat(chatGLMtokenizer, generator.create_prompt_tang(selected_words22),
                                           history=[])
            print(response22)
            print(response22,file_path, file = output_file)
            print("--------------------------------------------------", file=output_file)
            print("现在是基于图片生成的Song朝风格的诗1")
            selected_words3 = generator.get_topk_keywords_final(keys, generator.top_k)
            print(selected_words3, file=output_file)
            response3, history3 = model.chat(chatGLMtokenizer, generator.create_prompt_song(selected_words3),
                                           history=[])
            print(response3)
            print(response3,file_path, file = output_file)
            print("现在是基于图片生成的Song朝风格的诗2")
            selected_words33 = generator.get_topk_keywords_final2(keys, generator.top_k)
            print(selected_words33, file=output_file)
            response33, history33 = model.chat(chatGLMtokenizer, generator.create_prompt_song(selected_words33),
                                           history=[])
            print(response33)
            print(response33,file_path, file = output_file)
    # 关闭输出文件
    output_file.close()
