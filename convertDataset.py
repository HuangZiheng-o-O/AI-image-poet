import json

def convert_to_target_format(input_filepath, output_filepath):
    # Load the JSON file line by line
    data_json_list = []
    with open(input_filepath, 'r', encoding='utf-8') as file:
        for line in file:
            data_json_list.append(json.loads(line.strip()))

    # Convert data to the target format
    formatted_data = []
    for entry in data_json_list:
        formatted_entry = {
            "prompt": f"我想写一首古诗，使用这些关键词： {entry['keywords']}，使用的朝代风格是{entry['dynasty']}",
            "response": entry['content'].replace('|', '，'),
            "history": []
        }
        formatted_data.append(formatted_entry)

    # Save the formatted data to a JSON file
    with open(output_filepath, 'w', encoding='utf-8') as file:
        json.dump(formatted_data, file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    input_filepath = "/Users/huangziheng/PycharmProjects/Chinese-Poem-Generate-Based-on-GPT2/dataset/CCPC/ccpc_valid_v1.0.json"
    output_filepath = "valid_3.json"
    convert_to_target_format(input_filepath, output_filepath)
