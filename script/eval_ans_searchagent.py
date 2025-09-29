import os,sys
import argparse
p1 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(p1)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
try:
    from searchagent.utils.utils import remove_think_tags
except ImportError:
    print("Warning: Could not import remove_think_tags, using fallback")
    def remove_think_tags(text):
        return text
import json
import re
from openai import OpenAI

def load_source_document(source_path):
    """
    Load the source document content from the relative path.
    
    Args:
        source_path: Relative path to the MD file from ../data_pipeline/
        
    Returns:
        str: Content of the MD file, or empty string if not found
    """
    try:
        # Construct full path relative to the Data_pipeline directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        manusearch_dir = os.path.dirname(script_dir)
        croissant_dir = os.path.dirname(manusearch_dir)
        code_dir = os.path.dirname(croissant_dir)
        data_pipeline_dir = os.path.join(code_dir, "Data_pipeline")
        
        full_path = os.path.join(data_pipeline_dir, source_path)
        
        # Ensure the path is safe and within the data_pipeline directory
        full_path = os.path.abspath(full_path)
        data_pipeline_dir = os.path.abspath(data_pipeline_dir)
        
        if not full_path.startswith(data_pipeline_dir):
            return ""
        
        if not os.path.exists(full_path):
            return ""
        
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()
            return content
            
    except Exception as e:
        return ""

def parse_args():
    parser = argparse.ArgumentParser(description="Do Eval datasets with openai api")

    parser.add_argument('--model_name', type=str, required=True, help="Name of the planner model to use")
    parser.add_argument('--api_base', type=str, required=True, help="Base URL for the API endpoint")
    parser.add_argument('--api_key', type=str, required=True, help="api key for the planner model API endpoint")
    
    parser.add_argument('--file_path', type=str, required=True, help="Set you file to eval")

    
    return parser.parse_args()

args = parse_args()

client = OpenAI(
    api_key=args.api_key,
    base_url=args.api_base
)


def generate(messages, model_name):
    response = client.chat.completions.create(
        **{
            "model": model_name,
            "messages": messages,
            "max_tokens": 2048,
        }
    )
    response = response.choices[0].message.content
    return response

# 数据验证和结果保存
def validate_data(file, model_name):
#     PROMPT = """You will receive a question along with a reference answer and a predicted answer. Your task is to evaluate the accuracy of the predicted answer and provide a concise explanation.

# Compare the predicted answer to the reference answer to determine its correctness.

# **Guidelines**
# - The criteria for evaluating the predicted answer should not be overly strict. If the predicted answer's meaning aligns closely with that of the reference answer, it can be deemed correct.
# - For each question, provide a brief explanation of your reasoning, followed by "Correct" or "Incorrect." Include your final assessment within <assessment> tags.
# - Use the provided source document content as authoritative "hard facts" to validate the accuracy of the predicted answer against real information.

# **Output Format**
# [Explanation]: Provide a brief explanation supporting your judgment.
# [Assessment]: Provide your assessment **within <assessment> tags**.

# Here is the question:
# {question}

# Here is the reference answer:
# {reference}

# Here is the predicted answer:
# {prediction}

# Here is the source document content (use as authoritative reference):
# {source_content}
# """
    PROMPT = '''Given a Question and its Golden Answer, verify whether the Predicted Answer is correct. The prediction is correct if it fully aligns with the meaning and key information of the Golden Answer. Respond with True if the prediction is correct and False otherwise.
Golden Answer may have multiple options, and matching any one of them is considered correct. use source document content as authoritative reference to validate the accuracy of the predicted answer against real information.

Question: {question}
Golden Answer: {reference}
Predicted Answer: {prediction}
Source Document Content: {source_content}
    '''

    # 获取所有以_finished结尾的jsonl文件
    print("Begin: ",file)
    file_path =file
    print("Now file:",file_path)

    # 初始化计数器
    valid_num = 0
    correct_num = 0
    incorrect_num = 0
    source_docs_loaded = 0
    result_data = []  # 用来存储每个对象的处理结果
    
    # Load data based on file format
    objects = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.endswith('.json'):
                # Handle JSON array format
                data = json.load(f)
                objects = data if isinstance(data, list) else [data]
                print(f"Loaded {len(objects)} objects from JSON array file")
            else:
                # Handle JSONL format (one JSON object per line)
                for line in f:
                    line = line.strip()
                    if line:  # Skip empty lines
                        try:
                            obj = json.loads(line)
                            objects.append(obj)
                        except json.JSONDecodeError as e:
                            print(f"Skipping invalid JSON line: {line[:100]}... Error: {e}")
                            continue
                print(f"Loaded {len(objects)} objects from JSONL file")
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return
    valid_num = len(objects)
    # Process each object
    for obj in objects:
            prediction = obj.get("Final_Answer", "I don't know")
            # if isinstance(prediction, dict):
            #     prediction = prediction.get("content")
            #     if isinstance(prediction, dict):
            #         prediction=prediction.get("concise_answer","")
            # else:
            #     prediction = remove_think_tags(prediction)
            question = obj.get("question", "")
            answer = obj.get("answer", [])
            
            # Load source document content
            source_content = ""
            metadata = obj.get("metadata", {})
            source_path = metadata.get("source_path", "")
            if source_path:
                source_content = load_source_document(source_path)
                if source_content:
                    # Truncate source content if too long (keep first 8000 characters to stay within token limits)
                    max_source_length = 8000
                    if len(source_content) > max_source_length:
                        source_content = source_content[:max_source_length] + "...[TRUNCATED]"
                    source_docs_loaded += 1
                else:
                    pass  # Silently skip if source document not found
            else:
                pass  # No source_path in metadata
            
            print("=="*70)
            print("Question:",question)
            print("Prediction:",prediction)
            print("Reference:",answer)
            if source_content:
                print(f"Source content loaded: {len(source_content)} characters")

            gpt4o_input = PROMPT.format(
                question=question,
                reference=answer, 
                prediction=prediction,
                source_content=source_content if source_content else "No source document available"
            )
            messages = [{'role': 'user', 'content': gpt4o_input}]

            # 获取GPT模型的评判结果
            model_output = generate(messages, model_name)

            if "false" in model_output.lower():
                is_correct = False
            else:
                is_correct = True

            if is_correct:
                correct_num += 1
            else:
                incorrect_num += 1

            obj['check_ans'] = model_output
            print("no.",valid_num,": ",model_output)

            # 将带有评判结果的对象加入到结果列表中
            result_data.append(obj)
           

    # 计算准确率
    if valid_num > 0:
        accuracy = correct_num / valid_num * 100
    else:
        accuracy = 0

    # 输出结果
    print(f"File: {file}")
    print(f"Valid objects: {valid_num}")
    print(f"Correct objects: {correct_num}")
    print(f"Incorrect objects: {incorrect_num}")
    print(f"Source documents loaded: {source_docs_loaded}")
    print(f"Accuracy: {accuracy:.2f}%\n")

    # 保存到新的JSONL文件，名称与原文件相同
    base_name = os.path.splitext(file_path)[0]  # Remove extension
    output_file_path = f"{base_name}_with_check_ans.jsonl"
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for obj in result_data:
            output_file.write(json.dumps(obj, ensure_ascii=False) + '\n')
    # break



if __name__ == "__main__":
    
    validate_data(args.file_path, args.model_name)
