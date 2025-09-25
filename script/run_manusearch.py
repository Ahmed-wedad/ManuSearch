import sys, os, json, time, random
from zipfile import Path 
import argparse
import asyncio
from tqdm import tqdm

# Load environment variables from .env files (same as api_service_fastapi_legacy.py)
try:
    from dotenv import load_dotenv
    from pathlib import Path
    
    # Get the project root directory (same logic as api_service_fastapi_legacy.py)
    script_dir = Path(__file__).parent.parent.parent.absolute()
    
    # Try to load .env files in order of preference
    env_files = [
        script_dir / ".env",
        script_dir / ".env.openrag",  # Deprecated - kept for backward compatibility
    ]
    
    for env_file in env_files:
        if env_file.exists():
            load_dotenv(env_file)
            print(f"[ENV] Loaded environment variables from: {env_file}")
            break
    else:
        print("[WARNING] No .env file found, using system environment variables only")
        
except ImportError:
    print("[WARNING] python-dotenv not installed, using system environment variables only")
    print("   Install with: pip install python-dotenv")

p1 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(p1)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
# # Add the project root directory to the Python path
# script_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.dirname(script_dir)
# sys.path.insert(0, project_root)
from src.services.manusearch_agent_manager import warm_up_manusearch, get_manusearch_agent, is_manusearch_ready
from ManuSearch.searchagent.utils.utils import extract_answer_from_manusearch_result, normalize
# from searchagent.agent.agent import AgentInterface

def parse_args():
    parser = argparse.ArgumentParser(description="Run ManuSearch for various datasets and models.")
    parser.add_argument('--single_question', type=str, default=None, help="Single question to process instead of dataset")
    parser.add_argument('--dataset_name', type=str, required=False, default='custom', help="Name of the dataset to use.")
    parser.add_argument('--split', type=str, required=False, default='test', help="Dataset split to use.")
    parser.add_argument('--subset_num', type=int, default=-1, help="Number of examples to process. Defaults to all if not specified.")

    # parser.add_argument('--google_subscription_key', type=str, required=True, help="Google Search API subscription key(for serper.dev).")
    # parser.add_argument('--google_search_topk', type=int, default=5, help="topk returned documents for google search")
    # parser.add_argument('--proxy', type=str, help="port-based proxy(e.g., localhost:8080)")
    # parser.add_argument('--planner_model_name', type=str, required=True, help="Name of the planner model to use")
    # parser.add_argument('--planner_api_base', type=str, required=True, help="Base URL for the API endpoint")
    # parser.add_argument('--planner_api_key', type=str, required=True, help="api key for the planner model API endpoint")
    # parser.add_argument('--searcher_model_name', type=str, required=True, help="Name of the searcher model to use")
    # parser.add_argument('--searcher_api_base', type=str, required=True, help="Base URL for the API endpoint")
    # parser.add_argument('--searcher_api_key', type=str, required=True, help="api key for the searcher model API endpoint")
    # parser.add_argument('--reader_model_name', type=str, required=True, help="Name of the reader model to use")
    # parser.add_argument('--reader_api_base', type=str, required=True, help="Base URL for the API endpoint")
    # parser.add_argument('--reader_api_key', type=str, required=True, help="api key for the reader model API endpoint")
    # parser.add_argument('--cache_dir', type=str, required=False, help="cache for searched webpages")
    parser.add_argument('--concurrent_limit', type=int, default=32, help="Maximum number of concurrent API calls")

    # parser.add_argument('--temperature', type=float, default=0.6, help="Sampling temperature.")
    # parser.add_argument('--top_p', type=float, default=0.95, help="Top-p sampling parameter.")
    # parser.add_argument('--min_p', type=float, default=0.0, help="Minimum p sampling parameter.")
    # parser.add_argument('--top_k', type=int, default=30, help="Top-k sampling parameter.")
    # parser.add_argument('--repetition_penalty', type=float, default=1.05, help="Repetition penalty. If not set, defaults based on the model.")
    # parser.add_argument('--max_new_tokens', type=int, default=8192, help="Maximum number of new tokens to generate. If not set, defaults based on the model and dataset.")
    # parser.add_argument('--searcher_same_parameters', type=int, default=True, help="Whether adopt the same parameter as planner for searcher.")
    # parser.add_argument('--reader_same_parameters', type=int, default=True, help="Whether adopt the same parameter as planner for reader.")

    # parser.add_argument('--seed', type=int, default=None, help="Random seed for generation. If not set, will use current timestamp as seed.")
    return parser.parse_args()


async def process_single_sequence(agent, message, history=None):
    seq = {}

    # Use run_in_executor to execute synchronized methods
    loop = asyncio.get_event_loop()
    steps = await loop.run_in_executor(
        None,  # Default thread pool
        lambda: list(agent.get_answer(message, solve_method='iterative'))  # Convert to list to avoid generator problems
    )

    for step in steps:
        answer = step.get('final_resp', '')

    think = await loop.run_in_executor(
        None,
        agent.recorder.generate_reason_process
    )

    seq['output'] = answer
    seq['think'] = think
    return seq


async def main_async():
    # Parse arguments only when running as main script
    args = parse_args()
   
   
    try:
        if not is_manusearch_ready():
            warm_up_manusearch()
            # logging.info("ManuSearch agent pre-warmed")
        agent = get_manusearch_agent()
        # logging.info("ManuSearch agent initialized successfully")
    except Exception as e:
        # logging.warning(f"⚠️ Could not pre-warm ManuSearch agent: {e}")
        raise
    filtered_data = []
    with open("ManuSearch/data/enetcom_evaluation_dataset.jsonl", 'r', encoding='utf-8') as json_file:
        for line in json_file:
            line = line.strip()
            if line:  # Skip empty lines
                filtered_data.append(json.loads(line))  # ✅ Parse each line as JSON

    # Initialize batch output records
    batch_output_records = []
    start_time = time.time()

    # Create semaphore for concurrent API calls
    semaphore = asyncio.Semaphore(1)

    try:
        # Process questions 5-10 sequentially with complete processing
        reader = agent.reader
        with tqdm(total=50) as pbar:
            for item in filtered_data[:50]:
                # Process the question through ManuSearch
                seq = await process_single_sequence(
                    agent=agent, message=item['question'],
                )
                
                # Store raw results
                item['Output'] = seq['output']
                item['think'] = seq['think']  # Updated field name
                
                # Extract answer and think
                answer = extract_answer_from_manusearch_result(seq)
                think = normalize(seq.get('think'))
                
                # Determine input text: use answer if available, otherwise think
                input_text = answer if answer else think
                
                # Call Talker to generate conversational response
                talker_response = reader.talker_chat(
                    query=item['question'],
                    input_text=input_text,
                )
                
                # Update answer with Talker's response
                answer = talker_response
                item['Final_Answer'] = answer
                
                pbar.update(1)
    finally:
        total_time = time.time() - start_time

        t = time.localtime()
        random_num = str(random.randint(0, 99)).zfill(2)
        result_json_name = f'test_{random_num}_time_{t.tm_hour}_{t.tm_min}_{t.tm_sec}.json'
            
        with open(os.path.join("ManuSearch/outputs", result_json_name), mode='w', encoding='utf-8') as json_file:
            json.dump(filtered_data[:50], json_file, indent=4, ensure_ascii=False)
    
    total_time = time.time() - start_time

    t = time.localtime()
    random_num = str(random.randint(0, 99)).zfill(2)
    result_json_name = f'test_{random_num}_time_{t.tm_hour}_{t.tm_min}_{t.tm_sec}.json'
        
    with open(os.path.join("ManuSearch/outputs", result_json_name), mode='w', encoding='utf-8') as json_file:
        json.dump(filtered_data[:50], json_file, indent=4, ensure_ascii=False)


    print(f"Process completed. Results saved to {result_json_name} in {total_time:.2f} seconds.")

def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
