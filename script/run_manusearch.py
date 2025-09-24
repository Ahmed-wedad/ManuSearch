import sys, os, json, time, random
from zipfile import Path 
import numpy as np
import argparse
import asyncio
import aiohttp
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
    
    # # Set random seed
    # if args.seed is None:
    #     args.seed = int(time.time())
    # random.seed(args.seed)
    # np.random.seed(args.seed)  

    # # Set Search api
    # search_api_keys = [key.strip() for key in args.google_subscription_key.split(",")]
    # args.google_subscription_key = search_api_keys

    # # Set ManuSearch agent 
    # agent = AgentInterface(
    #     google_subscription_key=args.google_subscription_key,
    #     google_search_topk=args.google_search_topk,
    #     proxy=args.proxy,
    #     planner_model_name=args.planner_model_name,
    #     planner_api_base=args.planner_api_base,
    #     planner_api_key=args.planner_api_key,
    #     searcher_model_name=args.searcher_model_name,
    #     searcher_api_base=args.searcher_api_base,
    #     searcher_api_key=args.searcher_api_key,
    #     reader_model_name=args.reader_model_name,
    #     reader_api_base=args.reader_api_base,
    #     reader_api_key=args.reader_api_key,
    #     my_cache_dir=args.cache_dir,
    #     temperature=args.temperature,
    #     top_p=args.top_p, 
    #     min_p=args.min_p, 
    #     top_k=args.top_k,
    #     repetition_penalty=args.repetition_penalty, 
    #     max_new_tokens=args.max_new_tokens,
    #     searcher_same_parameters=args.searcher_same_parameters,
    #     reader_same_parameters=args.reader_same_parameters
    # )

    # # Modified data loading section
    # if args.single_question:
    #     # Create a single item in the same format as dataset items
    #     filtered_data = [{
    #         'Question': args.single_question,
    #     }]
    #     args.dataset_name = 'custom'  # Set dataset name to custom for single questions
    
    # else:
    #     # Original dataset loading logic
    #     if args.dataset_name == 'GAIA':
    #         data_path = f'../data/GAIA/{args.split}.json'
    #     elif args.dataset_name == 'FRAMES':
    #         data_path = f'../data/FRAMES/{args.split}.json'
    #     elif args.dataset_name == 'ORION':
    #         data_path = f'../data/ORION/{args.split}.json'
    #     else:
    #         data_path = f'../data/{args.dataset_name}.json'
        
    #     print('-----------------------')
    #     print(f'Using {args.dataset_name} {args.split} set.')
    #     print('-----------------------')


    # # Define output directory
    # if 'qwq' in args.planner_model_name.lower():
    #     model_short_name = 'qwq'
    #     if 'llama-8b' in args.searcher_model_name.lower():
    #         model_short_name = 'qwq-llama-8b'
    #     elif 'llama-70b' in args.searcher_model_name.lower():
    #         model_short_name = 'qwq-llama-70b'
    #     elif 'qwen-1.5b' in args.searcher_model_name.lower():
    #         model_short_name = 'qwq-qwen-1.5b'
    #     elif 'qwen-7b' in args.searcher_model_name.lower():
    #         model_short_name = 'qwq-qwen-7b'
    #     elif 'qwen-14b' in args.searcher_model_name.lower():
    #         model_short_name = 'qwq-qwen-14b'
    #     elif 'qwen-32b' in args.searcher_model_name.lower():
    #         model_short_name = 'qwq-qwen-32b'

    # elif 'deepseek' in args.planner_model_name.lower():
    #     model_short_name = 'dpsk'
    #     if 'llama-8b' in args.searcher_model_name.lower():
    #         model_short_name = 'dpsk-llama-8b'
    #     elif 'llama-70b' in args.searcher_model_name.lower():
    #         model_short_name = 'dpsk-llama-70b'
    #     elif 'qwen-1.5b' in args.searcher_model_name.lower():
    #         model_short_name = 'dpsk-qwen-1.5b'
    #     elif 'qwen-7b' in args.searcher_model_name.lower():
    #         model_short_name = 'dpsk-qwen-7b'
    #     elif 'qwen-14b' in args.searcher_model_name.lower():
    #         model_short_name = 'dpsk-qwen-14b'
    #     elif 'qwen-32b' in args.searcher_model_name.lower():
    #         model_short_name = 'dpsk-qwen-32b'

    # else:
    #     model_short_name = args.searcher_model_name.split('/')[-1].lower().replace('-instruct', '')

    # output_dir = f'../outputs/{args.dataset_name}.{model_short_name}.manusearch'
    # os.makedirs(output_dir, exist_ok=True)

    
    # if not args.single_question:
        # Load and prepare data
    
        # Pre-warm ManuSearch agent
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
    semaphore = asyncio.Semaphore(2)

    try:
        # Process only the first question for testing
        tasks = [
            process_single_sequence(
                agent=agent, message=question['question'],
            ) for question in filtered_data[:1]
        ]

        # Run all sequences concurrently with progress bar
        with tqdm(total=len(tasks)) as pbar:
            async def track_progress(task):
                result = await task
                pbar.update(1)
                return result
            
            tracked_tasks = [track_progress(task) for task in tasks]
            completed_sequences = await asyncio.gather(*tracked_tasks)
    finally:
        pass

    total_time = time.time() - start_time

    t = time.localtime()
    random_num = str(random.randint(0, 99)).zfill(2)
    result_json_name = f'test.json'

    for item, seq in zip(filtered_data[:1], completed_sequences):
        item['Output'] = seq['output']
        item['think'] = seq['think']  # Updated field name
        
    with open(os.path.join("ManuSearch/outputs", result_json_name), mode='w', encoding='utf-8') as json_file:
        json.dump(filtered_data, json_file, indent=4, ensure_ascii=False)


    print("Process completed.")

def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
