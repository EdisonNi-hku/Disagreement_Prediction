import argparse
import pandas as pd
import random
import torch
import pickle
import ast

from transformers import AutoTokenizer, set_seed, logging
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
# from huggingface_hub import login
#
# login(token="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")

GPU_NUM = torch.cuda.device_count()

logger = logging.get_logger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_cache_dir", type=str, default="cache")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--architecture", type=str, default='')
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--tokenizer_path", type=str, default="")
    parser.add_argument('--prompt_templates', nargs='+', help='List of prompt template files', required=True)
    parser.add_argument('--dataset_files', nargs='+', help='List of dataset files', required=True)
    parser.add_argument("--instruction_field", type=str, default="instruction")
    parser.add_argument('--output_files', nargs='+', help='List of output files', required=True)
    parser.add_argument("--max_new_token", type=int, default=4096)
    parser.add_argument("--max_model_len", type=int, default=-1)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--trust_remote_code", action="store_true", default=False)
    parser.add_argument("--load_tokenizer", action="store_true", default=False)
    parser.add_argument("--format", action="store_true", default=False)
    parser.add_argument("--force_budget", action="store_true", default=False)
    parser.add_argument("--logprobs", type=int, default=None)
    parser.add_argument("--times", type=int, default=1)
    return parser.parse_args()


def main(args):
    assert len(args.output_files) == len(args.prompt_templates) == len(args.dataset_files)
    all_prompts = []
    all_prompts_idx = []
    all_prompts_unflatten = []
    all_prompts_outputs = [[] for _ in range(len(args.prompt_templates))]

    if args.load_tokenizer:
        tokenizer_dir = args.model_path if args.lora_path is None else args.lora_path
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, cache_dir=args.model_cache_dir, padding_side='left',
                                                  local_files_only=False)
        tokenizer_name = args.model_path
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, cache_dir=args.model_cache_dir,
                                                  padding_side='left', local_files_only=False)
        tokenizer_name = args.tokenizer_path
    stops = [tokenizer.eos_token]

    for file_id, (prompt_template, dataset_file) in enumerate(zip(args.prompt_templates, args.dataset_files)):
        if args.force_budget:
            df = pd.read_csv(dataset_file)
            if df['output'][0].startswith('['):
                outputs = [ast.literal_eval(o)[0].split('</think>')[0] + '\nWait' for o in df['output']]
            else:
                outputs = df['output']
            prompts = [p + o for p, o in zip(df['prompt'], outputs)]
            args.format = False
        else:
            with open(prompt_template, 'r') as f:
                template = f.readlines()
            test_data = pd.read_csv(dataset_file)
            if 'ghc' in prompt_template:
                if 'r1' in prompt_template:
                    prompts = [[{"role": "user", "content": template.format(post=t)}] for t in test_data['Text'].to_list()]
                else:
                    prompts = [[
                            {"role": "system", "content": "You are a helpful assistant with expertise in detecting hate speech from social media posts."},
                            {"role": "user", "content": template.format(post=t)},
                        ] for t in test_data['Text'].to_list()]
            elif 'helpsteer2' in prompt_template:
                if 'r1' in prompt_template:
                    prompts = [[{"role": "user", "content": template.format(prompt=p, response_a=r_a, response_b=r_b)}] for p, r_a, r_b in zip(test_data['prompt'], test_data['response_1'], test_data['response_2'])]
                else:
                    prompts = [[
                            {"role": "system", "content": "You are a helpful research assistant with expertise in annotating preference for AI responses."},
                            {"role": "user", "content": template.format(prompt=p, response_a=r_a, response_b=r_b)
                        }] for p, r_a, r_b in zip(test_data['prompt'], test_data['response_1'], test_data['response_2'])]
            else:
                if 'r1' in prompt_template:
                    prompts = [[{"role": "user", "content": template.format(post=t)}] for t in test_data['text'].to_list()]
                else:
                    prompts = [[
                            {"role": "system", "content": "You are a helpful research assistant with expertise in detecting emotions from social media posts."},
                            {"role": "user", "content": template.format(post=t)},
                        ] for t in test_data['text'].to_list()]

        if args.format:
            if isinstance(prompts[0], str):
                prompts = [[{'role': 'user', 'content': p}] for p in prompts]
            prompts = [tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True) for p in prompts]

        print(prompts[0])

        if args.sample:
            prompts = prompts[:args.sample]

        all_prompts_unflatten.append(prompts)
        all_prompts.extend(prompts)
        all_prompts_idx.extend([file_id for _ in range(len(prompts))])

    raw_outputs = [[] for _ in range(len(all_prompts))]
    for t in range(args.times):
        sampling_params = SamplingParams(
            n=args.n,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            seed=args.seed + t,
            logprobs=args.logprobs,
            max_tokens=args.max_new_token,
            stop=stops,
            include_stop_str_in_output=True,
        )
        if args.max_model_len > 0:
            len_dict = {"max_model_len": args.max_model_len}
        else:
            len_dict = {}
        llm = LLM(
            model=args.model_path,
            tensor_parallel_size=GPU_NUM,
            enable_lora=True if args.lora_path is not None else False,
            download_dir=args.model_cache_dir,
            tokenizer=tokenizer_name,
            dtype='auto',
            seed=args.seed + t,
            trust_remote_code=True,
            max_lora_rank=64,
            gpu_memory_utilization=args.gpu_memory_utilization,
            **len_dict
        )

        if args.lora_path:
            outputs = llm.generate(all_prompts, sampling_params, lora_request=LoRARequest("lora", 1, args.lora_path))
        else:
            outputs = llm.generate(all_prompts, sampling_params)

        for i, output in enumerate(outputs):
            for output_sample in output.outputs:
                raw_outputs[i].append(output_sample.text)

        del llm
        del sampling_params

    for idx, raw_output in enumerate(raw_outputs):
        all_prompts_outputs[all_prompts_idx[idx]].append(raw_output)

    for p, o, output_file in zip(all_prompts_unflatten, all_prompts_outputs, args.output_files):
        output_df = pd.DataFrame({'prompt': p, 'output': o})
        output_df.to_csv(output_file, index=False)


if __name__ == "__main__":
    args = get_args()

    if args.seed >= 0:
        set_seed(args.seed)
        random.seed(args.seed)

    logging.set_verbosity_info()

    main(args)
