# Can Large Language Models Capture Human Annotator Disagreements? [![Paper](:scroll:)](https://arxiv.org/abs/2506.19467)

This repository provides code and data accompanying our research paper exploring how Large Language Models (LLMs) capture human annotation disagreements, an often overlooked yet informative aspect of NLP annotation tasks.

## Motivation

Human annotation variation—such as disagreements among annotators—is widespread and informative, highlighting task subjectivity and ambiguity. Despite growing reliance on LLMs for automated annotation, current evaluations predominantly focus on majority-voted ground truth labels. This overlooks crucial insights encoded in annotation disagreements.

## Contributions

* **Comprehensive Evaluation**: We extensively evaluate how effectively LLMs can predict annotation disagreements without access to repeated human annotations.
* **Insights into LLM Limitations**: We demonstrate that LLMs struggle with accurately modeling annotation disagreements, a critical limitation not captured by standard evaluations.
* **Analysis of RL-based Reasoning Approaches**: While Reinforcement Learning with Verifiable Rewards (RLVR)-style reasoning usually enhances overall LLM performance, we found that it surprisingly decreases their effectiveness in disagreement modeling.

## Findings

Our findings underscore the importance of explicitly evaluating and improving LLMs' capabilities in modeling annotation disagreements.

## Repository Content

* Source code for experiments
* Datasets used for evaluation
* Analysis scripts for reproducing paper findings

Explore the code and data to reproduce our experiments and gain deeper insights into LLM capabilities and limitations regarding annotation disagreement.

For details, please refer to our paper \[link to paper].
---
## Repository Structure

- `prompts/`: Contains various prompt templates (in .txt format) used for LLM input, covering different datasets and settings.
    - `anno` vs. `dist`: sampling-based distribution vs. verbalized distribution
    - `base_cot` vs. `base` vs. `r1`: RLHF LLM with / without CoT vs. RLVR LLM (R1 series)
    - `ghc`, `helpsteer2`, `pos`, `neg` , `amb`: Gab Hate Corpus, HelpSteer2, GoEmotion Positive, Negative, and Ambiguous
    - `few` vs. `zero`: few- or zero-shot prompting
- `test_data/`: Contains CSV files with test data for evaluating model performance.
    - `dis` vs. `rnd`: the HighVar and Random subset (details see paper)
- `train_data/`: training data for ModernBERT / DeBERTa fine-tuning.
- `llm_generations.zip`: **LLM Generations in our experiments. Users of this repo may want to analyze them directly without run all 100+ sets of LLM inference on their own.**
    - `deepseek_v3_r1_few_shot/`: LLM generations for DeepSeek models in few-shot settings.
    - `deepseek_v3_r1_zero_shot/`: LLM generations for DeepSeek models in zero-shot settings.
    - `other_models_few_shot_sampled_distribution/`: Generations from other models (few-shot, sampled distribution).
    - `other_models_few_shot_verbalized_distribution/`: Generations from other models (few-shot, verbalized distribution).
    - `other_models_zero_shot/`: Generations from other models in zero-shot settings.
    - `force_budget`: Results with forced longer reasoning chain.

---
## Run inference with vLLM

**Example:**
```shell
# Install vLLM
pip install -U vllm

MAX_NEW_TOKEN=4096
ALL_PROMPT_TEMPLATE=(
    "./prompts/anno_base_amb_few.txt"
    "./prompts/anno_base_amb_few.txt"
    "./prompts/anno_base_cot_ghc_few.txt"
)
ALL_INPUT_PATH=(
    "./test_data/test_goe_ambiguous_dis.csv"
    "./test_data/test_goe_ambiguous_rnd.csv"
    "./test_data/test_ghc_rnd.csv"
)
ALL_OUTPUT_PATH=(
    "prompts_annotation_base_amb_dis_llama_8b.csv"
    "prompts_annotation_base_amb_rnd_llama_8b.csv"
    "prompts_annotation_base_ghc_rnd_llama_8b.csv"
)
MODEL_PATH="allenai/Llama-3.1-Tulu-3.1-8B"

SAMPLE_NUM=10
SAMPLE_ROUND=1
TEMPERATURE=0.7
GPU_UTILITY=0.8
HUGGINGFACE_CACHE="cache_dir"


python vllm_inference.py --max_new_token ${MAX_NEW_TOKEN} --format --times $SAMPLE_ROUND --n $SAMPLE_NUM --model_cache_dir $HUGGINGFACE_CACHE --temperature ${TEMPERATURE} --model_path ${MODEL_PATH} --prompt_templates "${ALL_PROMPT_TEMPLATE[@]}" --prompt_templates "${ALL_INPUT_PATH[@]}" --output_files "${ALL_OUTPUT_PATH[@]}" --load_tokenizer
```

This script will construct prompts for LLMs using test sentences and prompt template of different settings. It accept multiple datasets and prompt templates to save LLM loading time. Importantly, only experiments using the same LLM and inference setting (e.g., temperature, sampling numbers) can be handled by a single command as above.

**Arguments to customize:**
- `--prompt_templates`: Paths to prompt template files (can specify multiple, e.g., "./prompts/anno_base_amb_few.txt" ...).
- `--input_files`: Paths to input data files (CSV or JSONL, can specify multiple, e.g., "./test_data/test_goe_ambiguous_dis.csv" ...).
- `--output_files`: Paths where you want to save the inference results (can specify multiple, e.g., "prompts_annotation_base_amb_dis_llama_8b.csv" ...).
- `--model_path`: Path or identifier of the LLM model to use (e.g., "allenai/Llama-3.1-Tulu-3.1-8B").
- `--max_new_token`: Maximum number of new tokens to generate per sample.
- `--times`: Number of rounds to repeat sampling (for multiple generations).
- `--n`: Number of samples to generate per input. Here we generate 10 samples to draw the sampling-based distribution.
- `--temperature`: Sampling temperature for generation. We use temperature 0.7 in this example to enable sampling
- `--model_cache_dir`: Directory for HuggingFace model cache.
- `--load_tokenizer`: Flag to load the tokenizer from huggingface.

---
## Run inference with LiteLLM and API call

Here’s a guideline for using `litellm_eval.py`, in a style consistent with your vLLM section:


**Example:**
```shell
# Install LiteLLM and required dependencies
pip install -U litellm

ALL_PROMPT_TEMPLATE=(
    "./prompts/dist_base_amb_few.txt"
    "./prompts/dist_base_cot_ghc_few.txt"
)
ALL_INPUT_PATH=(
    "./test_data/test_goe_ambiguous_dis.csv"
    "./test_data/test_ghc_rnd.csv"
)
ALL_OUTPUT_PATH=(
    "prompts_distribution_base_amb_dis_r1.csv"
    "prompts_distribution_base_ghc_rnd_r1.csv"
)
MODEL_NAME="deepseek-reasoner-original"
export DEEPSEEK_API_KEY="your_openai_api_key"

python litellm_eval.py \
    --llm ${MODEL_NAME} \
    --prompt_templates "${ALL_PROMPT_TEMPLATE[@]}" \
    --dataset_files "${ALL_INPUT_PATH[@]}" \
    --output_files "${ALL_OUTPUT_PATH[@]}"
```

This script will construct prompts for LLMs using test sentences and prompt templates of different settings. It accepts multiple datasets and prompt templates to save API call overhead. Only experiments using the same LLM and inference setting (e.g., temperature, sampling numbers) can be handled by a single command as above.

**Arguments to customize:**
- `--llm`: Name of the model to use via API.
- `--prompt_templates`: Paths to prompt template files (can specify multiple, e.g., "./prompts/anno_base_amb_few.txt" ...).
- `--input_files`: Paths to input data files (CSV or JSONL, can specify multiple, e.g., "./test_data/test_goe_ambiguous_dis.csv" ...).
- `--output_files`: Paths where you want to save the inference results (can specify multiple, e.g., "prompts_annotation_base_amb_dis_litellm.csv" ...).


---
## Train ModernBERT (or DeBERTa-v3)

**Example:**
```shell
ALL_EVAL_DATA=(
    "test_data/test_ghc_dis.csv"
    "test_data/test_ghc_rnd.csv"
)
python train_eval_modernbert.py --model_path answerdotai/ModernBERT-large --model_save_dir ghc_modernbert --train_data train_data/ghc_train.csv --train_text_column Text --label_column prob_positive --eval_data "${ALL_EVAL_DATA[@]}" --eval_text_column Text --output_name modernbert
```
This script trains ModernBERT on train_data/ghc_train.csv and evaluate it on HighVar and Random subsets. The output will correspondingly be saved in `test_data/test_ghc_dis_modernbert.csv` and `test_data/test_ghc_rnd_modernbert.csv`

**Arguments to customize:**
- `--model_path`: HuggingFace model name or path (e.g., answerdotai/ModernBERT-large or a local directory).
- `--model_save_dir`: Directory to save the fine-tuned model.
- `--train_data`: Path to the training CSV file.
- `--train_text_column`: Name of the column containing input text in the training data.
- `--label_column`: Name of the column containing the target label in the training data.
- `--eval_data`: One or more paths to evaluation CSV files (can be specified as an array).
- `--eval_text_column`: Name of the column containing input text in the evaluation data.
- `--output_name`: Prefix for the output prediction files.

---

**Paper citation**:

```
@misc{ni2025largelanguagemodelscapture,
      title={Can Large Language Models Capture Human Annotator Disagreements?}, 
      author={Jingwei Ni and Yu Fan and Vilém Zouhar and Donya Rooein and Alexander Hoyle and Mrinmaya Sachan and Markus Leippold and Dirk Hovy and Elliott Ash},
      year={2025},
      eprint={2506.19467},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2506.19467}, 
}
```
