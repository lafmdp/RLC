import os
from typing import List, Dict
import argparse

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from accelerate import Accelerator

import trlx
from trlx.data.configs import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)
from trlx.models.modeling_ppo import PPOConfig
import trlx.utils.logging as logging
from prompts import CHAIN_OF_THOUGHT_PROMPTS, OPENAI_API_KEY
from chatbot.chatbot import BaseBot, ChatGPTBot, T5, BART, GPT2
from utils.utils import *


try:
    import evaluate
except ImportError:
    raise ImportError(
        "To run this example, please install the `evaluate` and `nltk` packages" "by running `pip install evaluate`"
    )

def save(path):
    if not os.path.exists(path):
        os.makedirs(path)
    trainer.save_pretrained(path)

meteor = evaluate.load("meteor")  

if __name__ == "__main__":
    accelerator = Accelerator()
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name", 
        type=str, 
        default='google/flan-t5-xl', 
        help="which model to load",
    )
    parser.add_argument(
        "--ask_mode", 
        type=str, 
        help="how to ask for judgement",
        default='stadnard_answer_reward'
    )
    parser.add_argument(
        "--is_chain_of_thought", 
        type=bool, 
        help="ask with chain of thought or not for judgement",
        default=False
    )
    parser.add_argument(
        "--bbh_set", 
        type=str, 
        help="BBH set name",
        default="date_understanding"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=1000,
    )

    parser.add_argument(
        "--cache_dir", 
        type=str, 
        default=".cache/",
    )
    parser.add_argument(
        "--answer_num", 
        type=int, 
        default=1,
    )

    args = parser.parse_args()

    config = TRLConfig(
        train=TrainConfig(
            seq_length=512,
            epochs=100,
            total_steps=8000,
            batch_size=12,
            checkpoint_interval=1000,
            eval_interval=50,
            pipeline="PromptPipeline",
            trainer="AcceleratePPOTrainer",
            seed = args.seed
        ),
        model=ModelConfig(
            model_path=args.model_name,
            model_arch_type="seq2seq", 
            num_layers_unfrozen=2,
        ),
        tokenizer=TokenizerConfig(
            tokenizer_path=args.model_name,
            truncation_side="right",
        ),
        optimizer=OptimizerConfig(
            name="adamw",
            kwargs={
                "lr": 1.0e-4,
                "betas": [0.9, 0.999],
                "eps": 1.0e-8,
                "weight_decay": 1.0e-6,
            },
        ),
        scheduler=SchedulerConfig(
            name="cosine_annealing",
            kwargs={
                "T_max": 10000,
                "eta_min": 1.0e-6,
            },
        ),
        method=PPOConfig(
            name="PPOConfig",
            num_rollouts=512,
            chunk_size=12,
            ppo_epochs=4,
            init_kl_coef=0.1,
            target=6,
            horizon=10000,
            gamma=0.99,
            lam=0.95,
            cliprange=0.2,
            cliprange_value=0.2,
            vf_coef=1.0,
            scale_reward=None,
            ref_mean=None,
            ref_std=None,
            cliprange_reward=10,
            gen_kwargs={
                "max_new_tokens": 100,
            },
            gen_experience_kwargs={
                "max_new_tokens": 100,
                "do_sample": True,
                "temperature": 1.0,
                "top_k": 50,
                "top_p": 0.95,
            },
        ),
    )

    logger = logging.get_logger()
    logger.info(f"BBH set name: {args.bbh_set}")

    MODELS: Dict[str, BaseBot] = {
        "google/flan-t5-small": T5,
        "google/flan-t5-base": T5,
        "google/flan-t5-large": T5,
        "google/flan-t5-xl": T5,
        "google/flan-t5-xxl": T5,
        "facebook/bart-large": BART,
        "facebook/bart-base": BART,
        "gpt2-large": GPT2,
        "gpt2-medium": GPT2,
        "chatgpt": ChatGPTBot,
    }
    judge_model = MODELS[args.model_name](args) if args.model_name != "chatgpt" else MODELS[args.model_name](OPENAI_API_KEY)
    
    
    def reward_fn(samples: List[str], prompts: List[str], outputs: List[str]):
        targets = [prompt_label[prompt.strip()].lower() for prompt in prompts]

        scores = []
        for index in range(len(prompts)):
            assert len(prompts) == len(outputs), f"Different length :target:{targets}, outputs:{outputs}"
            question = prompts[index]
            question = question.split("Tell me the options directly, excluding the content of the options")[0]

            if args.ask_mode == 'standard_answer_reward':
                answer = outputs[index].lower()
                if targets[index] in answer:
                    scores.append(1)
                else:
                    scores.append(0)

            elif args.ask_mode == 'judge-direct':
                answer = outputs[index]
                prompt = get_judge_prompt(args.ask_mode, question, answer)

                judge_score = judge_model.ask(prompt)
               
                if len(judge_score) == 0:
                    print(f"judge_score is empty: {judge_score}")
                    scores.append(0)
                    continue

                judge_answer = judge_score.lower()
                if 'yes' in judge_answer:
                    scores.append(1)
                else:
                    scores.append(0)
        return scores




    def metric_fn(samples: List[str], prompts: List[str], outputs: List[str]) -> List[float]:
        """Compute COMET, BLEU and CHRF for evaluation"""
        targets = [prompt_label[prompt.strip()].lower() for prompt in prompts]
        scores = 0.0
        for index in range(len(prompts)):
            answer = outputs[index].lower()
            if targets[index] in answer:
                scores += 1.0
        return {"score": scores/len(prompts)}

    dataset = load_dataset("lukaemon/bbh", args.bbh_set)

    prompts = dataset["test"]["input"]
    targets = dataset["test"]["target"]
    if args.is_chain_of_thought:
        prompts = [prompt+CHAIN_OF_THOUGHT_PROMPTS for prompt in prompts]
    else:
        prompts = [prompt for prompt in prompts]

    tokenizer = AutoTokenizer.from_pretrained(config.model.model_path)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "right"
    tokenizer.sep_token = "<sep>"
    prompt_label = {}
    max_length = config.train.seq_length - config.method.gen_kwargs["max_new_tokens"]

    for i in tqdm(range(len(prompts))):
        key = tokenizer.decode(
            tokenizer(prompts[i], truncation=True, max_length=max_length, add_special_tokens=False)["input_ids"],
            skip_special_tokens=True,
        ) 
        prompt_label[key.strip()] = targets[i]

    length = len(prompts)
    train_prompts = prompts[:int(0.8*length)]

    save_args = {'ask_mode':args.ask_mode, 'save_dataset_name': args.bbh_set, 'save_model_name': args.model_name,'save_is_chain_of_thought': args.is_chain_of_thought, 'seed':args.seed}
    trainer = trlx.train(
        reward_fn=reward_fn,
        prompts=train_prompts,
        metric_fn=metric_fn,
        eval_prompts=prompts,
        config=config,
        save_args=save_args,
    )

    file_to_save = "-".join([str(_arg) for _arg in save_args.values()])
    path = "./results/rl_train/" + file_to_save
    save(path)


