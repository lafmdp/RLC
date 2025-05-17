
from typing import List, Dict
import argparse

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from accelerate import Accelerator
import torch as th

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
from utils.config import SELF_JUDGE_PROMPTS, CHAIN_OF_THOUGHT_PROMPTS, SELF_SCORE_PROMPTS, OPENAI_API_KEY, MODEL_NAME, MODEL_LLAMA
from chatbot.chatbot import BaseBot, ChatGPTBot, LLAMA, T5

from utils.preprocess import get_score


try:
    import evaluate
except ImportError:
    raise ImportError(
        "To run this example, please install the `evaluate` and `nltk` packages" "by running `pip install evaluate`"
    )

def get_evaluate_prompt(ask_mode: str, question: str, answer: str):
    if ask_mode == 'score':
        prompt = SELF_SCORE_PROMPTS.replace('[TASK]', question) \
                                    .replace('[ANSWER]', answer)

    else:
        assert False, "No such ask mode!"
    return prompt



meteor = evaluate.load("meteor")  # use meteor as the reward function

if __name__ == "__main__":
    accelerator = Accelerator()
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name", 
        type=str, 
        default='google/flan-t5-xl', 
        help="which model to load",
        choices=MODEL_NAME
    )
    parser.add_argument(
        "--ask_mode", 
        type=str, 
        help="how to ask for judgement",
        default='score'
    )
    parser.add_argument(
        "--is_chain_of_thought", 
        type=bool, 
        help="ask with chain of thought or not for judgement",
        default=False
    )
    parser.add_argument(
        "--answer_num", 
        type=int, 
        default=1,
    )
    args = parser.parse_args()

    config = TRLConfig(
        train=TrainConfig(
            seq_length=612,
            epochs=100,
            total_steps=14000,
            batch_size=12,
            checkpoint_interval=1000,
            eval_interval=50,
            pipeline="PromptPipeline",
            trainer="AcceleratePPOTrainer",
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

    MODELS: Dict[str, BaseBot] = {
        "llama": LLAMA,
        "google/flan-t5-large": T5,
        "google/flan-t5-xl": T5,
        "google/flan-t5-xxl": T5,
        "chatgpt": ChatGPTBot,
    }
    judge_model = MODELS[args.model_name](args) if args.model_name != "chatgpt" else MODELS[args.model_name](OPENAI_API_KEY)
    
    
    def reward_fn(samples: List[str], prompts: List[str], outputs: List[str]):
        scores = []
        for index in range(len(prompts)):
            question = prompts[index]
            answer = outputs[index]
            prompt = get_evaluate_prompt('score', question, answer)

            judge_score = judge_model.ask(prompt)
            judge_score = get_score(judge_score)
            scores.append(judge_score)
        return scores



    bleu_metric = evaluate.load("bleu")
    chrf_metric = evaluate.load("chrf")
    meteor_metric = evaluate.load("meteor")  # use meteor as the reward function
    bert_metric = evaluate.load("bertscore")

    def metric_fn(samples: List[str], prompts: List[str], outputs: List[str]) -> List[float]:
        """Compute COMET, BLEU and CHRF for evaluation"""

        original_summaries = [prompt_label[prompt.strip()] for prompt in prompts]

        bleu_score = bleu_metric.compute(
            predictions=[output.strip() for output in outputs],
            references=original_summaries,
        )["bleu"]
        chrf_score = chrf_metric.compute(
            predictions=[output.strip() for output in outputs],
            references=original_summaries,
        )["score"]
        meteor_scores = meteor_metric.compute(
            predictions=[output.strip() for output in outputs],
            references=original_summaries,
        )["meteor"]
        bert_scores = bert_metric.compute(
            predictions=[output.strip() for output in outputs],
            references=original_summaries,
            lang="en"
        )["precision"].mean()

        # TODO: This is needed since there seems to be a bug in the comet metric
        # that changes torch's determinism setting. Remove this once the bug is fixed.
        # Same issue as in `reward_fn`
        th.use_deterministic_algorithms(False, warn_only=True)

        # For corpus-level metrics, it's better to ignore the sentence-level scores
        print({"bleu": bleu_score, "chrf": chrf_score, "meteor": meteor_scores, "bert": bert_scores})
        return {"bleu": bleu_score, "chrf": chrf_score, "meteor": meteor_scores, "bert": bert_scores}
    
    dataset = load_dataset("cnn_dailymail", "3.0.0", cache_dir="data")


    # take 20,000 samples from the training set as prompts for training
    prompts = dataset["train"]["article"][0:20000]
    summaries = dataset["train"]["highlights"][0:20000]
    prompts = ["Summarize: " + prompt for prompt in prompts]

    # take 1,000 samples from the validation set as prompts for evaluation
    val_prompts = ["Summarize: " + prompt for prompt in dataset["validation"]["article"][0:1000]]
    val_summaries = dataset["validation"]["highlights"][0:1000]

    # make dictionary of prompts and labels to use for reward function
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
        )  # get prompt like trlx's prompt
        prompt_label[key.strip()] = summaries[i]

    for i in tqdm(range(len(val_prompts))):
        key = tokenizer.decode(
            tokenizer(val_prompts[i], truncation=True, max_length=max_length, add_special_tokens=False)["input_ids"],
            skip_special_tokens=True,
        )  # get prompt like trlx's prompt
        prompt_label[key.strip()] = val_summaries[i]

    trlx.train(
        reward_fn=reward_fn,
        prompts=prompts,
        metric_fn=metric_fn,
        eval_prompts=val_prompts,
        config=config,
    )
