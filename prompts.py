Self_Examination_PROMPTS = "Do you think your answer has solved the task? Please give a score for the completion of your answer. A score of 100 stands for extremely positive and 0 stands for extremely negative. You need only give me a score and do not explain about the score. The task is:[TASK] The result is:"

SELF_JUDGE_PROMPTS = "Q: [TASK] \nA: [ANSWER] \nIs the answer to this question correct?"

SELF_SCORE_PROMPTS = "Please help me evaluate the summary results of the following text. Only give a score from 1 to 10, without explanation.\nText: [TASK] \nSummary: [ANSWER]"

SELF_SCORE_PROMPTS_TRANSLATION = "Please help me evaluate the translation results. Only give a score from 1 to 10, without explanation.\nText: [TASK] \nTranslation: [ANSWER]"


CHAIN_OF_THOUGHT_PROMPTS = ". \nLet's think step by step"
TRANSLATION_PROMPTS = "Please help me translate the following Chinese text into English.\n\nText: [TASK]\n\nAnswer:"

MODEL_NAME = ["llama", 'google/flan-t5-large', 'google/flan-t5-xl', 'google/flan-t5-xxl', 'chatgpt']

MODEL_LLAMA = ["7B", "13B", "30B", "65B"]

OPENAI_API_KEY = 'sk-xxxxxx'  # replace it with your openai key here.