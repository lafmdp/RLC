import time
import random
import numpy as np
import re

from typing import List
from collections import Counter

from prompts import SELF_JUDGE_PROMPTS, SELF_SCORE_PROMPTS, SELF_SCORE_PROMPTS_TRANSLATION


def get_formated_time(time_2_get=None):

    if time_2_get is None:
        time_2_get = time.time()

    localtime = time.localtime(time.time())

    return time.strftime('%Y:%m:%d %H:%M:%S', localtime) 


def vote(candidate: List):

    result = Counter(candidate)
    
    if (np.array(list(result.values())) == 1).all():
         return random.choice(list(result.keys()))
    
    else:
        sorted_result = sorted(result.items(), key=lambda x:x[1], reverse=True)
        return sorted_result[0][0]


def get_option_content(question: str, answer: str):
    
    findword = fr"\({answer}\)\s*(.*)"
    
    
    
    try:
        match = re.search(findword, question)
        if match:
            result_with_option = match.group(0)
        else:
            result_with_option = None
        
        
    except:
        print("Re error with findword:", findword)
        result_with_option = None
    return result_with_option



def get_judge_prompt(ask_mode: str, question: str, answer: str):
    
    if "\nOptions" in question:

        
        
        if ask_mode == 'judge-with-content': 
            result_with_option = get_option_content(question, answer)
            if result_with_option is None:
                print("Error: result_with_option is None")
                result_with_option = answer
            
            prompt = SELF_JUDGE_PROMPTS.replace('[TASK]', question) \
                                        .replace('[ANSWER]', result_with_option)
        
        
        
        elif ask_mode == "judge-direct":
            prompt = SELF_JUDGE_PROMPTS.replace('[TASK]', question) \
                                        .replace('[ANSWER]', answer)

        
        
        elif ask_mode == 'judge-without-options':
            prompt = SELF_JUDGE_PROMPTS.replace('[TASK]', question.split('\nOptions')[0]) \
                                        .replace('[ANSWER]', result_with_option[3:])
    
    else:

        if ask_mode == 'judge-direct':
            prompt = SELF_JUDGE_PROMPTS.replace('[TASK]', question) \
                                        .replace('[ANSWER]', answer) 
        
        
        
        
        elif ask_mode == 'score':
            prompt = SELF_SCORE_PROMPTS.replace('[TASK]', question) \
                                        .replace('[ANSWER]', answer)

        elif ask_mode == 'score_translation':
            prompt = SELF_SCORE_PROMPTS_TRANSLATION.replace('[TASK]', question) \
                                        .replace('[ANSWER]', answer)
        else:
            assert False, "No such ask mode!"
    return prompt
