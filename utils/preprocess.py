import time
import re
import openai
import numpy as np


def get_score(sentence: str):
        score = None
        try:
            score = "".join(list(filter(str.isdigit, sentence)))
            if score == '':
                 score = '0'
            score = np.clip(float(score), 0, 10)
        except openai.error.RateLimitError:
            print("Reach rate limits, sleep 10s")
            
        except Exception as e:
            print(e, score)

        return score


def normalizeString(s):
    s = s.lower().strip()
    s = re.sub(r"([.!?])", r" \1", s)  
    s = re.sub(r"[^a-zA-Z]+", r" ", s)  
    s = re.sub(r'[\s]+', " ", s)  
    return s
