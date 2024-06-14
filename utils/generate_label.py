import json
import os

from utils import vote
from evaluate import *


question_type = {
    "choice": [
        '(a)', '(b)', '(c)', '(d)', '(e)', 
        '(f)', '(g)', '(h)', '(i)', '(j)', 
        '(k)', '(l)', '(m)', '(n)', '(o)', 
        '(p)', '(q)', '(r)', '(s)', '(t)', 
        '(u)', '(v)', '(w)', '(x)', '(y)', '(z)'
    ],
    "yes-no": ['yes', 'no'],
    "boolean": ['true', 'false'],
    "valid-or-not": ['valid', 'invalid'],
}


def generate_label(file_path, file2read):
    question_answer_list = []
    file_path = os.path.join(file_path, file2read)
    with open(file_path, 'r') as file_:
        text = json.load(file_)

        for t in text:
            question = t['input']                
            results = t['results']
            target = t['target']

            results_only_answer_list = []
            for result in results:
                result = result.lower().replace("\n", "").replace(".", "")
                if len(result) > 0 and not result.isspace():
                    result = result.split()[-1]
                results_only_answer_list.append(result)

                
            vote_result = vote(results_only_answer_list)
            question_answer_list.append({"input":question, "vote_result": vote_result, "target": target})
                

    file_name = "./datasets/BBH_generated_label/" + file2read.split('.')[0]
    with open(file_name+".json", "w") as f:
            jd = json.dumps(question_answer_list)
            print(jd, file=f)



if __name__ == '__main__':


    file2read_list = [
        "web_of_lies.json",
        "sports_understanding.json",
        "hyperbaton.json",
        "tracking_shuffled_objects_five_objects.json",
        "formal_fallacies.json",
        "date_understanding.json",
        "logical_deduction_five_objects.json",
        "causal_judgement.json",
        "dyck_languages.json",
        "penguins_in_a_table.json",
        "logical_deduction_three_objects.json",
        "reasoning_about_colored_objects.json",
        "multistep_arithmetic_two.json",
        "logical_deduction_seven_objects.json",
        "boolean_expressions.json",
        "navigate.json",
        "word_sorting.json",
        "tracking_shuffled_objects_seven_objects.json",
        "temporal_sequences.json",
        "salient_translation_error_detection.json",
        "movie_recommendation.json",
        "object_counting.json",
        "disambiguation_qa.json",
        "tracking_shuffled_objects_three_objects.json",
        "ruin_names.json",
        "snarks.json",
        "geometric_shapes.json",
    ]

    file_path = 'logs/2023-05-14_14-48-57&model=google/flan-t5-large&ask_mode=judge-direct&llm_generate_mode=multinomial_sampling'
    for file2read in file2read_list:
        print(file2read)
        generate_label(
            file_path,
            file2read
        )
