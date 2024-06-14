import json
import os
import re
import random
import pandas as pd
import numpy as np

from utils import get_formated_time
from utils import vote
from preprocess import get_score
from evaluate import *
# from utils.utils import get_formated_time
# from utils.utils import vote
# from utils.preprocess import get_score
# from utils.evaluate import *


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

def voting_result(judge_yes_result, target, judge_times, vote_accuracy_after_judge_yes):
    if len(judge_yes_result) != 0:
        judge_times += 1
        judge_yes_vote_result = vote(judge_yes_result)
        if target in judge_yes_vote_result:
            # print(judge_yes_result, t['target'].lower())
            vote_accuracy_after_judge_yes += 1
    return judge_times, vote_accuracy_after_judge_yes
                

def calculate_LLM_accuracy_BBH(file_path: str, file_with_options):
    file_list = []
    for file_ in os.listdir(file_path):
        if file_.endswith('.json') and file_ != "parameter.json":
            file_list.append(file_)
            
            
    dataset_name_list = []
    predict_list = []
    judge_list = []
    answer_with_option_list = []
    vote_accuracy_list = []
    judge_vote_accuracy_random_list = []
    judge_vote_accuracy_direct_list = []
    judge_vote_accuracy_delete_list = []
    judge_vote_accuracy_direct_average_list = []
    

    for saved_file in file_list:

        print(saved_file)
        generate_data_path = os.path.join(file_path, saved_file)
        with open(generate_data_path, 'r') as file_:
            text = json.load(file_)
            
            predict_accuracy = 0.0
            judge_accuracy = 0.0
            answer_with_option = 0.0
            vote_accuracy = 0.0
            vote_accuracy_after_judge_yes_random = 0.0
            vote_accuracy_after_judge_yes_direct = 0.0
            vote_accuracy_after_judge_yes_delete = 0.0
            vote_accuracy_average_direct = 0.0

            judge_times_random = 0
            judge_times_direct = 0
            judge_times_delete = 0


            for t in text:
                
                target = t['target'].lower()
                results = t['results']
                judge_results = t['judge_results']
                question = t['input'].lower()

                results_only_answer_list = []
                ask_times = len(results)
                for result in results:
                    result = result.lower().replace("\n", "").replace(".", "")
                    results_only_answer_list.append(result)


                judge_yes_result_random = []
                judge_yes_result_direct_again = []
                judge_yes_result_delete_again = []
                for i, result in enumerate(results_only_answer_list):
                    
                    if target in result:
                        predict_accuracy += 1
                        if 'yes' in judge_results[i].split(" [AGAIN] ")[0].lower():
                            judge_accuracy += 1
                            
                    else:
                        if 'no' in judge_results[i].split(" [AGAIN] ")[0].lower():
                            judge_accuracy += 1 


                    if 'yes' in judge_results[i].split(" [AGAIN] ")[0].lower():
                        judge_yes_result_random.append(result)
                        judge_yes_result_delete_again.append(result)
                        judge_yes_result_direct_again.append(result)
                    else:
                        answer_again = judge_results[i].split(" [AGAIN] ")[-1]
                        answer_again = answer_again.lower().replace("\n", " ").replace(".", "")
                        judge_yes_result_delete_again.append(answer_again)

                        answer_again = judge_results[i].split(" [AGAIN] ")[-2]
                        answer_again = answer_again.lower().replace("\n", " ").replace(".", "")
                        judge_yes_result_direct_again.append(answer_again)

                        if saved_file in file_with_options.keys():
                            options = question_type[file_with_options[saved_file]]
                            other_answer = options.copy()
                            if result in options:
                                other_answer.remove(result)
                            judge_yes_result_random.append(random.choice(other_answer))
                        else:
                            answer_again = judge_results[i].split(" [AGAIN] ")[-2]
                            answer_again = answer_again.lower().replace("\n", " ").replace(".", "")
                            judge_yes_result_random.append(answer_again)


                if saved_file in file_with_options.keys():
                    if file_with_options[saved_file] == "choice":
                        options_regex = r"\([a-z]\)"
                        options = re.findall(options_regex, question)
                    elif file_with_options[saved_file] == "yes-no":
                        options = ['yes', 'no']
                    elif file_with_options[saved_file] == "valid-or-not":
                        options = ['valid', 'invalid']
                    elif file_with_options[saved_file] == "boolean":
                        options = ['true', 'false']

                    count_dict = {option: 0 for option in options}

                    for r in results_only_answer_list:
                        for option in options:
                            if option in r:
                                count_dict[option] += 1

                    max_count = max(count_dict.values())
                    most_frequent = [k for k, v in count_dict.items() if v == max_count]

                    if len(most_frequent) > 1:
                        vote_result = random.choice(most_frequent)
                    else:
                        vote_result = most_frequent[0]
                
                    if target in vote_result:
                        vote_accuracy += 1
                else:
                    vote_list = []
                    for r in results_only_answer_list:
                        if target in r:
                            vote_list.append(1)
                        else:
                            vote_list.append(0)
                    if sum(vote_list) > ask_times//2:
                        vote_accuracy += 1

                judge_times_random, vote_accuracy_after_judge_yes_random = voting_result(judge_yes_result_random, target, judge_times_random, vote_accuracy_after_judge_yes_random)
                judge_times_direct, vote_accuracy_after_judge_yes_direct = voting_result(judge_yes_result_direct_again, target, judge_times_direct, vote_accuracy_after_judge_yes_direct)
                judge_times_delete, vote_accuracy_after_judge_yes_delete = voting_result(judge_yes_result_delete_again, target, judge_times_delete, vote_accuracy_after_judge_yes_delete)

                for r in judge_yes_result_direct_again:
                    if target in r:
                        vote_accuracy_average_direct += 1

            text_length = len(text)
            predict_accuracy = predict_accuracy / text_length/ask_times
            judge_accuracy = judge_accuracy / text_length/ask_times
            answer_with_option = answer_with_option / text_length/ask_times
            vote_accuracy = vote_accuracy / text_length
            vote_accuracy_after_judge_yes_random = vote_accuracy_after_judge_yes_random / judge_times_random
            vote_accuracy_after_judge_yes_direct = vote_accuracy_after_judge_yes_direct / judge_times_direct
            vote_accuracy_after_judge_yes_delete = vote_accuracy_after_judge_yes_delete / judge_times_delete
            vote_accuracy_average_direct = vote_accuracy_average_direct / text_length/ask_times

            dataset_name_list.append(saved_file.split('.')[0])
            predict_list.append(predict_accuracy)
            judge_list.append(judge_accuracy)
            answer_with_option_list.append(answer_with_option)
            vote_accuracy_list.append(vote_accuracy)
            judge_vote_accuracy_random_list.append(vote_accuracy_after_judge_yes_random)
            judge_vote_accuracy_direct_list.append(vote_accuracy_after_judge_yes_direct)
            judge_vote_accuracy_delete_list.append(vote_accuracy_after_judge_yes_delete)
            judge_vote_accuracy_direct_average_list.append(vote_accuracy_average_direct)
            print(saved_file, text_length, judge_times_delete,text_length == judge_times_delete)

                
    dataframe = pd.DataFrame({
        'Dataset':dataset_name_list, 
        'Predict Accuracy':predict_list, 
        'Judge Arruracy': judge_list, 
        'Vote Arruracy': vote_accuracy_list,
        'Vote Arruracy with Judge Random': judge_vote_accuracy_random_list,
        'Vote Arruracy with Judge Direct': judge_vote_accuracy_direct_list,
        'Vote Arruracy with Judge Delete': judge_vote_accuracy_delete_list,
        'Vote Average Arruracy with Judge Direct': judge_vote_accuracy_direct_average_list,
    })

    time_ = get_formated_time()
    print(time_)
    pattern = r"(flan-t5-[\w-]+)&ask_mode"
    match = re.search(pattern, file_path)
    model_str = match.group(1)
    dataframe.to_csv("./results/calculate_result/calculate_" +model_str + "_"+ time_ + ".csv",index=False,sep=',')



def calculate_LLM_accuracy_human_annotations(file_path: str, saved_file_name):

    dataset_name_list = []
    bleu_max_list = []
    bleu_avg_list = []

    rouge_1_max_list = []
    rouge_2_max_list = []
    rouge_l_max_list = []

    rouge_1_avg_list= []
    rouge_2_avg_list = []
    rouge_l_avg_list = []

    bert_max_list = []
    bert_avg_list = []
    

    for saved_file in saved_file_name:

        print(saved_file)
        generate_data_path = os.path.join(file_path, saved_file)
        with open(generate_data_path, 'r') as file_:
            text = json.load(file_)

            bleu_max_score = 0 
            bleu_avg_score = 0 

            rouge_1_max_score = 0 
            rouge_2_max_score = 0 
            rouge_l_max_score = 0  

            rouge_1_avg_score = 0 
            rouge_2_avg_score = 0 
            rouge_l_avg_score = 0 

            bert_max_score = 0 
            bert_avg_score = 0 


            for t in text:
                target = t['target'] 
                results = t['results']
                judge_result = t['judge_results']

                judge_only_score_list = []
                for j in judge_result:
                    j = get_score(j)
                    judge_only_score_list.append(j)
                
                print(judge_only_score_list)

                results_score_temp = {
                    'bleu':[], 
                    'rouge-1':[],
                    'rouge-2':[],
                    'rouge-l':[],
                    'bert':[]
                }

                for i, result in enumerate(results):
                    
                    bleu_score = bleu_evaluate(result, target)
                    rouge_score = rouge_evaluate(result, target)
                    bert_score = bert_evaluate(result, target)
                    rouge_1_score_p = rouge_score["rouge-1"]['p']
                    rouge_2_score_p = rouge_score["rouge-2"]['p']
                    rouge_l_score_p = rouge_score["rouge-l"]['p']

                    results_score_temp['bleu'].append(bleu_score)
                    results_score_temp['rouge-1'].append(rouge_1_score_p)
                    results_score_temp['rouge-2'].append(rouge_2_score_p)
                    results_score_temp['rouge-l'].append(rouge_l_score_p)
                    results_score_temp['bert'].append(bert_score)

                bleu_avg_score += np.mean(results_score_temp['bleu'])
                rouge_1_avg_score += np.mean(results_score_temp['rouge-1'])
                rouge_2_avg_score += np.mean(results_score_temp['rouge-2'])
                rouge_l_avg_score += np.mean(results_score_temp['rouge-l'])
                bert_avg_score += np.mean(results_score_temp['bert'])


            text_length = len(text)
            bleu_max_score = bleu_max_score / text_length
            bleu_avg_score = bleu_avg_score / text_length

            rouge_1_max_score = rouge_1_max_score / text_length
            rouge_2_max_score = rouge_2_max_score / text_length
            rouge_l_max_score = rouge_l_max_score / text_length

            rouge_1_avg_score = rouge_1_avg_score / text_length
            rouge_2_avg_score = rouge_2_avg_score / text_length
            rouge_l_avg_score = rouge_l_avg_score / text_length

            bert_max_score = bert_max_score / text_length
            bert_avg_score = bert_avg_score / text_length

            dataset_name_list.append(saved_file.split('.')[0])
            bleu_max_list.append(bleu_max_score)
            bleu_avg_list.append(bleu_avg_score)

            rouge_1_max_list.append(rouge_1_max_score)
            rouge_2_max_list.append(rouge_2_max_score)
            rouge_l_max_list.append(rouge_l_max_score)

            rouge_1_avg_list.append(rouge_1_avg_score)
            rouge_2_avg_list.append(rouge_2_avg_score)
            rouge_l_avg_list.append(rouge_l_avg_score) 

            bert_max_list.append(bert_max_score)
            bert_avg_list.append(bert_avg_score)
            print(saved_file, text_length)

                
    dataframe = pd.DataFrame({
        'Dataset':dataset_name_list, 

        'bleu_avg':bleu_avg_list, 
        'bleu_max':bleu_max_list, 

        'rouge_1_avg':rouge_1_avg_list, 
        'rouge_1_max':rouge_1_max_list, 

        'rouge_2_avg':rouge_2_avg_list, 
        'rouge_2_max':rouge_2_max_list, 

        'rouge_l_avg':rouge_l_avg_list, 
        'rouge_l_max':rouge_l_max_list, 

        'bert_avg':bert_avg_list, 
        'bert_max':bert_max_list, 
    })

    dataframe.to_csv("./results/calculate_result/calculate_" + get_formated_time() + ".csv",index=False,sep=',')

    test_generate(
        dataset_name_list, 
        bleu_avg_list,
        rouge_1_avg_list,
        rouge_2_avg_list,
        rouge_l_avg_list,
        bert_avg_list
    )


def test_generate(
        dataset_name_list, 
        bleu_avg_list,
        rouge_1_avg_list,
        rouge_2_avg_list,
        rouge_l_avg_list,
        bert_avg_list
    ):
    bleu_avg_list.append(mean(bleu_avg_list))
    dataframe = pd.DataFrame({
        'Dataset':dataset_name_list, 

        'bleu':bleu_avg_list, 

        'rouge_1':rouge_1_avg_list, 

        'rouge_2':rouge_2_avg_list, 

        'rouge_l':rouge_l_avg_list, 

        'bert':bert_avg_list, 
    })

    dataframe.to_csv("./results/calculate_result/calculate_" + get_formated_time() + ".csv",index=False,sep=',')


if __name__ == '__main__':


    file_with_options = {
        "boolean_expressions.json": "boolean",
        "causal_judgement.json": "yes-no",
        "date_understanding.json": "choice",
        "disambiguation_qa.json": "choice",
        "formal_fallacies.json": "valid-or-not",
        "geometric_shapes.json": "choice",
        "hyperbaton.json": "choice",
        "logical_deduction_five_objects.json": "choice",
        "logical_deduction_seven_objects.json": "choice",
        "logical_deduction_three_objects.json": "choice",
        "movie_recommendation.json": "choice",
        "navigate.json": "yes-no",
        "penguins_in_a_table.json": "choice",
        "reasoning_about_colored_objects.json": "choice",
        "ruin_names.json": "choice",
        "salient_translation_error_detection.json": "choice",
        "snarks.json": "choice",
        "sports_understanding.json": "yes-no",
        "temporal_sequences.json": "choice",
        "tracking_shuffled_objects_five_objects.json": "choice",
        "tracking_shuffled_objects_seven_objects.json": "choice",
        "tracking_shuffled_objects_three_objects.json": "choice",
        "web_of_lies.json": "yes-no",
    }

    file_path = "logs/2023-05-15_23-27-20&model=google/flan-t5-large&ask_mode=judge-direct&llm_generate_mode=multinomial_sampling"

    calculate_LLM_accuracy_BBH(
        file_path,
        file_with_options
    )
