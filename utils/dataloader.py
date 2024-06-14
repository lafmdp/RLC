import os
import json
import jsonlines

import pandas as pd

from abc import abstractmethod
from typing import Dict, List

# from utils import get_formated_time, vote
# from config import MODEL_NAME, MODEL_LLAMA
# from logger import IntegratedLogger
from utils.logger import IntegratedLogger
from utils.utils import vote


class JSONFolderReader:
    def __init__(self, folder_path, file_type):
        self.folder_path = folder_path
        self.file_type = file_type
        self.json_files = [f for f in os.listdir(self.folder_path) if f.endswith(self.file_type)]

        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.json_files):
            raise StopIteration
        filename = self.json_files[self.index]

        self.index += 1
        return filename
    


class DataLoaderBase():
    def __init__(self, args, logger) -> None:
        self.args = args 

        self.logger = logger
        # self.file_feature_name_list = [args.model_name, args.ask_mode]
        self.file_type = '.json'
        self.dataset_path = self.args.root_dataset_path + self.args.dataset_name
        self.jsonloader = JSONFolderReader(self.dataset_path, self.file_type)


    def set_data_file_name(self, data_file_name_with_file_type: str) -> str:
        
        self._data_file_name = data_file_name_with_file_type.split('.')[0]


    @abstractmethod
    def load_data(self)-> List[Dict]:
        pass

    def save_data(self, data_to_save: List[Dict])->None:
        self.logger.save_data(data_to_save, self._data_file_name)
        

class DataLoaderBigBench(DataLoaderBase):
    """Load BigBench format data"""
    def __init__(self, args, logger) -> None:
        super().__init__(args, logger)
        self.cot_prompt_path = "BIG-Bench-Hard/cot-prompts/"
        self.file_with_options = {
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
            "temporal_sequences.json": "choice",
            "tracking_shuffled_objects_five_objects.json": "choice",
            "tracking_shuffled_objects_seven_objects.json": "choice",
            "tracking_shuffled_objects_three_objects.json": "choice",
        }
        

    def is_choice(self):
        if self._data_file_name+self.file_type in self.file_with_options.keys():
            return True
        return False
    
    def load_data(self,) -> List[Dict]:
        path = os.path.join(self.dataset_path, self._data_file_name+self.file_type)

        assert os.path.isfile(path), f"{path} must be a file!"

        with open(path) as text_file:
            text = json.load(text_file)

        if self.args.is_chain_of_thought and self.args.few_shot_cot > 0:
            cot_prompt_path = os.path.join(self.args.root_dataset_path + self.cot_prompt_path, self._data_file_name+".txt")
            with open(cot_prompt_path, "r") as cot_file:
                prompts = cot_file.read()
                prompts = prompts.split("-----\n")[1]
                prompts = prompts.split("\n\n")
            return text["examples"], prompts
        return text["examples"]


class DataLoaderCommonQA(DataLoaderBase):
    """Load CommonQA format data"""
    def __init__(self, args, logger) -> None:
        super().__init__(args, logger)
        self.dataset_path = self.args.root_dataset_path + self.args.dataset_name
        self.file_type = '.jsonl'
        self.jsonloader = JSONFolderReader(self.dataset_path, self.file_type)

    def load_data(self) -> List[Dict]:

        path = os.path.join(self.dataset_path, self._data_file_name+self.file_type)

        assert os.path.isfile(path), f"{path} must be a file!"
        
        data_list:List = []
        with open(path, "r") as f:
            for item in jsonlines.Reader(f):
                target = '(' + item['answerKey'] + ')'
                question = item['question']['stem']

                options = '\nOptions:'
                for choice in item['question']['choices']:
                    options += '\n(' + choice['label'] + ')' + choice['text']
                question = question + options
                data_list.append({"input":question, "target":target})

        return data_list



class DataLoaderTruthfulQA(DataLoaderBase):
    """Load TruthfulQA format data"""
    def __init__(self, args, logger) -> None:
        super().__init__(args, logger)
        self.dataset_path = self.args.root_dataset_path + self.args.dataset_name
        
        self.file_type = '.csv' 
        self.jsonloader = JSONFolderReader(self.dataset_path, self.file_type)

    def load_data(self) -> List[Dict]:
        path = os.path.join(self.dataset_path, self._data_file_name+self.file_type)

        assert os.path.isfile(path), f"{path} must be a file!"

        truthful_data_list = [] 
        truthful_all_data = pd.read_csv(path, skiprows=1)
        for _, row in truthful_all_data.iterrows():
            truthful_data_list.append({
                "input": row[2], 
                "Best Answer": row[3], 
                "Correct Answers": row[4], 
                "Incorrect Answers":row[5]
            })

        return truthful_data_list



class DataLoaderHumanAnnotations(DataLoaderBase):
    """Load human_annotations data"""
    def __init__(self, args, logger) -> None:
        super().__init__(args, logger)
        self.dataset_path = self.args.root_dataset_path + self.args.dataset_name
        
        self.SUMMARY_PROMPT = "Summarize the following article:\n"
        self.file_type = '.json' 
        self.jsonloader = JSONFolderReader(self.dataset_path, self.file_type)

    def load_data(self) -> List[Dict]:
        path = os.path.join(self.dataset_path, self._data_file_name+self.file_type)

        assert os.path.isfile(path), f"{path} must be a file!"

        human_annotation_data_list = []
        with open(path) as human_annotation_file:
            human_annotation_text = json.load(human_annotation_file)
        for index, content in human_annotation_text.items():
            vote_result = vote(
                [human_annotation_text[index]['annotators'][i]['best_summary'][0] for i in range(3)]
            )

            target = content[vote_result]['text']
            article = self.SUMMARY_PROMPT + content['article']
            human_annotation_data_list.append(
                {'input':article, 'target':target}
            )

        return human_annotation_data_list


if __name__ == '__main__':
    import argparse
    from typing import Dict

    parser = argparse.ArgumentParser()

    parser.add_argument("--result_path", type=str, default="./results/")
    parser.add_argument(
        "--dataset_name", 
        type=str, 
        help="The Specific dataset name:[TruthfulQA, CommonQA, BIG-Bench-Hard/bbh, human_annotations]",
        choices=["TruthfulQA", "CommonQA", "BIG-Bench-Hard/bbh", "human_annotations"],
        default="human_annotations"
    )
    parser.add_argument(
        "--root_dataset_path", 
        type=str, 
        help="root dataset", 
        default="./"
    )
    args = parser.parse_args()
    logger_args = {
    }

    logger = IntegratedLogger(record_param=[], log_root="logs", args=logger_args)

    dataloader = DataLoaderHumanAnnotations(args, logger)
    dataloader.set_data_file_name("bbc_human.json")
    data = dataloader.load_data()