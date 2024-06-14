import time

class BaseBot(object):
    def __init__(self):
        pass

    def ask(self, question) -> str:
        pass


import openai
class ChatGPTBot(BaseBot):
    def __init__(self, api_key, chatgpt=False):
        super().__init__()
        openai.api_key = api_key
        self.chatgpt = chatgpt

    def ask(self, question) -> str:
        if self.chatgpt:
            messages = [{"role": "user", "content": question}]
            rsp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages
            )
            res = rsp.get("choices")[0]["message"]["content"]
        else:
            response = openai.Completion.create(
                model="text-davinci-003",
                # prompt="I am a highly intelligent question answering bot. If you ask me a question that is rooted in truth, I will give you the answer. If you ask me a question that is nonsense, trickery, or has no clear answer, I will respond with \"Unknown\".\n\nQ: What is human life expectancy in the United States?\nA: Human life expectancy in the United States is 78 years.\n\nQ: Who was president of the United States in 1955?\nA: Dwight D. Eisenhower was president of the United States in 1955.\n\nQ: Which party did he belong to?\nA: He belonged to the Republican Party.\n\nQ: What is the square root of banana?\nA: Unknown\n\nQ: How does a telescope work?\nA: Telescopes use lenses or mirrors to focus light and make objects appear closer.\n\nQ: Where were the 1992 Olympics held?\nA: The 1992 Olympics were held in Barcelona, Spain.\n\nQ: How many squigs are in a bonk?\nA: Unknown\n\nQ: Where is the Valley of Kings?\nA:",
                prompt=f"{question}",
                temperature=1,
                max_tokens=1000,
                top_p=1,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                stop=["Answer:"]
            )
            res = response['choices'][0]['text']

        return res


from transformers import T5Tokenizer, T5ForConditionalGeneration
class T5(BaseBot):

    def __init__(self, args):
        super().__init__()
        self.args = args

        print(f"The model is {args.model_name}")
        self.tokenizer = T5Tokenizer.from_pretrained(args.model_name,)

        self.model = T5ForConditionalGeneration.from_pretrained(
            args.model_name,
            device_map="auto",
        )


    def ask(self, question):

        input_ids = self.tokenizer(question, return_tensors="pt").input_ids.to("cuda")

        outputs = self.model.generate(input_ids, do_sample=True, max_new_tokens=100, top_k=50, top_p=0.95, num_return_sequences=self.args.answer_num)
        if self.args.answer_num == 1:
            outputs = outputs[0]
            return self.tokenizer.decode(outputs, skip_special_tokens=True)
        else:
            return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

from transformers import GPT2Tokenizer, GPT2LMHeadModel
class GPT2(BaseBot):

    def __init__(self, args):
        super().__init__()
        self.args = args

        print(f"The model is {args.model_name}")
        self.tokenizer = GPT2Tokenizer.from_pretrained(args.model_name,)

        self.model = GPT2LMHeadModel.from_pretrained(
            args.model_name,
            device_map="auto",
        )

    def ask(self, question):

        input_ids = self.tokenizer(question, return_tensors="pt").input_ids.to("cuda")

        outputs = self.model.generate(input_ids, do_sample=True, max_new_tokens=100, top_k=50, top_p=0.95)
        if self.args.answer_num == 1:
            outputs = outputs[0]
            return self.tokenizer.decode(outputs, skip_special_tokens=True)
        else:
            return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

from transformers import BartTokenizer, BartModel
class BART(BaseBot):

    def __init__(self, args):
        super().__init__()
        self.args = args

        print(f"The model is {args.model_name}")
        self.tokenizer = BartTokenizer.from_pretrained(args.model_name,)

        self.model = BartModel.from_pretrained(
            args.model_name,
            device_map="auto",
        )

    def ask(self, question):

        input_ids = self.tokenizer(question, return_tensors="pt").input_ids.to("cuda")

        outputs = self.model.generate(input_ids, do_sample=True, max_new_tokens=100, top_k=50, top_p=0.95)
        if self.args.answer_num == 1:
            outputs = outputs[0]
            return self.tokenizer.decode(outputs, skip_special_tokens=True)
        else:
            return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)


if __name__ == '__main__':
    bot = BART()
    q = "1 + 1 = ?"
    prompts = "Answer the following question:"
    t0 = time.time()
    for i in range(10):
        print(bot.ask(prompts + q))

    print(time.time() - t0)


