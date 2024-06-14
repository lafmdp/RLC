import openai, time

class gpt_agent():

    def __init__(self, api_key:str):
        openai.api_key = api_key
        self.api_key = api_key
        self.ask_call_cnt = 0
        self.ask_call_cnt_sup = 3

    def get_embedding(self, text) -> list:
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-ada-002"
        )
        embeddings = response['data'][0]['embedding']

        return embeddings

    def ask(self, question) -> str:
        res = None
        self.ask_call_cnt = self.ask_call_cnt + 1
        if self.ask_call_cnt > self.ask_call_cnt_sup:
            print("======> Achieve call count limit, Return!")
            self.ask_call_cnt = 0
            return res

        messages = [{"role": "user", "content": question}]
        try:
            rsp = openai.ChatCompletion.create(
                # model="gpt-3.5-turbo",
                model="gpt-4",
                messages=messages,
                temperature=0.7
            )
            res = rsp.get("choices")[0]["message"]["content"]
            self.ask_call_cnt = 0
        except openai.error.AuthenticationError as e:
            print("======> openai.error.AuthenticationError", e)
        except openai.error.RateLimitError as e:
            """
            no need to exit!
            if "quota" in str(e.error):
                print("openai.error.RateLimitError", e)
                exit(0)
            else:
                print("Achieve ChatGPT rate limit, sleep!")
                time.sleep(10)
                return self.ask(question)
            """
            print(f"======> {self.api_key} <===== \nAchieve ChatGPT rate limit, sleep!", e)
            if "quota" in str(e.error):
                pass
            else:
                time.sleep(10)
                return self.ask(question)
        except openai.error.ServiceUnavailableError:
            print('======> Service unavailable error: will retry after 10 seconds')
            time.sleep(10)
            return self.ask(question)
        except Exception as e:
            print("======> Exception occurs!", e)

        return res


