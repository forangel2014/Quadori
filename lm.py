import os
import openai
import numpy as np
import torch
import math
from transformers import pipeline
from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM
    
class LLM():
    
    def __init__(self, model, device):
        
        self.model = model
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.device = device

        if model == 'gpt-j':
            print("---------loading model-------------")
            tokenizer_path = "/data/sunwangtao/.cache/huggingface/hub/models--EleutherAI--gpt-j-6B/snapshots/6e35e2148e92edf096e94d39ac2b98ad59e25975"
            model_path = "/data/sunwangtao/.cache/huggingface/hub/models--EleutherAI--gpt-j-6B/snapshots/b71ae8bc86cac13154e03e92b5855203086b722e"
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            self.local_lm = AutoModelForCausalLM.from_pretrained(
                model_path,
                revision="float16",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            ).to(self.device)
            print("---------finish loading model-------------")
    
        elif model == 'flan-t5':
            print("---------loading model-------------")
            tokenizer_path = "/data/sunwangtao/.cache/huggingface/hub/models--google--flan-t5-small/snapshots/f6b63ff0230b8e19027b922964cab639c1c6da9c"
            model_path = "/data/sunwangtao/.cache/huggingface/hub/models--google--flan-t5-small/snapshots/f6b63ff0230b8e19027b922964cab639c1c6da9c"
            self.tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-xl')
            self.local_lm = AutoModelForSeq2SeqLM.from_pretrained(
                'google/flan-t5-xl'
            ).to(self.device)
            self.ending_idx = 3
            print("---------finish loading model-------------")
    
        elif model == 'chatglm':
            print("---------loading model-------------")
            tokenizer_path = "/data/sunwangtao/.cache/huggingface/hub/models--google--flan-t5-small/snapshots/f6b63ff0230b8e19027b922964cab639c1c6da9c"
            model_path = "/data/sunwangtao/.cache/huggingface/hub/models--google--flan-t5-small/snapshots/f6b63ff0230b8e19027b922964cab639c1c6da9c"
            self.tokenizer = AutoTokenizer.from_pretrained('THUDM/chatglm-6b', trust_remote_code=True)
            self.local_lm = AutoModel.from_pretrained(
                'THUDM/chatglm-6b',
                trust_remote_code=True
            ).half().to(self.device)
            self.ending_idx = 4
            print("---------finish loading model-------------")
            
    def __call__(self, prompt, k=1, temp=1, stop=["\n"]):
        if self.model == 'gpt-j' or self.model == 'flan-t5' or self.model == 'chatglm':
            return self.local(prompt, k, temp, stop)
        elif self.model == 'gpt3':
            return self.gpt3(prompt, k, temp, stop)
        elif self.model == 'chatgpt':
            return self.chatgpt(prompt, k, temp, stop)
        else:
            raise Exception("Model not support")
    
    def local(self, prompt, k, temp, stop):
        # stop = ["."]
        ending_idx = self.tokenizer.encode(stop[0])[0]
        inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        response = self.local_lm.generate(inputs, 
                                          max_length=inputs.shape[1]+100, 
                                          early_stopping=True,
                                          eos_token_id=self.ending_idx, 
                                          pad_token_id=self.ending_idx)
        response = response[0][inputs.shape[1]:]
        response_text = self.tokenizer.decode(response).strip()
        return response_text
        # generator = pipeline("text-generation", tokenizer=self.tokenizer, model=self.local_lm)
        # generated_text = generator(
        #     prompt,
        #     max_length=50,
        #     early_stopping=True,
        #     stop_token="\n",
        #     num_return_sequences=1
        # )

        # print(generated_text[0]["generated_text"])
    
    def gpt3(self, prompt, k, temp, stop):
        response = openai.Completion.create(
                        model="text-davinci-003",
                        prompt=prompt,
                        temperature=temp,
                        max_tokens=100,
                        top_p=1,
                        frequency_penalty=0.0,
                        presence_penalty=0.0,
                        stop=stop,
                        n=k
                    )
        return [choice["text"] for choice in response["choices"]]
    
    def chatgpt(self, prompt, k, temp, stop=["\n"]):
        messages = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo", 
                messages=messages,
                temperature=temp,
                max_tokens=100,
                top_p=1,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                stop=stop,
                n=k
            )
        
        return [choice["message"]["content"] for choice in response["choices"]]
    
