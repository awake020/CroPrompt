from vllm import LLM, SamplingParams

import transformers


class Llama3Model():
    def __init__(self, model='/home/share/models/Meta-Llama-3-70B-Instruct', tensor_parallel_size=4):
        self.llm = LLM(model=model, tensor_parallel_size=tensor_parallel_size)
        self.tokenizer = self.llm.get_tokenizer()

    def get_prompt_text(self, input):
        prompt_text = self.tokenizer.apply_chat_template(
            input,
            tokenize=False
        )
        return prompt_text


    def chat_complete(self, input, temperature=0, max_tokens=4096, use_tqdm=False, **kwargs):
        outputs = self.llm.generate(
            input, 
            SamplingParams(
                temperature=temperature, 
                top_p=0.9, 
                max_tokens=max_tokens, 
                stop_token_ids=[self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]),
                use_tqdm=use_tqdm
        )
        return [x.outputs[0].text for x in outputs]