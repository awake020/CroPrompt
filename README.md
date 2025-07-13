# CroPrompt: Cross-task Interactive Prompting for Zero-shot Spoken Language Understanding

Repository for ICASSP2025 [CroPrompt: Cross-task Interactive Prompting for Zero-shot Spoken Language Understanding](https://ieeexplore.ieee.org/document/10889329)

```bash
# method example
# for openai model
python method_api/croprompt.py -mn gpt-3.5-turbo -op out.jsonl -t 0
# for llama model
python method_open_source/croprompt.py -op out.json -t 0


# self-consistency example
cd out/llama3-8b/croprompt-self-consistency/
python essemble.py


# evaluation example
python snips_evaluation.py out/llama3-8b/croprompt-self-consistency/snips_out_croprompt-consistency.jsonl
700
```
