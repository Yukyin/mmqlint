from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-8B-Instruct", trust_remote_code=True)

def render(messages):
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False, #Key point
        add_generation_prompt=True,
    )
