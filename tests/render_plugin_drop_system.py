from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-8B-Instruct", trust_remote_code=True)

def render(messages):
    messages = [m for m in messages if m.get("role") != "system"]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
