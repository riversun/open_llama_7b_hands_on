from transformers import set_seed
from chat_core import process_chat
from model_loader_for_llama import load_hf_model

# Fix seed value for verification.
seed_value = 42
set_seed(seed_value)

# model_path = "/home/user/sandbox/open_llama_7b_preview_200bt/open_llama_7b_preview_200bt_transformers_weights"
model_path ="/home/user/sandbox/open_llama_7b_preview_300bt/open_llama_7b_preview_300bt_transformers_weights"
device = "cuda"

model, tokenizer, device = load_hf_model(model_path=model_path)

chatPrompt = None
chat_mode = False  # I am studying a stable prompt for two-party conversations.

while True:
    user_input = input("YOU: ")
    if user_input.lower() == "exit":
        break

    if chat_mode:
        chatPrompt.add_requester_msg(user_input)
        chatPrompt.add_responder_msg(None)
        prompt = chatPrompt.create_prompt()
        stop_str = chatPrompt.get_stop_str()

    else:
        prompt = user_input
        stop_str = None

    params = {
        "prompt": prompt,
        "temperature": 0.5,
        "max_new_tokens": 256,
        "context_len": 1024,
        "use_top_k_sampling": True,
        "use_bos_for_input": False, # 200bt model needs bos token in the input tokens.
        "force_set_bos_token_id": 1,
        "force_set_eos_token_id": 2,
        "stop_strs": stop_str,
    }

    generator = process_chat(model, tokenizer, device, params)

    prev = ""

    for index, response_text in enumerate(generator):

        if index == 0:
            print("AI : ", end="", flush=True)

        if chat_mode:
            response_text = response_text[chatPrompt.get_skip_len():].strip()
        else:
            # response_text = response_text[len(prompt):].strip()
            pass

        updated_text = response_text[len(prev):]

        print(updated_text, end="", flush=True)

        prev = response_text

    print()

    if chat_mode:
        chatPrompt.set_responder_last_msg(response_text.strip())
