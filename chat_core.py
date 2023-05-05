import torch


def process_chat(model, tokenizer, device, params):
    """
    in/out process for HuggingFace models
    :param params:
    :param model:
    :param tokenizer:
    :param device:
    :return:
    """
    stream_interval = 1

    prompt = params["prompt"]

    temperature = float(params.get("temperature", 1.0))
    max_new_tokens = int(params.get("max_new_tokens", 256))
    context_len = int(params.get("context_len", 1024))
    stop_strs = params.get("stop_strs", None)
    force_set_bos_token_id = params.get("force_set_bos_token_id", None)
    force_set_eos_token_id = params.get("force_set_eos_token_id", None)
    use_top_k_sampling = params.get("use_top_k_sampling", False)
    top_k_value = params.get("top_k_value", 10)

    use_bos_for_input = params.get("use_bos_for_input", False)

    if force_set_bos_token_id:
        tokenizer.bos_token_id = force_set_bos_token_id  # <= acceptable

    if force_set_eos_token_id:
        # tokenizer.eos_token_id = force_set_eos_token_id <= buggy
        stop_token_ids = params.get("stop_ids", [force_set_eos_token_id])
    else:
        stop_token_ids = params.get("stop_ids", [tokenizer.eos_token_id])

    l_prompt = len(prompt)
    if use_bos_for_input:
        input_ids = [tokenizer.bos_token_id] + tokenizer(prompt).input_ids
        l_prompt -= len(tokenizer.decode([tokenizer.bos_token_id]))
    else:
        input_ids = tokenizer(prompt).input_ids

    output_token_ids = list(input_ids)

    max_src_len = context_len - max_new_tokens - 8
    input_ids = input_ids[-max_src_len:]
    with torch.no_grad():
        for i in range(max_new_tokens):
            if i == 0:
                out = model(input_ids=torch.as_tensor([input_ids], device=device), use_cache=True)
                logits = out.logits
                past_key_values = out.past_key_values
            else:
                out = model(
                    input_ids=torch.as_tensor([[token_id]], device=device),
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                logits = out.logits
                past_key_values = out.past_key_values

            last_token_logits = logits[0][-1]

            if device == "mps":
                last_token_logits = last_token_logits.float().to("cpu")

            def top_k_sampling(logits, k=top_k_value):
                top_k = torch.topk(logits, k)
                top_k_indices = top_k.indices
                top_k_values = top_k.values
                probabilities = torch.softmax(top_k_values, dim=-1)
                choice = torch.multinomial(probabilities, num_samples=1)
                token_id = int(top_k_indices[choice])
                return token_id

            if temperature < 1e-4:
                token_id = int(torch.argmax(last_token_logits))
            else:
                # Adjust with Softmax with temperature
                # very nice article below
                # https://shivammehta25.github.io/posts/temperature-in-language-models-open-ai-whisper-probabilistic-machine-learning/
                probabilities = torch.softmax(last_token_logits / temperature, dim=-1)

                if use_top_k_sampling:
                    token_id = top_k_sampling(last_token_logits)
                else:
                    token_id = int(torch.multinomial(probabilities, num_samples=1))

            output_token_ids.append(token_id)

            if token_id in stop_token_ids:
                stopped = True
            else:
                stopped = False

            if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
                output = tokenizer.decode(output_token_ids, skip_special_tokens=True)
                if stop_strs:
                    for stop_str in stop_strs:
                        if stop_str:

                            pos = output.rfind(stop_str, l_prompt)
                            is_stop_str_found = (pos != -1)
                            if is_stop_str_found:
                                output = output[:pos]
                                stopped = True

                yield output

            if stopped:
                break

    del past_key_values
