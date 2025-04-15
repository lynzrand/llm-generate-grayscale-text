# Parts of this file comes from the LLaDA repository:, primarily the inference
# part. No intent to infringe copyright, but do note that not all of this file
# is under WTFPL.

import sys
import token
from typing import cast
from PIL import Image
import os
import numpy as np
import argparse
import llama_cpp

import torch
import numpy as np
import torch.nn.functional as F

import llada

from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.auto.tokenization_auto import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Display image in grayscale text using LLM"
    )
    parser.add_argument("image_path", type=str, help="Path to the image file")
    parser.add_argument(
        "--width", type=int, default=10, help="Width of the output image in characters"
    )
    parser.add_argument(
        "--mapping",
        type=str,
        default="grayscale.json",
        help="File containing the grayscale mapping for each character",
    )
    parser.add_argument(
        "--prompt", type=str, default="prompt.txt", help="File containing the prompt"
    )
    return parser.parse_args()


def load_grayscale_chars(mapping_file) -> dict[str, float]:
    """
    Load the grayscale mapping from a JSON file.
    The mapping should be a dictionary with characters as keys and their corresponding
    grayscale values as values.
    """
    import json

    with open(mapping_file, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    max_of_mapping = max(mapping.values())
    min_of_mapping = min(mapping.values())

    # Normalize the mapping values to be between a configurable range, then clamp to 0-1
    normalization_min = -0.4  # Configurable minimum value
    normalization_max = 2  # Configurable maximum value
    for char in mapping:
        normalized_value = (mapping[char] - min_of_mapping) / (
            max_of_mapping - min_of_mapping
        ) * (normalization_max - normalization_min) + normalization_min
        mapping[char] = max(0, min(1, normalized_value))  # Clamp to 0-1

    return mapping


def load_image(image_path, width):
    """
    Load an image and convert it to grayscale.
    Resize the image to fit the specified width and height.
    """
    image = Image.open(image_path)
    image = image.convert("L")  # Convert to grayscale
    aspect_ratio = image.width / image.height
    height = int(width / aspect_ratio)
    image = image.resize((width, height))
    return np.array(image)


def matchness(mapping: dict[str, float], target: list[float], start_idx: int, s: str):
    """
    Calculate how much a string matches the target grayscale values. A string
    might contain multiple characters, so we need to calculate the average
    grayscale matchness for the string.

    Return None if the string cannot be fully represented by the characters in
    the mapping.
    """
    if not all(c in mapping for c in s):
        return None
    if len(s) == 0:
        return None
    if len(s) > len(target) - start_idx:
        return None  # Not enough target values

    # Mapping contains only one char, so we can iterate over the string
    # and calculate the average matchness
    target_idx = start_idx
    total_matchness = 0
    for c in s:
        target_value = target[target_idx]
        c_value = mapping[c]

        err = abs(target_value - c_value)
        total_matchness += err**2  # Squared error

        target_idx += 1

    # Calculate the average matchness
    avg_match = total_matchness / len(s)
    return avg_match


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt,
    # our params
    image: np.ndarray,
    mappings: dict[str, float],
    # LLaDA params
    steps=128,
    temperature=0.0,
    cfg_scale=0.0,
    mask_id=126336,
):
    # Copyright (c) 2025 NieShenRuc
    # This function comes from the LLaDA codebase, with custom mods
    """
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).

        image: The image to match grayscale with
        mappings: The mapping of characters to grayscale values

        steps_per_block: Sampling steps per each block.

        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    """

    image_grayscale_list = list(image.flatten())

    # some utilities
    eot_token = 126081
    chars_target = len(image_grayscale_list)
    vocab: dict[str, int] = tokenizer.get_vocab()
    vocab_size = len(vocab)

    # Create arrays for vocabulary length and reverse index
    vocab_len = np.zeros(vocab_size, dtype=np.int32)
    rev_vocab_index = [""] * vocab_size

    # Populate arrays instead of dictionaries
    for tok_str, tok_int in vocab.items():
        # If the token is not contained in the mapping, skip it
        not_in_mapping = False
        for ch in tok_str:
            if ch not in mappings:
                not_in_mapping = True
                break
        if not_in_mapping:
            continue

        vocab_len[tok_int] = len(tok_str)
        rev_vocab_index[tok_int] = tok_str

    def iteratively_decode(logits):
        """
        Iteratively decode the current logits and decide which tokens to use
        based on the grayscale values and the mapping.
        """

        # The number of chars we have decoded
        decoded_len = 0
        token_idx = 0

        logits_len = logits.shape[1]

        # Iterative decode
        while decoded_len < chars_target and token_idx < logits_len:
            # Modify the current token based on the grayscale values
            for i in range(vocab_size):
                matchness_value = matchness(
                    mappings,
                    image_grayscale_list,
                    decoded_len,
                    rev_vocab_index[i],
                )
                if matchness_value is not None:
                    # Adjust the logits based on the matchness
                    logits[0, token_idx, i] -= matchness_value * 10
                else:
                    # If the token is not in the mapping, set its logit to -inf
                    logits[0, token_idx, i] = -np.inf

            # Use the token with highest logit as our current token for the next step
            this_token = int(torch.argmax(logits[token_idx]).item())
            # Update the decoded length
            decoded_len += vocab_len[this_token]
            # Update the token index
            token_idx += 1

        # Other tokens past the decoded length should be masks
        logits[:, token_idx:, :] = -np.inf
        # Set the logits for the mask token to 0
        logits[:, :, mask_id] = 0.0

    # Max gen length should be (size of image) characters.
    # min. 1 ch per token, plus LFs, so we be aggressive with the block length
    gen_length = image.size + image.shape[0]

    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(
        model.device
    )
    x[:, : prompt.shape[1]] = prompt.clone()

    prompt_index = x != mask_id

    # We have only one block, fortunately
    # This thing is all true
    block_mask_index = torch.full((1, gen_length), True, dtype=torch.bool).to(
        model.device
    )
    num_transfer_tokens = llada.get_num_transfer_tokens(block_mask_index, steps)
    for i in range(steps):
        print("Step", i + 1)

        mask_index = x == mask_id
        if cfg_scale > 0.0:
            un_x = x.clone()
            un_x[prompt_index] = mask_id
            x_ = torch.cat([x, un_x], dim=0)
            logits = model(x_).logits
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
        else:
            logits = model(x).logits

        print(logits.shape)

        iteratively_decode(logits)

        logits_with_noise = llada.add_gumbel_noise(logits, temperature=temperature)
        x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l

        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
        )  # b, l

        x0 = torch.where(mask_index, x0, x)
        confidence = torch.where(mask_index, x0_p, -np.inf)

        transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
        for j in range(confidence.shape[0]):
            _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
            transfer_index[j, select_index] = True
        x[transfer_index] = x0[transfer_index]

    return x


def infer(
    model: llama_cpp.Llama,
    prompt: str,
    grayscale_values: list[float],
    mapping: dict[str, float],
    width: int,
):
    # Not using the trie for now

    current_str = ""
    idx = 0
    row_remaining_width = width

    while idx < len(grayscale_values):
        print("Inferring for index:", idx)

        # Query the LLM for the next character
        model_resp = cast(
            llama_cpp.CreateCompletionResponse,
            model.create_completion(
                prompt + current_str, max_tokens=1, logprobs=20000, top_k=20000, top_p=1
            ),
        )
        choices0 = model_resp["choices"][0]
        if "logprobs" not in choices0:
            raise RuntimeError("No logprobs in response")
            return
        logprobs = cast(llama_cpp.CompletionLogprobs, choices0["logprobs"])
        if "top_logprobs" not in logprobs:
            raise RuntimeError("No top_logprobs in response")
            return
        top_logprobs_list = logprobs["top_logprobs"]
        if not top_logprobs_list:
            raise RuntimeError("No top logprobs in response")
            return
        top_logprobs = top_logprobs_list[0]
        if not top_logprobs:
            raise RuntimeError("No top logprobs in response")
            return
        # { token: logprob as np.float32 }

        # For each token in the top logprobs, calculate the adjusted probability
        adjusted_probs = {}

        adj_ruled_out_by_no_grayscale = 0
        for completion, logprob in top_logprobs.items():
            # reject tokens that don't fit in the row
            if len(completion) > row_remaining_width:
                continue

            # Calculate matchness for each completion
            matchness_value = matchness(mapping, grayscale_values, idx, completion)
            if matchness_value is None:
                adj_ruled_out_by_no_grayscale += 1
                continue

            # Adjust probability based on matchness with a stronger adjustment factor
            adjusted_prob = np.exp(logprob) * matchness_value**3
            adjusted_probs[completion] = adjusted_prob

        if not adjusted_probs:
            print("No adjusted probabilities found")
            print("Ruled out by no grayscale:", adj_ruled_out_by_no_grayscale)
            raise RuntimeError("No adjusted probabilities found")

        # Normalize the adjusted probabilities
        adj_prob_array = np.array(list(adjusted_probs.values()))
        adj_prob_array = adj_prob_array / np.sum(adj_prob_array)
        # Sample a token based on the adjusted probabilities
        sampled_token = np.random.choice(list(adjusted_probs.keys()), p=adj_prob_array)

        print("Sampled token:", repr(sampled_token))
        # Print the matchness of the sampled token
        sampled_matchness = matchness(mapping, grayscale_values, idx, sampled_token)
        print("Matchness of sampled token:", sampled_matchness)

        # Append the sampled token to the current string
        current_str += sampled_token
        idx += len(sampled_token)
        row_remaining_width -= len(sampled_token)
        if row_remaining_width <= 0:
            # Move to the next line
            current_str += "\n"
            row_remaining_width = width

    return current_str


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_args()

    # Load the grayscale mapping
    mapping = load_grayscale_chars(args.mapping)

    # Load the image
    image = load_image(args.image_path, args.width)

    # Get the list of grayscale values to match
    image = image / 255.0  # Normalize the image to [0, 1]
    grayscale_values = image.flatten().tolist()

    # Get prompt
    with open(args.prompt, "r", encoding="utf-8") as f:
        prompt = f.read()

    model = (
        AutoModel.from_pretrained(
            "GSAI-ML/LLaDA-8B-Instruct",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        .to(device)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True
    )

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    m = [
        {"role": "user", "content": prompt},
    ]
    prompt = tokenizer.apply_chat_template(
        m, add_generation_prompt=True, tokenize=False
    )

    input_ids = tokenizer(prompt)["input_ids"]
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    out = generate(
        model,
        tokenizer,
        input_ids,
        image,
        mappings=mapping,
    )

    print(
        tokenizer.batch_decode(out[:, input_ids.shape[1] :], skip_special_tokens=True)[
            0
        ]
    )


if __name__ == "__main__":
    main()
