import sys
from typing import cast
from PIL import Image
import os
import numpy as np
import argparse
import llama_cpp


def parse_args():
    parser = argparse.ArgumentParser(
        description="Display image in grayscale text using LLM"
    )
    parser.add_argument("image_path", type=str, help="Path to the image file")
    parser.add_argument(
        "--width", type=int, default=10, help="Width of the output image in characters"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Height of the output image in characters",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Path to the LLM model file",
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
    normalization_min = -0.1  # Configurable minimum value
    normalization_max = 1.5  # Configurable maximum value
    for char in mapping:
        normalized_value = (mapping[char] - min_of_mapping) / (
            max_of_mapping - min_of_mapping
        ) * (normalization_max - normalization_min) + normalization_min
        mapping[char] = max(0, min(1, normalized_value))  # Clamp to 0-1

    return mapping


def load_image(image_path, width, height):
    """
    Load an image and convert it to grayscale.
    Resize the image to fit the specified width and height.
    """
    image = Image.open(image_path)
    image = image.convert("L")  # Convert to grayscale
    if height is None:
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
    stdev = 0.1
    variance = stdev**2
    for c in s:
        target_value = target[target_idx]
        c_value = mapping[c]

        err = abs(target_value - c_value)
        # Use a normal distribution-like function
        matchness_value = np.exp(-((err**2) / (2 * variance)))
        total_matchness += matchness_value

        target_idx += 1

    # Calculate the average matchness
    avg_match = total_matchness / len(s)
    return avg_match


def infer(
    model: llama_cpp.Llama,
    prompt: str,
    grayscale_values: list[float],
    mapping: dict[str, float],
):
    # Not using the trie for now

    current_str = ""
    idx = 0

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

    return current_str


def main():
    args = parse_args()

    # Load the grayscale mapping
    mapping = load_grayscale_chars(args.mapping)

    # Load the image
    image = load_image(args.image_path, args.width, args.height)

    # Initialize the LLM model
    model = llama_cpp.Llama(
        model_path=args.model, logits_all=True, verbose=False, n_threads=12
    )

    # Get prompt
    with open(args.prompt, "r", encoding="utf-8") as f:
        prompt = f.read()

    print("Input prompt is:", repr(prompt))

    # Get the list of grayscale values to match
    image = image / 255.0  # Normalize the image to [0, 1]
    grayscale_values = image.flatten().tolist()

    out = infer(model, prompt, grayscale_values, mapping)

    # Split the output into rows and print
    rows = [out[i : i + args.width] for i in range(0, len(out), args.width)]
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
