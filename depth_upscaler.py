import argparse
from copy import deepcopy

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def depth_upscale_model(
    base_model_name_or_path, models_with_layers, save=False, device_map="auto"
):
    # Load base model and tokenizer
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)

    # Initialize an empty list for layers to copy
    layers_to_copy = []

    for model_name_or_path, top_layers, bottom_layers in models_with_layers:
        # Load the model to copy layers from
        model_to_copy_from = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
        )

        # Add the specified top and bottom layers to the list
        layers_to_copy.extend(deepcopy(model_to_copy_from.model.layers[:top_layers]))
        layers_to_copy.extend(
            deepcopy(model_to_copy_from.model.layers[-bottom_layers:])
        )

        # Copy the embedding and lm_head layers from the first model only
        if not hasattr(base_model.model, "embed_tokens"):
            base_model.model.embed_tokens = deepcopy(
                model_to_copy_from.model.embed_tokens
            )
        if not hasattr(base_model, "lm_head"):
            base_model.lm_head = deepcopy(model_to_copy_from.lm_head)

    # Create a new Sequential module with the copied layers
    base_model.model.layers = nn.ModuleList(layers_to_copy)

    # Update base model configuration
    config = AutoConfig.from_pretrained(
        base_model_name_or_path,
        num_hidden_layers=len(base_model.model.layers),
    )
    base_model.config = config

    # Save model if requested
    if save:
        save_path = f"{base_model_name_or_path}_depth_upscaled"
        base_model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"Model and tokenizer saved to {save_path}")

    return base_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Depth upscale a base language model by adding specified top and bottom layers from other models."
    )
    parser.add_argument(
        "--base_model_name_or_path",
        type=str,
        required=True,
        help="The name or path of the base pre-trained model.",
    )
    parser.add_argument(
        "--models_with_layers",
        nargs="+",
        required=True,
        help="List of models with the number of top and bottom layers to copy, e.g., model1:5:5 model2:4:4.",
    )
    parser.add_argument(
        "--save", action="store_true", help="Save the modified model and tokenizer."
    )
    parser.add_argument(
        "--device_map", type=str, default="auto", help="The device map for the model."
    )

    args = parser.parse_args()

    # Parse the models_with_layers argument into a list of tuples
    models_with_layers = [
        (model.split(":")[0], int(model.split(":")[1]), int(model.split(":")[2]))
        for model in args.models_with_layers
    ]

    depth_upscale_model(
        args.base_model_name_or_path, models_with_layers, args.save, args.device_map
    )
