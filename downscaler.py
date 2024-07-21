import argparse

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def nparams(model):
    """Calculate the total number of model parameters"""
    nparams = sum(p.numel() for p in model.parameters())
    return nparams


def downscale_model(
    model_name_or_path, top_layers, bottom_layers, save=False, device_map="auto"
):
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
    )
    n_params = nparams(model)
    print(f"Original model has {n_params} parameters")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    # Modify model layers
    layers = model.model.layers
    model.model.layers = layers[:top_layers] + layers[-bottom_layers:]

    # Update model configuration
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_hidden_layers=len(model.model.layers),
    )
    model.config = config
    n_params = nparams(model)
    print(f"Downscaled model has {n_params} parameters")

    # Save model if requested
    if save:
        model.save_pretrained(f"{model_name_or_path}_downscaled")
        tokenizer.save_pretrained(f"{model_name_or_path}_downscaled")
        print(f"Model and tokenizer saved to {model_name_or_path}_downscaled")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Downscale a language model by keeping specified top and bottom layers."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="The name or path of the pre-trained model.",
    )
    parser.add_argument(
        "--device_map", type=str, default="auto", help="The device map for the model."
    )
    parser.add_argument(
        "--top_layers",
        type=int,
        required=True,
        help="The number of top layers to keep.",
    )
    parser.add_argument(
        "--bottom_layers",
        type=int,
        required=True,
        help="The number of bottom layers to keep.",
    )
    parser.add_argument(
        "--save", action="store_true", help="Save the modified model and tokenizer."
    )

    args = parser.parse_args()

    downscale_model(
        args.model_name_or_path,
        args.top_layers,
        args.bottom_layers,
        args.save,
        args.device_map,
    )
