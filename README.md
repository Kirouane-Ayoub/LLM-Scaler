# LLMs Scaler
This repository contains the code for the LLMs Scaler project , it is a tool for scaling down large language models (LLMs) to smaller sizes or upscaling them to larger sizes.


## Dependencies
To install the dependencies, run the following command:
```bash
pip install -r requirements.txt
```


## Downscaler Usage Example :
To use the script from the command line:
```bash
python downscaler.py --model_name_or_path HuggingFaceTB/SmolLM-135M-Instruct --top_layers 5 --bottom_layers 5 --save
```


## Depth-up Scaler Usage Example (Multiple Models):
To use the script from the command line with multiple models:
```bash
python depth_upscaler.py --base_model_name_or_path HuggingFaceTB/SmolLM-135M-Instruct --models_with_layers model_path_or_id1:5:5 model_path_or_id2:4:4 --save
```
