# GPT-2-Text-Generation

This is a simple project with a .py file and Jupyter notebook to demonstrate how to use the GPT-2 language model for text generation using PyTorch Transformers.

## Requirements
- Python 3.x
- PyTorch
- pytorch_transformers
- numpy
- scipy

## Setup
```bash
pip install numpy scipy pytorch_transformers
```

## Usage
The notebook contains examples of:
1. Loading the pre-trained GPT-2 model and tokenizer
2. Converting text to tokens
3. Making predictions with the model
4. Generating multiple tokens for longer text completions

## Notes
- The model generates text based on statistical probability of next tokens
- Longer generations may become less coherent due to the token-by-token generation process
- Results can vary based on the specificity and structure of the input prompt
