# GPT-2-Text-Generation

# GPT-2 Text Generation Example

This Jupyter notebook demonstrates how to use the GPT-2 language model for text generation using PyTorch Transformers.

## Features
- Load and use pre-trained GPT-2 model
- Generate text completions for given prompts
- Examples of single-token and multi-token generation
- Demonstration of model evaluation mode and inference

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

## Examples
- Single token completion for "What mammal has over 40,000 muscles in one body part?"
- Multi-token generation for "the oldest university in Illinois is"
- Historical fact completion for "the first president of the USA was President"

## Notes
- The model generates text based on statistical probability of next tokens
- Longer generations may become less coherent due to the token-by-token generation process
- Results can vary based on the specificity and structure of the input prompt
