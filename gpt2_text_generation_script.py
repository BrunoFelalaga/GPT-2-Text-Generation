import torch
import sys
from pytorch_transformers import GPT2Tokenizer, GPT2LMHeadModel

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LENGTH = 50  # Maximum length of generated text
MODEL_NAME = 'gpt2'

def generate_text(prompt, max_new_tokens=10):
    """
    Generate text continuation from a prompt using GPT-2
    
    Args:
        prompt (str): Input text to continue
        max_new_tokens (int): Number of new tokens to generate
        
    Returns:
        str: Generated text including prompt
    """
    # Encode prompt
    indexed_tokens = tokenizer.encode(prompt)
    tokens_tensor = torch.tensor([indexed_tokens]).to(DEVICE)
    
    # Generate tokens
    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(tokens_tensor)
            predictions = outputs[0]
            predicted_index = torch.argmax(predictions[0, -1, :]).item()
            tokens_tensor = torch.cat((tokens_tensor, 
                                    torch.tensor([[predicted_index]]).to(DEVICE)), 
                                    dim=1)
    
    # Decode and return text
    return tokenizer.decode(tokens_tensor[0].tolist(), skip_special_tokens=True)

# 4. Add example usage function
def run_examples():
    """Run example text generations with different prompts"""
    examples = [
        "What mammal has over 40,000 muscles in one body part?",
        "the oldest university in Illinois is",
        "the first president of the USA was President"
    ]
    
    for prompt in examples:
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generate_text(prompt)}")
        print("-" * 50)

run_examples()
