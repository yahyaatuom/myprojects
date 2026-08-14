import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

p_prompt = "This movie was absolutely fantastic and I loved every moment of it."
n_prompt = "This movie was terrible and i hated every moment of it"

p_token = "great"
n_token = "terrible"

pos_inputs = tokenizer(p_prompt,return_tensors="pt",padding=True)
neg_inputs = tokenizer(n_prompt,return_tensors="pt",padding=True)

pos_token_id = tokenizer.encode(p_token)[0]
neg_token_id = tokenizer.encode(n_token)[0]

activations = {}

def get_activation_hook(layer_name):
    def hook(module, input, output):
        activations[layer_name] = output[0].detach()
        print(f"Captured activations at: {layer_name}: shape {output[0].shape}")
    return hook

layer_to_steer = 6
hook_handle = model.transformer.h[layer_to_steer].register_forward_hook(
    get_activation_hook(f"layer_{layer_to_steer}")
)

with torch.no_grad():
    activations.clear()
    pos_outputs = model(**pos_inputs)
    pos_activations = activations["layer_6"].clone()
    print(f"Positive prompt tokens: {pos_inputs['input_ids'].shape[1]} tokens")
    print(f"Positive activations shape: {pos_activations.shape}")

with torch.no_grad():
    activations.clear()
    neg_outputs = model(**neg_inputs)
    neg_activations = activations["layer_6"].clone()

pos_last_token = pos_activations[:,-1,:]
neg_last_token = neg_activations[:,-1,:]

delta = pos_last_token - neg_last_token
print(f"Delta Shape: {delta.shape}")

delta = delta / torch.norm(delta)

def create_steering_vector(delta,strength=1.0):
    return delta*strength

steering_strength = 0.5
steering_vector = create_steering_vector(delta,steering_strength)
print(f"Steering vector shape: {steering_vector.shape}")

def apply_steering_hook(layer_name,steering_vector):
    def hook(module,input,output):
        hidden_states = output[0]
        hidden_states[:,-1,:] = hidden_states[:,-1,:] + steering_vector
        return (hidden_states,)+ output[1:]
    return hook

steering_hook = model.transformer.h[layer_to_steer].register_forward_hook(
    apply_steering_hook(f"layer_{layer_to_steer}",steering_vector.to(model.device))
)
test_prompt = "I just watched this new film and i think it was"
test_inputs = tokenizer(test_prompt,return_tensors="pt")

with torch.no_grad():
    steered_output = model.generate(
        **test_inputs,
        max_new_tokens=20,
        temperature=0.7,
        do_sample=True,
        pad_token_id = tokenizer.eos_token_id
    )
steered_text = tokenizer.decode(steered_output[0],skip_special_tokens=True)
print(f"Steered output: {steered_text}")

steering_hook.remove()

with torch.no_grad():
    normal_output = model.generate(
        **test_inputs,
        max_new_tokens=20,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
normal_text = tokenizer.decode(normal_output[0], skip_special_tokens=True)
print(f"Normal output: {normal_text}")

def get_token_probabilities(model, tokenizer, prompt, target_tokens):
    """Get probabilities of specific tokens given a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]  # Last token's logits
        probs = torch.softmax(logits, dim=-1)
    
    # Get probabilities for our target tokens
    results = {}
    for token in target_tokens:
        token_id = tokenizer.encode(token)[0]
        results[token] = probs[token_id].item()
    
    return results

# Test with a neutral prompt
neutral_prompt = "This movie was"
target_tokens = ["great", "terrible"]

# Without steering
print("Without steering:")
probs_normal = get_token_probabilities(model, tokenizer, neutral_prompt, target_tokens)
for token, prob in probs_normal.items():
    print(f"  P('{token}') = {prob:.4f}")

# With steering (we need to reapply the hook)
steering_hook = model.transformer.h[layer_to_steer].register_forward_hook(
    apply_steering_hook(f"layer_{layer_to_steer}", steering_vector.to(model.device))
)

print("\nWith steering:")
probs_steered = get_token_probabilities(model, tokenizer, neutral_prompt, target_tokens)
for token, prob in probs_steered.items():
    print(f"  P('{token}') = {prob:.4f}")

# Clean up
steering_hook.remove()