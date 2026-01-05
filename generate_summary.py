# Use a pipeline as a high-level helper
from transformers import pipeline
import torch


user_input = ''
with open('text/conv2.txt', 'r') as f:
    user_input = f.read()

# Check if CUDA is available and set device accordingly
device = 0 if torch.cuda.is_available() else -1
print(f"Using device: {'GPU' if device == 0 else 'CPU'}")

pipe = pipeline("summarization", model="facebook/bart-large-cnn", device=device)
# messages = [
#     {"role": "user", "content": f'summarize the text: {user_input}',}
# ]
messages = f'summarize the text and pay attention that there are different speakers: {user_input}'
result = pipe(messages, max_length=1500, min_length=10, do_sample=False)
print(result)

# Write result to file
with open('summary', 'w') as f:
    f.write(str(result))
