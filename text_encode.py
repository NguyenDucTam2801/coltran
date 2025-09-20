import torch
from transformers import CLIPProcessor, CLIPTextModel


class TextEncoder:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        model_name = "openai/clip-vit-base-patch32"
        self.model = CLIPTextModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.device = device
        print(f"CLIP Text Encoder loaded on device: {self.device}")

    @torch.no_grad()
    def encode_batch(self, text_list):
        """Encodes a batch of text prompts."""
        inputs = self.processor(text=text_list, return_tensors="pt", padding=True, truncation=True).to(self.device)
        outputs = self.model(**inputs)

        # We'll save the pooled_output for global conditioning.
        # If you need cross-attention, save last_hidden_state instead.
        return outputs.pooler_output.cpu().numpy()  # Move to CPU and convert to NumPy for saving