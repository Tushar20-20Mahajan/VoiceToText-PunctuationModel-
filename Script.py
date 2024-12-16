# import torch
# from transformers import AutoTokenizer, AutoModelForTokenClassification
# import coremltools as ct

# # Load pre-trained Hugging Face model and tokenizer for punctuation
# tokenizer = AutoTokenizer.from_pretrained("oliverguhr/fullstop-punctuation-multilang")
# model = AutoModelForTokenClassification.from_pretrained("oliverguhr/fullstop-punctuation-multilang")

# # Test input (replace this with real transcription data in your case)
# input_text = "this is an example of a transcription without punctuation"
# inputs = tokenizer(input_text, return_tensors="pt")

# # Convert the model output (for testing)
# model.eval()
# with torch.no_grad():
#     outputs = model(**inputs)

# # Convert the Hugging Face model to CoreML
# traced_model = torch.jit.trace(model, [inputs["input_ids"], inputs["attention_mask"]])

# # Use coremltools to convert the traced PyTorch model to CoreML format
# mlmodel = ct.convert(traced_model, inputs=[
#     ct.TensorType(name="input_ids", shape=inputs["input_ids"].shape),
#     ct.TensorType(name="attention_mask", shape=inputs["attention_mask"].shape)
# ])

# # Save the CoreML model to a file
# mlmodel.save("PunctuationModel.mlmodel")

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import coremltools as ct
import numpy as np

# Using BERT model instead of T5
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Set the model to evaluation mode
model.eval()

# Test input
input_text = "this is an example of a transcription without punctuation"
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)

# Create a wrapper class to handle the model's forward pass
class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

# Create and trace the wrapped model
wrapped_model = ModelWrapper(model)
wrapped_model.eval()

# Create example inputs for tracing
example_input_ids = inputs["input_ids"]
example_attention_mask = inputs["attention_mask"]

# Trace the model
traced_model = torch.jit.trace(wrapped_model, (example_input_ids, example_attention_mask))

# Convert to CoreML
mlmodel = ct.convert(
    traced_model,
    inputs=[
        ct.TensorType(
            name="input_ids", 
            shape=example_input_ids.shape, 
            dtype=np.int32
        ),
        ct.TensorType(
            name="attention_mask", 
            shape=example_attention_mask.shape, 
            dtype=np.int32
        )
    ],
    minimum_deployment_target=ct.target.iOS15
)

# Save the CoreML model
# Change only the save line to use .mlpackage extension
mlmodel.save("PunctuationModel.mlpackage")

