from model import ChessModel, ChessModelConfig
import torch 

config = ChessModelConfig()
config.n_layer = 6
config.n_embed = 64
config.n_head = 8
config.bias = False
config.dropout = 0.0

model = ChessModel(config)
scripted_model = torch.jit.script(model)  # Convert to TorchScript
scripted_model.save("untrained_model.pt")  # Save the model
print("Model saved as untrained_model.pt")
print(f"The model has {model.get_num_params()} parameters")

input_tensor = torch.randn(8, 64, 64)

# Pass the input tensor through the model
output = model(input_tensor)

# Print the output dimensions
evaluation, move_logits = output
print("Evaluation output dimensions:", evaluation.shape)
print("Move logits output dimensions:", move_logits.shape)