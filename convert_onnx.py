from torch import nn
from torch.nn import functional as F
import torch.onnx


class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.input_func = nn.Linear(input_size, 250)
        self.hidden_func = nn.Linear(250, 100)
        self.output_func = nn.Linear(100, output_size)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        h_1 = F.relu(self.input_func(x))
        h_2 = F.relu(self.hidden_func(h_1))
        y_pred = self.output_func(h_2)
        return y_pred


model = MLP(input_size=28 * 28, output_size=10)
model.load_state_dict(torch.load("tut1-model.pt"))
model.eval()

test_input = torch.rand(1, 28 * 28)

torch.onnx.export(model, test_input, "tut1-model.onnx", export_params=True, opset_version=11, do_constant_folding=True,
                  input_names=["input"], output_names=["output"],
                  dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}})
