from flask import Flask, render_template, request
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io
import time


app = Flask(__name__)


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_func = nn.Linear(input_dim, 250)
        self.hidden_func = nn.Linear(250, 100)
        self.output_func = nn.Linear(100, output_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        h_1 = F.relu(self.input_func(x))
        h_2 = F.relu(self.hidden_func(h_1))
        y_pred = self.output_func(h_2)
        return y_pred, h_2


model = MLP(input_dim=28 * 28, output_dim=10)
model.load_state_dict(
    torch.load("Image-Classification-App\\tut1-model.pt", map_location="cpu")
)
model.eval()


transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    prediction_time = None
    if request.method == "POST":
        file = request.files["file"]
        if file:
            img_bytes = file.read()
            img = Image.open(io.BytesIO(img_bytes))
            img = transform(img).unsqueeze(0)
            start_time = time.time()

            with torch.no_grad():
                y_pred, _ = model(img)
                predicted_class = y_pred.argmax(dim=1).item()
                prediction = predicted_class

            end_time = time.time()
            prediction_time = end_time - start_time
    return render_template(
        "index.html", prediction=prediction, prediction_time=prediction_time
    )


if __name__ == "__main__":
    app.run(debug=True)
