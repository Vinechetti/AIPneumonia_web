from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
import logging


class pneumonia(nn.Module):
    def __init__(self, num_classes=2):
        super(pneumonia, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1
        )

        self.bn1 = nn.BatchNorm2d(num_features=12)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(
            in_channels=12, out_channels=20, kernel_size=3, stride=1, padding=1
        )
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(
            in_channels=20, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.bn3 = nn.BatchNorm2d(num_features=32)
        self.relu3 = nn.ReLU()
        self.fc = nn.Linear(in_features=32 * 112 * 112, out_features=num_classes)

    def forward(self, input):
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.relu1(output)
        output = self.pool(output)
        output = self.conv2(output)
        output = self.relu2(output)
        output = self.conv3(output)
        output = self.bn3(output)
        output = self.relu3(output)
        output = output.view(-1, 32 * 112 * 112)
        output = self.fc(output)

        return output


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"jpg", "jpeg"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model_path = "model/model.pth"
model = torch.load(model_path, weights_only=True)
model.eval()

data_T = T.Compose(
    [
        T.Resize(size=(224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    logger.debug("Predict endpoint hit")

    if "file" not in request.files:
        logger.error("No file part in request")
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        logger.error("No selected file")
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        logger.debug(f"Saving file to: {filepath}")

        try:
            file.save(filepath)
            logger.debug("File saved successfully")

            # Process the image
            logger.debug("Opening image")
            img = Image.open(filepath).convert("RGB")

            logger.debug("Applying transforms")
            img_tensor = data_T(img)
            img_new = img_tensor.view(1, 3, 224, 224)

            # Make prediction
            logger.debug("Making prediction")
            with torch.no_grad():
                logps = model(img_new)
            ps = torch.exp(logps)
            probab = list(ps.cpu()[0])
            pred_label = probab.index(max(probab))

            logger.debug(f"Prediction complete - label: {pred_label}%")

            # Clean up
            os.remove(filepath)
            logger.debug("Temporary file removed")

            return jsonify(
                {
                    "prediction": (
                        "Pneumonia detected" if pred_label else "Pneumonia not detected"
                    ),
                }
            )

        except Exception as e:
            logger.error(f"Error during processing: {str(e)}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    logger.error(f"Invalid file type: {file.filename if file else 'no file'}")
    return jsonify({"error": "Invalid file type"}), 400


if __name__ == "__main__":
    app.run(debug=True)
