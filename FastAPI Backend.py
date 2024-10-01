from fastapi import FastAPI, UploadFile, File
import uvicorn
from PIL import Image
import torch

app = FastAPI()

model = torch.load('best_model.pth', map_location=torch.device('cpu'))
model.eval()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file)
    # Preprocess image and convert to tensor
    input_image = preprocess_image(image)  # Apply your preprocessing
    output = model(input_image.unsqueeze(0))
    # Return segmentation result (convert to desired format)
    return {"prediction": output.tolist()}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
