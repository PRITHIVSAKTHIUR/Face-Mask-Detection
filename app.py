import gradio as gr
from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image
import torch

# Load model and processor
model_name = "prithivMLmods/Face-Mask-Detection"
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

# ID to label mapping
id2label = {
    "0": "Face_Mask Found",
    "1": "Face_Mask Not_Found"
}

def detect_face_mask(image):
    image = Image.fromarray(image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()

    prediction = {id2label[str(i)]: round(probs[i], 3) for i in range(len(probs))}
    return prediction

# Gradio Interface
iface = gr.Interface(
    fn=detect_face_mask,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(num_top_classes=2, label="Mask Status"),
    title="Face-Mask-Detection",
    description="Upload an image to check if a person is wearing a face mask or not."
)

if __name__ == "__main__":
    iface.launch()
