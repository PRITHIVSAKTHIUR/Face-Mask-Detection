![6.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/UOJ8KTNPv3KPtk9ZyY9Ll.png)

# **Face-Mask-Detection**

> **Face-Mask-Detection** is a binary image classification model based on `google/siglip2-base-patch16-224`, trained to detect whether a person is **wearing a face mask** or **not**. This model can be used in **public health monitoring**, **access control systems**, and **workplace compliance enforcement**.

```py
Classification Report:
                     precision    recall  f1-score   support

    Face_Mask Found     0.9662    0.9561    0.9611      5883
Face_Mask Not_Found     0.9568    0.9667    0.9617      5909

           accuracy                         0.9614     11792
          macro avg     0.9615    0.9614    0.9614     11792
       weighted avg     0.9615    0.9614    0.9614     11792
```

![download.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/sHVD-hwWEVZT2lQMmlq9_.png)

---

## **Label Classes**

The model distinguishes between the following face mask statuses:

```
0: Face_Mask Found  
1: Face_Mask Not_Found
```

---

## **Installation**

```bash
!pip install transformers torch pillow gradio hf_xet
```

---

## **Example Inference Code**

```python
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
```

---

## **Applications**

* **COVID-19 Compliance Monitoring**
* **Security and Access Control**
* **Automated Surveillance Systems**
* **Health Safety Enforcement in Public Spaces** 
