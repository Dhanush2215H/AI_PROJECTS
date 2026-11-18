# Build a Vision AI App with Python

A beginner-friendly Python notebook that uses **Florence-2** to perform object detection on an image and convert the model's output into structured bounding-box coordinates.

## Overview

This project loads the `florence-community/Florence-2-base` model from Hugging Face, downloads a sample car image, sends the `<OD>` object-detection prompt to Florence-2, and prints the parsed detection results.

The notebook demonstrates the core workflow for a Python vision application:

1. Install the required machine-learning libraries.
2. Load a Florence-2 model and processor.
3. Download an image from a URL.
4. Prepare the image and task prompt for the model.
5. Generate detections.
6. Parse and display object labels and bounding-box coordinates.

## Features

- Loads Florence-2 from Hugging Face.
- Uses automatic device mapping for supported accelerators.
- Downloads a sample RGB image from the internet.
- Sends the Florence-2 `<OD>` object-detection task prompt.
- Decodes model output into structured results.
- Produces coordinates for the image's original dimensions.

## Requirements

- Python 3.9 or newer
- Jupyter Notebook or JupyterLab
- Internet access for the model and sample image
- A Hugging Face-compatible environment

A GPU is recommended for faster inference, but CPU execution may also work depending on available memory and processor support.

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows, activate the environment with:

```bash
.venv\Scripts\activate
```

Install Jupyter and the notebook dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install jupyter
python -m pip install transformers torch torchvision pillow requests accelerate
```

## Usage

From the project directory, start Jupyter:

```bash
jupyter notebook
```

Open `Vision_AI_App_with_Python.ipynb` and run the cells in order.

The first run downloads the Florence-2 model files, so it may take several minutes. The notebook then loads a car image from Hugging Face and produces an object-detection result similar to:

```python
{
    "<OD>": {
        "bboxes": [[...]],
        "labels": ["..."]
    }
}
```

The exact detections and labels depend on the model, image, and inference environment.

## Notebook Overview

| Section | Description |
|---|---|
| Dependency setup | Installs PyTorch, Transformers, image, and download libraries |
| Model loading | Loads `florence-community/Florence-2-base` and its processor |
| Image loading | Opens the sample image and converts it to RGB |
| Input preparation | Applies the `<OD>` prompt and prepares tensor inputs |
| Inference | Generates object-detection tokens with the model |
| Post-processing | Converts generated text into labels and bounding boxes |

## Use Your Own Image

Replace the sample URL with a direct image URL:

```python
image_url = "https://example.com/your-image.jpg"
image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
```

For a local image, use:

```python
image = Image.open("path/to/your-image.jpg").convert("RGB")
```

## Project Structure

```text
1_Build_a_Vision_AI_App_with_Python/
├── Vision_AI_App_with_Python.ipynb
└── README.md
```

## Notes

- The notebook requires internet access when downloading the model and sample image.
- The first model download can use substantial disk space and may require additional memory.
- Detection quality may vary based on the input image and the base Florence-2 model.
- Bounding boxes are not drawn on the image in the current notebook; they are returned as structured post-processed data.

## License

This project is available under the repository's [Apache License 2.0](../../LICENSE).