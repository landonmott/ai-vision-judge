# ai-vision-judge
AI vision model that inputs presentations as PDF and outputs JSON file to be fed to final judge AI

AI vision model pipeline that analyzes presentation slides and generates structured feedback for judging.

PDF → slide images → vision model (qwen3-vl:2b via Ollama) → structured evaluation → scoring

Step-by-step:

  PDF is loaded using fitz (PyMuPDF)

  Each slide is converted into an image
  
  Each image is passed to a vision model (qwen3-vl:2b)

  The model returns structured JSON describing:

  layout

  visual elements

  summary

Install:
pip install pymupdf pillow ollama

Install and run Ollama and pull the model:
ollama pull qwen3-vl:2b


To use run:  
  python Ollama_vision_hackathon.py

or to modify path for pdf:
  pdf_path = "your_presentation.pdf"

Example code:
results = analyze_deck("test_slides.pdf")
print(results)

Example result (inside of returned JSON file):
[
  {
    "slide": 1,
    "analysis": {
      "layout": "...",
      "visual_elements": "...",
      "summary": "..."
    }
  }
]
