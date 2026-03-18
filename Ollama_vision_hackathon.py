import fitz  # PyMuPDF
from PIL import Image
import ollama
import tempfile
import os
import sys
import json
import re


# ── helpers ──────────────────────────────────────────────────────────────────

def pdf_to_images(pdf_path: str) -> list[Image.Image]:
    """Convert PDF slides to PIL images."""
    doc = fitz.open(pdf_path)
    return [
        Image.frombytes("RGB", [p.width, p.height], p.samples)
        for p in (page.get_pixmap(dpi=300) for page in doc)
    ]


def extract_json(text: str) -> dict:
    """
    Strip markdown fences and parse JSON from model output.
    Returns parsed dict, or a fallback dict with raw text on failure.
    """
    # Remove ```json ... ``` or ``` ... ``` wrappers
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()

    # Try to grab the first {...} block if there's surrounding prose
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"parse_error": True, "raw_text": text}


def analyze_slide(image: Image.Image, slide_num: int) -> dict:
    """Analyze a single slide using a vision model. Returns a parsed dict."""
    prompt = """Analyze this presentation slide. Return ONLY valid JSON (no markdown, no explanation) with these keys:
- "layout": string describing the slide layout
- "visual_elements": list of strings (charts, diagrams, images, etc.)
- "summary": one-sentence description of the slide's content
"""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        temp_path = tmp.name
        image.save(temp_path)

    try:
        response = ollama.chat(
            model="qwen3-vl:2b",
            messages=[{
                "role": "user",
                "content": prompt,
                "images": [temp_path]
            }]
        )
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    raw = response["message"]["content"]
    parsed = extract_json(raw)

    if parsed.get("parse_error"):
        print(f"  ⚠️  Slide {slide_num}: JSON parse failed, storing raw text.")
    
    return parsed


# ── judge ─────────────────────────────────────────────────────────────────────

JUDGE_SYSTEM_PROMPT = """You are an expert presentation coach. You will receive JSON descriptions of slides.
Return ONLY valid JSON (no markdown) with:
- "overall_score": integer 1-10
- "strengths": list of strings
- "weaknesses": list of strings
- "recommendations": list of actionable improvement strings
"""

def judge_deck(slide_analyses: list[dict]) -> dict:
    """Feed all slide JSON descriptions to a judge model."""
    payload = json.dumps(slide_analyses, indent=2)
    prompt = f"Here are the slide-by-slide analyses of a presentation deck:\n\n{payload}\n\nEvaluate the overall deck."

    response = ollama.chat(
        model="qwen3-vl:2b",  # lightweight text model for judging
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user",   "content": prompt}
        ]
    )

    raw = response["message"]["content"]
    parsed = judge_extract_json(raw)
    return parsed


def judge_extract_json(text: str) -> dict:
    """Same JSON cleaning for judge output."""
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"parse_error": True, "raw_text": text}


# ── pipeline ──────────────────────────────────────────────────────────────────

def analyze_deck(pdf_path: str) -> tuple[list[dict], dict]:
    """
    Full pipeline:
      1. Convert PDF → images
      2. Vision model → JSON per slide
      3. Judge model → overall evaluation
    Returns (slide_results, judge_verdict).
    """
    slides = pdf_to_images(pdf_path)
    slide_results = []

    for i, slide in enumerate(slides, start=1):
        print(f"  Analyzing slide {i}/{len(slides)}...")
        analysis = analyze_slide(slide, i)
        slide_results.append({"slide": i, "analysis": analysis})

    print("\n  Running judge model...")
    judge_verdict = judge_deck(slide_results)

    return slide_results, judge_verdict


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python slide_analyzer.py <path/to/deck.pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]

    if not os.path.exists(pdf_path):
        print(f"Error: '{pdf_path}' not found.")
        sys.exit(1)

    print(f"\n📄 Processing: {pdf_path}\n")
    slide_results, judge_verdict = analyze_deck(pdf_path)

    # Optionally save results
    output = {
        "slides": slide_results,
        "judge_verdict": judge_verdict
    }

    out_path = pdf_path.replace(".pdf", "_analysis.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ Done! Results saved to: {out_path}")
    print("\n── Judge Verdict ──────────────────────────────")
    print(json.dumps(judge_verdict, indent=2))