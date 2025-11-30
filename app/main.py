from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from openai import OpenAI
from PIL import Image
import io
import base64
import os

app = FastAPI()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class SolveRequest(BaseModel):
    question: str

@app.get("/")
def home():
    return {"status": "Backend running successfully!"}

@app.post("/solve-ocr")
async def solve_ocr(file: UploadFile = File(...)):
    image_bytes = await file.read()
    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{base64_image}"},
                    {"type": "text", "text": "Extract the math problem from this image and solve it step-by-step."}
                ]
            }
        ]
    )

    answer = response.choices[0].message["content"]
    return {"solution": answer}

@app.post("/solve-text")
async def solve_text(req: SolveRequest):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": f"Solve this math problem step-by-step: {req.question}"
            }
        ]
    )

    answer = response.choices[0].message["content"]
    return {"solution": answer}
