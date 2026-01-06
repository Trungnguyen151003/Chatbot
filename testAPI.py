from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time
from llama_cpp import Llama
import uvicorn
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI()

# Gắn thư mục chứa file HTML, CSS, JS
app.mount("/static", StaticFiles(directory="static"), name="static")

model_local_path = "E:/Phi-3.5-mini-instruct-IQ2_M.gguf"
process = Llama(model_local_path, n_ctx=4096, n_threads=10, n_gpu_layers=32, verbose=False)

class Question(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(q: Question):
    try:
        response = process.create_chat_completion(
            messages=[{"role": "user", "content": q.question}],
        )
        answer = response['choices'][0]['message']['content'].strip()
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def home():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
