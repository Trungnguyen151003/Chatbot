from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import subprocess
import uvicorn
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI()

# Gắn thư mục chứa file HTML, CSS, JS
app.mount("/static", StaticFiles(directory="static"), name="static")

class Question(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(q: Question):
    try:
        result = subprocess.run(
            ["ollama", "run", "llama2", q.question],
            capture_output=True, text=True
        )
        return {"answer": result.stdout}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def home():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
