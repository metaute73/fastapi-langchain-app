from fastapi import FastAPI
from .funcionalidad1 import usar_infoSE

app = FastAPI()

@app.get("/ask/")
async def ask_question(question: str):
    # Usamos la función ya creada que invoca la cadena
    return {"response": usar_infoSE(question)}