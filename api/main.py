from fastapi import FastAPI
from .routers.model import model

app = FastAPI()
app.include_router(model)