from fastapi import FastAPI
from app.routes import  aws_routes, mongo_routes

app = FastAPI(title="FastAPI MongoDB AWS Control")

app.include_router(mongo_routes.router)
app.include_router(aws_routes.router)

@app.get("/")
def home():
    return {"message": "text to 3d pipeline"}
