from fastapi import FastAPI
from routers import photo, investigator, video, sketch


app = FastAPI(title="ForensAI AI Service")


app.include_router(photo.router)
app.include_router(investigator.router)
app.include_router(video.router)
app.include_router(sketch.router)


@app.get("/")
def root():
    return {
        "message": "ForensAI AI Service is running!"
    }