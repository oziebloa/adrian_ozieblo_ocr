### API router, handling all http routes ###
import os
from pathlib import Path
from typing import List

from fastapi import APIRouter, FastAPI, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.ml import prediction_service

app = FastAPI()
router = APIRouter()
templates = Jinja2Templates(directory="src/adapters/http/ui/templates")
app.mount(
    "/static",
    StaticFiles(directory=Path(__file__).parent.parent.absolute() / "ui/static"),
    name="static",
)
app.mount(
    "/tmp",
    StaticFiles(directory=Path(__file__).parent.parent.parent.parent.parent.absolute() / "tmp"),
    name="tmp",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@router.get('/', response_class=HTMLResponse)
async def get_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@router.get('/about', response_class=HTMLResponse)
async def get_about(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})


@router.get('/ocr', response_class=HTMLResponse)
async def get_ocr_form(request: Request):
    return templates.TemplateResponse("ocr_form.html", {"request": request})


@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    return FileResponse('src/adapters/http/ui/static/images/favicon.ico')


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, ml_choice: str = Form(), ocr_img: List[UploadFile] = File()):
    results_subdir = prediction_service.get_list_of_images_transcribed(ocr_img, ml_choice)
    filenames = [os.path.splitext(img.filename)[0] for img in ocr_img]
    return templates.TemplateResponse("results.html", {"request": request, "ml_choice": ml_choice,
                                                       "subdir": results_subdir, "filenames": filenames})
