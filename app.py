import uvicorn
from fastapi import FastAPI, Request, File, UploadFile,Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import cv2
import pickle
import numpy as np
import os
import requests
import subprocess
import shutil
from fastapi.responses import RedirectResponse

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

UPLOAD_FOLDER = "static"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.get('/')
def index(request: Request):
    return templates.TemplateResponse("open.html", {"request": request})

@app.get('/index')
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get('/login')
def login(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get('/sign')
def sign(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})


@app.post("/login")
def login():
    return RedirectResponse("/index", status_code=303)

    





@app.post("/upload_image", response_class=HTMLResponse)
async def upload_image( request: Request,image_file: UploadFile = File(...)):
    image_path = f"{UPLOAD_FOLDER}/{image_file.filename}"
    save_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
    
    with open(save_path, "wb") as image:
        content = await image_file.read()
        image.write(content)
    
    predictions = process_image(image_path,image_file.filename)
    
    context = {
        "request": request,
    }

    return templates.TemplateResponse("index.html", context)

def process_image(file_path,imgname):
                        
    source_image = file_path
    weights_path = 'yolov5/model.pt'
    det_path='yolov5/detect.py'
    run_detection(source_image, weights_path,det_path,imgname)
    


def run_detection(source_image, weights_path,det_path,imgname):
    
    command = ["python", det_path, "--source", source_image, "--weights", weights_path]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    
    if process.returncode != 0:
        print(f"Error occurred: {stderr.decode('utf-8')}")
    else:
        print(f"Detection successful:\n{stdout.decode('utf-8')}")
    
    
    exp_folders = [folder for folder in os.listdir("yolov5/runs/detect") if folder.startswith("exp") and folder[3:].isdigit()]

    # If there are no exp folders, exit or handle the case accordingly
    if not exp_folders:
        print("No exp folders found.")
        exit()

    # Sort the exp folders by their numerical value
    sorted_exp_folders = sorted(exp_folders, key=lambda x: int(x[3:]))

    # Select the exp folder with the highest number
    latest_exp_folder = sorted_exp_folders[-1]

    # Get the full path of the latest exp folder
    latest_exp_folder = os.path.join("yolov5/runs/detect", latest_exp_folder)
     
    os.rename(f"{latest_exp_folder}/{imgname}",f"{latest_exp_folder}/output.jpg")

    destination_folder = "static"
    output_file = "output.jpg"

    # Check if the file already exists in the destination folder
    destination_path = os.path.join(destination_folder, output_file)
    if os.path.exists(destination_path):
        os.remove(destination_path)
        print(f"Existing file '{output_file}' removed from '{destination_folder}'.")

    # Move the new file to the destination folder
    shutil.move(f"{latest_exp_folder}/{output_file}", destination_folder)


    
if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
