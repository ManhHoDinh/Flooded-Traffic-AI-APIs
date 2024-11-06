####################################### IMPORT #################################
import json
import pandas as pd
from PIL import Image
from loguru import logger
import sys

from fastapi import FastAPI, File, status, BackgroundTasks
from fastapi.responses import RedirectResponse
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import HTTPException
import asyncio
from io import BytesIO

from test import get_image_from_bytes
from test import detect_sample_model
from test import add_bboxs_on_img
from test import get_bytes_from_image
import httpx
import time
import os

####################################### logger #################################

logger.remove()
logger.add(
    sys.stderr,
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>",
    level=10,
)
logger.add("log.log", rotation="1 MB", level="DEBUG", compression="zip")

###################### FastAPI Setup #############################

# title
app = FastAPI(
    title="Object Detection FastAPI Template",
    description="""Obtain object value out of image
                    and return image and json result""",
    version="2023.1.31",
)

# This function is needed if you want to allow client requests 
# from specific domains (specified in the origins argument) 
# to access resources from the FastAPI server, 
# and the client and server are hosted on different domains.
origins = [
    "http://localhost",
    "http://localhost:8008",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def fetch_cameras():
    url = "https://api.notis.vn/v4/cameras/bybbox?lat1=11.160767&lng1=106.554166&lat2=9.45&lng2=128.99999"
    headers = {
        'accept': 'application/json',
        'device-id': 'bf738a0a3e6eddc2',
        'origin': 'http://localhost',
    }
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        return response.json()

async def get_image_and_detect(url):
    print("URL: ",url)
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            if response.status_code == 200:
                image_bytes = response.content
                # Xử lý image_bytes với chức năng detect
                input_image = get_image_from_bytes(image_bytes)
                predict = detect_sample_model(input_image)
                logger.info("Detection result: {}", predict)
            else:
                logger.error("Failed to fetch image, status code: {}", response.status_code)
    except Exception as e:
        logger.error("Error in fetching or detecting image: {}", e)

async def process_camera_images(duration_in_seconds=60):
    camera_list = await fetch_cameras()    
    print("Camera Length: ",len(camera_list))
    start = os.getenv("START")
    end = os.getenv("END")
    
    if start is None or end is None:
        print("START and END environment variables are required")
    else:
        try:
            start = int(start)
            end = int(end)
        except ValueError:
            print("START and END must be integers")
            start, end = 0, len(camera_list)  # Gán giá trị mặc định nếu lỗi

        if 0 <= start < len(camera_list) and 0 < end <= len(camera_list):
            camera_list = camera_list[start:end + 1]
            print("Start: ", start)
            print("End: ", end)
        else:
            print("START and END must be within valid range")

    start_time = time.time()
    while time.time() - start_time < duration_in_seconds:
        tasks = []
        for camera in camera_list:
            url = f"http://giaothong.hochiminhcity.gov.vn/render/ImageHandler.ashx?id={camera['_id']}"
            tasks.append(get_image_and_detect(url))
        await asyncio.gather(*tasks)
        await asyncio.sleep(5)  

async def periodic_task():
    while True:
        print("Thực hiện nhiệm vụ.")
        await process_camera_images()
        print("Xử lý xong.")
        await asyncio.sleep(5)  # Đợi 60 giây trước khi thực hiện nhiệm vụ tiếp theo
@app.on_event("startup")
async def save_openapi_json():
    '''This function is used to save the OpenAPI documentation 
    data of the FastAPI application to a JSON file. 
    The purpose of saving the OpenAPI documentation data is to have 
    a permanent and offline record of the API specification, 
    which can be used for documentation purposes or 
    to generate client libraries. It is not necessarily needed, 
    but can be helpful in certain scenarios.'''
    asyncio.create_task(periodic_task())
    openapi_data = app.openapi()
    # Change "openapi.json" to desired filename
    with open("openapi.json", "w") as file:
        json.dump(openapi_data, file)

# redirect
@app.get("/", include_in_schema=False)
async def redirect():
    return RedirectResponse("/docs")


@app.get('/healthcheck', status_code=status.HTTP_200_OK)
def perform_healthcheck():
    '''
    It basically sends a GET request to the route & hopes to get a "200"
    response code. Failing to return a 200 response code just enables
    the GitHub Actions to rollback to the last version the project was
    found in a "working condition". It acts as a last line of defense in
    case something goes south.
    Additionally, it also returns a JSON response in the form of:
    {
        'healtcheck': 'Everything OK!'
    }
    '''
    return {'healthcheck': 'Everything OK!'}


######################### Support Func #################################

def crop_image_by_predict(image: Image, predict: pd.DataFrame(), crop_class_name: str,) -> Image:
    """Crop an image based on the detection of a certain object in the image.
    
    Args:
        image: Image to be cropped.
        predict (pd.DataFrame): Dataframe containing the prediction results of object detection model.
        crop_class_name (str, optional): The name of the object class to crop the image by. if not provided, function returns the first object found in the image.
    
    Returns:
        Image: Cropped image or None
    """
    crop_predicts = predict[(predict['name'] == crop_class_name)]

    if crop_predicts.empty:
        raise HTTPException(status_code=400, detail=f"{crop_class_name} not found in photo")

    # if there are several detections, choose the one with more confidence
    if len(crop_predicts) > 1:
        crop_predicts = crop_predicts.sort_values(by=['confidence'], ascending=False)

    crop_bbox = crop_predicts[['xmin', 'ymin', 'xmax','ymax']].iloc[0].values
    # crop
    img_crop = image.crop(crop_bbox)
    return(img_crop)


######################### MAIN Func #################################


@app.post("/img_object_detection_to_json")
def img_object_detection_to_json(file: bytes = File(...)):
    """
    Object Detection from an image.

    Args:
        file (bytes): The image file in bytes format.
    Returns:
        dict: JSON format containing the Objects Detections.
    """
    # Step 1: Initialize the result dictionary with None values
    result={'detect_objects': None}

    # Step 2: Convert the image file to an image object
    input_image = get_image_from_bytes(file)

    # Step 3: Predict from model
    predict = detect_sample_model(input_image)

    # Step 4: Select detect obj return info
    # here you can choose what data to send to the result
    detect_res = predict[['name', 'confidence']]
    objects = detect_res['name'].values

    result['detect_objects_names'] = ', '.join(objects)
    result['detect_objects'] = json.loads(detect_res.to_json(orient='records'))

    # Step 5: Logs and return
    logger.info("results: {}", result)
    return result

@app.post("/img_object_detection_to_img")
def img_object_detection_to_img(file: bytes = File(...)):
    """
    Object Detection from an image plot bbox on image

    Args:
        file (bytes): The image file in bytes format.
    Returns:
        Image: Image in bytes with bbox annotations.
    """
    # get image from bytes
    input_image = get_image_from_bytes(file)

    # model predict
    predict = detect_sample_model(input_image)

    # add bbox on image
    final_image = add_bboxs_on_img(image = input_image, predict = predict)

    # return image in bytes format
    return StreamingResponse(content=get_bytes_from_image(final_image), media_type="image/jpeg")
