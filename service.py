import base64
import numpy as np
import cv2

def save_img(img_base64):
    #binary <- string base64
    img_binary = base64.b64decode(img_base64)
    #jpg <- binary
    img_jpg=np.frombuffer(img_binary, dtype=np.uint8)
    #raw image <- jpg
    img = cv2.imdecode(img_jpg, cv2.IMREAD_COLOR)
    #Path to save the decoded image
    image_file="/home/capture_img/images/img0000.jpg "
    #Save image
    cv2.imwrite(image_file, img)
    return "SUCCESS"
