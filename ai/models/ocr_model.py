# def ocr_tag(image_url):

#     result = {
#         "brand": "오뚜기",
#         "product": "컵누들"
#     }

#     return result
# from core.util.image_util import load_image

import torch
import cv2
import os
import re
import numpy as np
import shutil
from easyocr import Reader
from rapidfuzz import fuzz
import urllib.request

# 이미지 URL
# url = 'https://hkhan2023.s3.ap-northeast-2.amazonaws.com/tag_24.jpg'

# urllib를 사용하여 이미지 다운로드
# response = urllib.request.urlopen(url)
# image_data = response.read()

# # 이미지를 파일로 저장
# with open('downloaded_tag_24.jpg', 'wb') as f:
#     f.write(image_data)

def ocr_tag(img_url):
    response = urllib.request.urlopen(img_url)
    image_data = response.read()

    local_img_url = 'downloaded_tag_24.jpg'
    with open(local_img_url, 'wb') as f:
        f.write(image_data)

    new_img = cv2.imread(local_img_url)

    # Run yolov5
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='ai\\weights\\ocr_weights_2.pt', force_reload=True)
    results = model(new_img)

    # save cropped area
    results.crop()
    # cur_dir = os.getcwd()
    crop_path = 'runs\\detect\\exp\\crops\\words\\image0.jpg'
    crop_img = cv2.imread(crop_path)

    # run easyOCR in cropped
    langs = ['ko', 'en']
    print("[INFO] OCR'ing input image...")
    reader = Reader(lang_list=langs, gpu=True)
    results = reader.readtext(crop_img)

    real = []
    for (bbox, text, prob) in results:
      pot = [')', '>', '"\"', '원', '#', ","]
      # if ')' in text or '>' in text or '"\"' in text or '원' in text:
      for i in pot:
        if i in text:
          real.append(text)
      (tl, tr, br, bl) = bbox
      tl = (int(tl[0]), int(tl[1]))
      tr = (int(tr[0]), int(tr[1]))
      br = (int(br[0]), int(br[1]))
      bl = (int(bl[0]), int(bl[1]))
    
    #  cv2.rectangle(org_image, tl, br, (0, 255, 0), 3)
    info = dict()
    brand = ''
    product = ''
    price = ''
    for i in real:
        if ')' in i:
            brand = i.split(')')[0]
            product = i.split(')')[1]
            info['brand'] = brand
            info['product'] = product
        if '>' in i:
            price = i.split('>')[-1]
            info['price'] = price
        if "'\'" in i:
            price = i.split("'\'")[-1]
            info['price'] = price
        if '#' in i:
            price = i.split("#")[-1]
            info['price'] = price
        if "원" in i:
            price = i.split("원")[0]
            info['price'] = price
        if "," in i:
            price = str(i)
            info['price'] = price
        if "00" in i:
            price = str(i)
            info['price'] = price
    
    product_list = ["신라면", "신라면블랙", "진라면", "안성탕면", "나가사끼짬뽕", "카구리", "너구리", "불닭볶음면",  "컵누들우동맛", "컵누들매운맛", "비빔면큰컵", "새우깡", "매운새우깡", "꼬깔콘", "포카칩" , "초코파이", "빵부장", "초코하임", "화이트하임", "메로나", "누가바"]
    brand_list = ["매일", "농심", "빙그레", "서울우유", "롯데", "크라운", "팔도", "오리온", "오뚜기", "리츠"]

    for i in info:
        info[i] = re.sub(r"[^\uAC00-\uD7A30-9a-zA-Z\s]", '', info[i])

    idxi, idxj = -1, -1
    max_brand, max_product = 0.0, 0.0
    
    # if len(info['brand']) == 0:
    for i in brand_list:
        prob = fuzz.ratio(i, info['brand'])
        if np.float16(prob) > np.float16(max_brand):
            max_brand = prob
            idxi = i

    # if len(info['product']) != 0:
    for j in product_list:
        prob2 = fuzz.ratio(j, info['product'])
        if (np.float16(prob2) > np.float16(max_product)):
            max_product = prob2
            idxj = j

    info['brand'] = idxi
    info['product'] = idxj

    shutil.rmtree('runs/detect')
    print('ocr process done.')
    print('detected:', info)
    # 출력 예시 : {'brand': '롯데', 'product': ' 제로후르츠젤리52g', 'price': '1200'}
    return info