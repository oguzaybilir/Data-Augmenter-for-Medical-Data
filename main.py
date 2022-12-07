import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob 
from scipy import signal
from PIL import Image
from skimage.exposure import rescale_intensity
from PIL import ImageEnhance


paths = glob.glob("./images/*/*.jpg")

def sobel_edge_detection(image, kernel):

    (iH, iW) = image.shape[:2]
    (kH, kW) = kernel.shape[:2]

    pad = (kW-1)//2 
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    output = np.zeros((iH, iW), dtype="float32")

    for y in np.arange(pad, iH + pad):
        for x in np.arange(pad, iW + pad):

            roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]
            k = (roi * kernel).sum()
            output[y - pad, x - pad] = k

            output = rescale_intensity(output, in_range=(0,255))
            output = (output * 255).astype("uint8")
            return output
def contrast(image):
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    kopya = image.copy()
    kopya = cv2.cvtColor(kopya, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(kopya,(5,5),0)
    thresh = cv2.threshold(blur,10,255, cv2.THRESH_BINARY)[1]
    kontur = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kontur = kontur[0][0]
    kontur = kontur[:,0,:]
    x1 = tuple(kontur[kontur[:,0].argmin()])[0]
    y1 = tuple(kontur[kontur[:,1].argmin()])[1]
    x2 = tuple(kontur[kontur[:,0].argmax()])[0]
    y2 = tuple(kontur[kontur[:,1].argmax()])[1]
    x = int(x2-x1)*4//50
    y = int(y2-y1)*5//50
    kopya2 = image.copy()
    lab = cv2.cvtColor(kopya2, cv2.COLOR_RGB2LAB)
    l,a,b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=5.0,tileGridSize=((8,8)))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    son = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    med_son = cv2.medianBlur(son, 3)
    arka_plan = cv2.medianBlur(son, 37)
    maske = cv2.addWeighted(med_son,1,arka_plan,-1,255)
    son_img = cv2.bitwise_and(maske,med_son)
    return son_img
def negative_maker(image):

    img_bgr = image
    height, width, _ = img_bgr.shape
    
    for i in range(0, height - 1):
        for j in range(0, width - 1):
        
            pixel = img_bgr[i, j]
        
            pixel[0] = 255 - pixel[0]
            pixel[1] = 255 - pixel[1]   
            pixel[2] = 255 - pixel[2]
            img_bgr[i, j] = pixel
    
    return img_bgr
def cvt_lab(image):
    
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab_image)
    
    return lab_image , l 
def four_process(image):

    img = image
    bilateral = cv2.bilateralFilter(img, 11, 75, 75)
    gauss = cv2.GaussianBlur(img, (5, 5), 7)
    wiener = signal.wiener(img, 5, 5)

    return bilateral, gauss, wiener
def four_process_new(image):

    img = image
    bilateral = cv2.bilateralFilter(img, 11, 75, 75)
    gauss = cv2.GaussianBlur(img, (9, 9), 7)
    wiener = signal.wiener(img, 7, 7)

    return bilateral, gauss, wiener
def sobel(image):

    sobelX= np.array((
        [1,0,1],
        [-2,0,2],
        [1,0,1]), dtype="float32"
    )
    sobelY = np.array((
        [-1,-2,-1],
        [0,0,0],
        [1,2,1]), dtype="float32"
    )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    convoleOutput = sobel_edge_detection(gray, sobelX)
    opencvOutput = cv2.filter2D(gray, -1, sobelX)

    convoleOutput = sobel_edge_detection(gray, sobelY)
    opencvOutput = cv2.filter2D(gray, -1, sobelY)

    return convoleOutput,opencvOutput,convoleOutput,opencvOutput
def bright1(image):

    new_image = np.zeros(image.shape, image.dtype)
    alpha = 1.0 # Simple contrast control
    beta = 0    # Simple brightness control

    try:
        alpha = 1.4
        beta = 70
    except ValueError:
        print('Error, not a number')

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            for c in range(image.shape[2]):
                new_image[y,x,c] = np.clip(alpha*image[y,x,c] + beta, 0, 255)

    return new_image
def bright2(image):

    new_image = np.zeros(image.shape, image.dtype)
    alpha = 1.0 # Simple contrast control
    beta = 0    # Simple brightness control

    try:
        alpha = 2.6
        beta = 30
    except ValueError:
        print('Error, not a number')

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            for c in range(image.shape[2]):
                new_image[y,x,c] = np.clip(alpha*image[y,x,c] + beta, 0, 255)

    return new_image
def bright3(image):

    new_image = np.zeros(image.shape, image.dtype)
    alpha = 1.0 # Simple contrast control
    beta = 0    # Simple brightness control

    try:
        alpha = 1.5
        beta = 60
    except ValueError:
        print('Error, not a number')

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            for c in range(image.shape[2]):
                new_image[y,x,c] = np.clip(alpha*image[y,x,c] + beta, 0, 255)

    return new_image
def bright4(image):

    new_image = np.zeros(image.shape, image.dtype)
    alpha = 1.0 # Simple contrast control
    beta = 0    # Simple brightness control

    try:
        alpha = 2.5
        beta = 20
    except ValueError:
        print('Error, not a number')

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            for c in range(image.shape[2]):
                new_image[y,x,c] = np.clip(alpha*image[y,x,c] + beta, 0, 255)

    return new_image

def data_augmenter(path):

    image_orig = cv2.imread(path)
    save_path = "images\\" + path.split("\\")[-2] + "\\" +  (path.split("\\")[-1]).split(".")[0]

    clahe = contrast(image_orig)
    negative_image = negative_maker(image_orig)
    lab_image, l = cvt_lab(image_orig)
    bilateral, gauss, wiener = four_process(image_orig)
    bilateral_new, gauss_new, wiener_new = four_process_new(image_orig)
    convoleOutput, opencvOutput, convoleOutput, opencvOutput = sobel(image_orig)

    bright_image1 = bright1(image_orig)
    bright_image2 = bright2(image_orig)
    bright_image3 = bright3(image_orig)
    bright_image4 = bright3(image_orig)

    cv2.imwrite(f"{save_path}_clahe.jpg",clahe)
    cv2.imwrite(f"{save_path}_negative.jpg",negative_image)
    cv2.imwrite(f"{save_path}_lab_image.jpg",lab_image)
    cv2.imwrite(f"{save_path}_lab_image_L.jpg",l)

    cv2.imwrite(f"{save_path}_bilateral.jpg",bilateral)
    cv2.imwrite(f"{save_path}_gauss.jpg",gauss)
    cv2.imwrite(f"{save_path}_wiener.jpg",wiener)

    cv2.imwrite(f"{save_path}_bilateral_new.jpg",bilateral_new)
    cv2.imwrite(f"{save_path}_gauss_new.jpg",gauss_new)
    cv2.imwrite(f"{save_path}_wiener_new.jpg",wiener_new)

    cv2.imwrite(f"{save_path}_sobel_convoleOutput.jpg",convoleOutput)
    cv2.imwrite(f"{save_path}_sobel_opencvOutput.jpg",opencvOutput)
    cv2.imwrite(f"{save_path}_sobel_convoleOutput.jpg",convoleOutput)
    cv2.imwrite(f"{save_path}_sobel_opencvOutput.jpg",opencvOutput)

    cv2.imwrite(f"{save_path}_bright_image1.jpg",bright_image1)
    cv2.imwrite(f"{save_path}_bright_image2.jpg",bright_image2)
    cv2.imwrite(f"{save_path}_bright_image3.jpg",bright_image3)
    cv2.imwrite(f"{save_path}_bright_image4.jpg",bright_image4)


for path in paths:
 
    data_augmenter(path)
  





