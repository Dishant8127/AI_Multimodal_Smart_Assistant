import numpy as np
from PIL import Image
import io
import torchvision.transforms as transforms

def preprocess_image_file(uploaded_file, target_size=(224,224)):
    # uploaded_file is a bytes-like object or file-like
    img = Image.open(uploaded_file).convert('RGB')
    img = img.resize(target_size)
    arr = np.array(img)/255.0
    # transpose for PyTorch (C,H,W)
    arr_t = arr.transpose(2,0,1).astype('float32')
    return arr_t
