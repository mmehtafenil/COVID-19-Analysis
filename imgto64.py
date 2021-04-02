from PIL import Image
import base64
from io import BytesIO

def b64(file):
    
    #PIL to base64 String
    img = Image.open(file)
    im_file = BytesIO()
    img.save(im_file, format="JPEG")
    im_bytes = im_file.getvalue()
    im_b64 = base64.b64encode(im_bytes)        
    return(im_b64) 



