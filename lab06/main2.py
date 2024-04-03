from PIL import Image
import numpy as np

pictures = ['forest.png', 'desert.png', 'sea.png']

def convert_to_gray_avg(image):
    return np.mean(image, axis=2)

def convert_to_gray_weighted(image):
    return 0.299*image[:,:,0] + 0.587*image[:,:,1] + 0.114*image[:,:,2]

for picture in pictures:
    image = Image.open(picture)
    image = np.array(image)
    print(image)

    gray_avg = convert_to_gray_avg(image)
    Image.fromarray(gray_avg.astype(np.uint8)).save(f"grey_avg_{picture}")

    gray_weighted = convert_to_gray_weighted(image)
    Image.fromarray(gray_weighted.astype(np.uint8)).save(f"gray_weighted_{picture}")