import cv2
from PIL import Image
import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument('file')
parser.add_argument('-o', '--output')
parser.add_argument('--width', type=int, default=80)
parser.add_argument('--height', type=int, default=80)

args = parser.parse_args()

IMG = args.file
WIDTH = args.width
HEIGHT = args.height
OUTPUT = args.output

ascii_char = list("$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. ")


def get_char(r, g, b, alpha=256):
    if alpha == 0:
        return ' '
    length = len(ascii_char)
    gray = int(0.2126 * r + 0.7152 * g + 0.0722 * b)

    unit = (256.0 + 1) / length
    return ascii_char[int(gray / unit)]


if __name__ == '__main__':

    im = Image.open(IMG)
    im = im.resize((WIDTH, HEIGHT), Image.NEAREST)

    txt = ""

    for i in range(HEIGHT):
        for j in range(WIDTH):
            txt += get_char(*im.getpixel((j, i)))
        txt += '\n'

    print(txt)

    if OUTPUT:
        with open(OUTPUT, 'w') as f:
            f.write(txt)
    else:
        with open("output.txt", 'w') as f:
            f.write(txt)

#
# face_cascade = cv2.CascadeClassifier(
#     '/usr/local/lib/python2.7/site-packages/cv2/data/haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier('/usr/local/lib/python2.7/site-packages/cv2/data/haarcascade_eye.xml')
#
# img = cv2.imread('./nagase_tomoya.jpg')
# gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# _, threshold_img = cv2.threshold(gray_img, 60, 255, cv2.THRESH_BINARY)
#
# # show image
# threshold_img = cv2.cvtColor(threshold_img, cv2.COLOR_GRAY2RGB)
#
#
# plt.imshow(threshold_img)
# plt.show()

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# faces = face_cascade.detectMultiScale(gray, 1.3, 5)
# for (x, y, w, h) in faces:
#     img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
#     roi_gray = gray[y:y + h, x:x + w]
#     roi_color = img[y:y + h, x:x + w]
#     eyes = eye_cascade.detectMultiScale(roi_gray)
#     for (ex, ey, ew, eh) in eyes:
#         cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
#
#
# cv2.imshow('img', img)
#
# cv2.startWindowThread()
# cv2.waitKey(5000)
# cv2.destroyAllWindows()
