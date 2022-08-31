import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

inputFile = open(sys.argv[1], "rb")
if inputFile.readline() != b'P5\n':
    print('请输入PGM格式图像文件')
    assert inputFile.readline() == b'P5\n'
(width, height) = [int(i) for i in inputFile.readline().split()]
depth = int(inputFile.readline())
assert depth <= 255
inputImage = np.zeros((height, width))
for i in range(height):
    for j in range(width):
        inputImage[i, j] = ord(inputFile.read(1))
inputFile.close()
#fig, ax = plt.subplots()
#ax.imshow(inputImage, cmap='gray', vmin=0, vmax=255)
#fig.tight_layout()
#fig.savefig(sys.argv[1] + ".png")
img = Image.fromarray(inputImage)
if img.mode == "F":
    img = img.convert('L')
img.save(sys.argv[1] + ".jpg")
