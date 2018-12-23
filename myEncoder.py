from myfunctions import *
import sys
import struct
# import matplotlib.pyplot as plt
#################################
# 读入P5 PGM格式图像数据
# inputImage为height * width的矩阵，depth为最大值
if len(sys.argv) != 3:
    print('请加参数[InputImageFileName] [OutputFileName]')
    assert len(sys.argv) == 3
inputFile = open(sys.argv[1], "rb")
# print(inputFile.read(40))    ##########debug
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
##################################
# 设置量化矩阵
quantizationMatrix = np.array([[1, 1, 1, 1, 1, 1, 1, 1],
                               [1, 1, 1, 1, 1, 1, 1, 1],
                               [1, 1, 1, 1, 1, 1, 1, 1],
                               [1, 1, 1, 1, 1, 1, 1, 0],
                               [1, 1, 1, 1, 1, 1, 0, 0],
                               [1, 1, 1, 1, 1, 0, 0, 0],
                               [1, 1, 1, 1, 0, 0, 0, 0],
                               [1, 1, 1, 0, 0, 0, 0, 0]])
# 计算延拓图像，为了处理长和宽非8的倍数的情形
meanValue = np.mean(inputImage)
extendHeight = 8 * (int((height-1)/8) + 1)
extendWidth = 8 * (int((width-1)/8) + 1)
extendImage = np.zeros((extendHeight, extendWidth))
extendImage[:height, :width] = inputImage
extendImage[height:, :] = meanValue
extendImage[:, width:] = meanValue
modeTable = np.zeros((int(extendHeight/8), int(extendWidth/8)))  # 各个分块的预测模式表， 解码时分别使用对应的预测模式解码
##############################################
# 分别对每个块作DCT和量化
zigzagArrayMatrix = np.zeros((int(extendHeight/8), int(extendWidth/8), 8*8))
for i in range(0, extendHeight, 8):
    for j in range(0, extendWidth, 8):
        # 分块
        input8x8 = extendImage[i:i+8, j:j+8]
        # 预测
        predictMaximum = np.zeros(numberOfModes)  # 残差图像的一范数最值
        predictSigma = np.zeros(numberOfModes)    # 残差图像的方差
        error8x8 = np.zeros((numberOfModes, 8, 8))
        for k in range(numberOfModes):
            predictedError = predict(input8x8, k + 1, meanValue)
            error8x8[k, :, :] = predictedError[:, :]
            predictMaximum[k] = np.max(np.abs(error8x8[k, :, :]))
            predictSigma[k] = np.cov(error8x8[k, :, :].reshape(-1))
        bestMode = np.argmin(predictMaximum * (predictSigma + 1)) + 1    # 最小化最大值与方差的乘积，以达到最大压缩比
        modeTable[int(i / 8), int(j / 8)] = bestMode
        # DCT
        dctError8x8 = dct(error8x8[bestMode - 1, :, :])
        # 量化
        quantizationDctError = np.round(quantizationMatrix * dctError8x8 / quantizationScale)
        # zigzag
        zigzagArrayMatrix[int(i/8), int(j/8), :] = zigzag(quantizationDctError)
####################
# 哈夫曼编码
huffmanTree = create_tree(numberOfModes)
codeDict = walk_tree(huffmanTree)
outputHuffmanString = ''
for i in range(int(extendHeight / 8)):
    for j in range(int(extendWidth / 8)):
        outputHuffmanString += codeDict[modeTable[i, j]]
outputHuffmanByteArray = zero_one_to_byte_array(outputHuffmanString)
#####################
# 游程编码
# zigzagArrayMatrix => rlcArray
rlcArray = rlc(zigzagArrayMatrix)
rlcArray_length = len(rlcArray)
#####################
# 算术编码
outputString = arithmetic_encoder(rlcArray)
outputByteArray = zero_one_to_byte_array(outputString)
plt.hist(modeTable.reshape(-1),bins=9)
plt.show()
# plt.imshow(inputImage, cmap=plt.cm.gray)
# plt.show()
# 写入文件
with open(sys.argv[2], 'wb') as outputFile:
    outputFile.write(width.to_bytes(2, byteorder='big'))
    outputFile.write(height.to_bytes(2, byteorder='big'))
    outputFile.write(depth.to_bytes(2, byteorder='big'))
    outputFile.write(struct.pack('>f', meanValue))
    outputFile.write(rlcArray_length.to_bytes(4, byteorder='big'))
    outputFile.write(len(outputHuffmanByteArray).to_bytes(2, byteorder='big'))
    outputFile.write(outputHuffmanByteArray)
    outputFile.write(outputByteArray)


