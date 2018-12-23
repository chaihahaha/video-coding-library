from myfunctions import *
# import matplotlib.pyplot as plt
import sys
import struct
################################
# 测试： 解压
if len(sys.argv) != 3:
    print('请加参数[InputImageFileName] [OutputFileName]')
    assert len(sys.argv) == 3
################################
# 读入压缩数据
with open(sys.argv[1], 'rb') as decodeInputFile:
    inputWidth = int.from_bytes(decodeInputFile.read(2), byteorder='big')
    inputHeight = int.from_bytes(decodeInputFile.read(2), byteorder='big')
    inputDepth = int.from_bytes(decodeInputFile.read(2), byteorder='big')
    extendHeight = 8 * (int((inputHeight - 1) / 8) + 1)
    extendWidth = 8 * (int((inputWidth - 1) / 8) + 1)
    meanValue = struct.unpack('>f', decodeInputFile.read(4))
    rlcArray_length = int.from_bytes(decodeInputFile.read(4), byteorder='big')
    inputHuffmanByteNumber = int.from_bytes(decodeInputFile.read(2), byteorder='big')
    inputHuffmanByteArray = decodeInputFile.read(inputHuffmanByteNumber)
    inputByteArray = decodeInputFile.read()
################
# 哈夫曼解码
inputHuffmanString = byte_array_to_zero_one(inputHuffmanByteArray)
decodeHuffmanTree = create_tree(numberOfModes)
tmpDict = walk_tree(decodeHuffmanTree)
deCodeDict = dict([[v, k] for k, v in tmpDict.items()])
symbolStack = list(inputHuffmanString)
symbolStack.reverse()
key = ''
modeTableHeight = int(extendHeight / 8)
modeTableWidth = int(extendWidth / 8)
modeTable = np.zeros((modeTableHeight, modeTableWidth))
i = 0
while symbolStack and int(i/modeTableWidth) < modeTableHeight and i % modeTableWidth < modeTableWidth:
    key += symbolStack.pop()
    if key in deCodeDict:
        modeTable[int(i/modeTableWidth), i % modeTableWidth] = deCodeDict[key]
        i += 1
        key = ''
#################
# 算术解码
inputString = byte_array_to_zero_one(inputByteArray)
rlcArray = arithmetic_decoder(inputString, rlcArray_length)
#################
# 游程解码
zigzagArrayMatrix = de_rlc(rlcArray, extendHeight, extendWidth)
########################
# 分块反变换，反量化
outputExtend = np.zeros((extendHeight, extendWidth))
for i in range(0, extendHeight, 8):
    for j in range(0, extendWidth, 8):
        #################################
        # 对zigzag向量转换为图像
        # decodeZigZagArray => deZigzagErrorImage
        deZigzagErrorImage = de_zigzag(quantizationScale * zigzagArrayMatrix[int(i/8), int(j/8), :], 8, 8)
        ################################
        # 反DCT变换
        # deZigzagErrorImage => idctErrorImage8x8
        idctErrorImage8x8 = idct(deZigzagErrorImage)
        outputExtend[i: i + 8, j: j + 8] = de_predict(idctErrorImage8x8, modeTable[int(i / 8), int(j / 8)], meanValue)

outputImage = outputExtend[: inputHeight, : inputWidth]
maximum = np.max(outputImage)
outputImage[outputImage < 0] = 0
if maximum > 255:
    outputImage = 255*outputImage/maximum
############################
# 写入PGM文件
with open(sys.argv[2], 'wb') as decodedFile:
    decodedFile.write(b'P5\n')
    decodedFile.write((str(inputWidth)+' '+str(inputHeight)+'\n'+str(inputDepth)+'\n').encode('utf-8'))
    for i in range(inputHeight):
        for j in range(inputWidth):
            decodedFile.write(int(outputImage[i, j]).to_bytes(1, byteorder='little'))
###########################
# 显示解码后的图像
# plt.imshow(outputImage, cmap=plt.cm.gray)
# plt.show()
