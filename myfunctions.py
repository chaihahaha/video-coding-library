# 导出要使用的模块
import numpy as np
import matplotlib.pyplot as plt

################################
# 设置程序算法内置参数
numberOfModes = 9  # 实际使用的预测模式数
precision = 27  # 精度过小时解码会进入死循环
whole = 2 ** precision
half = int(whole / 2)
quarter = int(half / 2)

# 由于精度问题以下参数应满足exp(- symbolMax/(sigma * quantizationScale)) > 1e -20 即 sigma > symbolMax / (quantizationScale * 20)
quantizationScale = 10
symbolMax = 900  # 残差图像符号集大小，过大时解码时间长，过小时会无法覆盖实际符号
sigma = 100  # 过小时会用完精度，解码进入死循环
# 计算频率表
# (0, 1)区间内插入的小数是游程编码的码元
symbols = np.hstack([np.arange(-int(symbolMax/quantizationScale), 0), np.round(np.linspace(0.01, 0.64, 64), 2), np.arange(1, int(symbolMax/quantizationScale))])
frequencyTable = np.vstack([np.exp(np.abs(symbols) / sigma), symbols]).T  # 符号频率分布为拉普拉斯分布，使用拉普拉斯分布公式代替计算频率表
# frequencyTable[:, 0] = frequencyTable[:, 0] / np.sum(frequencyTable[:, 0])
###############################
# 定义数据结构
class HuffmanNode(object):
    def __init__(self, left=None, right=None, root=None):
        self.left = left
        self.right = right
        self.root = root  # Why?  Not needed for anything.

    def children(self):
        return self.left, self.right


# 定义函数
# 游程编码函数
def rlc(zigzagArrayMatrix):
    max_i, max_j, max_k = zigzagArrayMatrix.shape
    rlcArray = []
    count_zeros = 0
    for i in range(max_i):
        for j in range(max_j):
            for k in range(max_k):
                if zigzagArrayMatrix[i, j, k] != 0.0:
                    if count_zeros > 0:
                        rlcArray.append(0.01 * count_zeros)
                        count_zeros = 0

                    rlcArray.append(zigzagArrayMatrix[i, j, k])
                else:
                    count_zeros += 1
            if count_zeros > 0:
                rlcArray.append(0.01 * count_zeros)
                count_zeros = 0
    rlcArray = np.round(rlcArray, 2)
    return rlcArray


# 反游程编码函数
def de_rlc(rlcArray, extendHeight, extendWidth):
    length = len(rlcArray)
    i = 0
    j = 0
    k = 0
    max_i = int(extendHeight / 8)
    max_j = int(extendWidth / 8)
    max_k = 64
    zigzagArrayMatrix = np.zeros((max_i, max_j, 64))
    for rlc_i in range(length):
        if 0 < rlcArray[rlc_i] < 1:
            k += int(round(rlcArray[rlc_i] / 0.01)) - 1  # 必须四舍五入取整，否则会出现int(29.0)=28的问题，根本原因是二进制符点数的近似问题
        else:
            zigzagArrayMatrix[i, j, k] = rlcArray[rlc_i]
        k += 1
        if k >= max_k:
            k = k - max_k
            j += 1
        if j >= max_j:
            j = j - max_j
            i += 1
        if i >= max_i:
            assert rlc_i < length
            break
    return zigzagArrayMatrix


# 01字符串转字节数组，便于写入文件
def zero_one_to_byte_array(outputString):
    outputByteArray = bytearray()
    outputBitNumber = len(outputString)
    for k in range(0, outputBitNumber, 8):
        outputByteArray.append(int(outputString[k: k + 8], 2))
    return outputByteArray


# 字节数组转01字符串，便于从文件读取数据并解码
def byte_array_to_zero_one(inputByteArray):
    inputString = ''
    for i in range(len(inputByteArray)):
        inputString += format(inputByteArray[i], '08b')
    return inputString


# 自适应算术编码中重新计算概率表的函数
def compute_c_d_R(frequencyTable, symbol):
    index = np.argwhere(frequencyTable[:, 1] == symbol)
    frequencyTable[index[0, 0], 0] = frequencyTable[index[0, 0], 0] + 0.1
    c = np.zeros(frequencyTable.shape)
    c[:, 1] = frequencyTable[:, 1]
    c[0, 0] = 0
    for i in range(1, len(frequencyTable)):
        c[i, 0] = frequencyTable[i - 1, 0] + c[i - 1, 0]
    d = np.zeros(c.shape)
    d[:, 1] = c[:, 1]
    d[:, 0] = c[:, 0] + frequencyTable[:, 0]
    c = {c[i, 1]: c[i, 0] for i in range(len(c))}
    d = {d[i, 1]: d[i, 0] for i in range(len(d))}
    return c, d, d[symbols[-1]]


# 自适应算术编码器
def arithmetic_encoder(zigzagArray):
    # 算术编码
    # 设置算术编码参数
    c = np.zeros(frequencyTable.shape)
    c[:, 1] = frequencyTable[:, 1]
    c[0, 0] = 0
    R = np.sum(frequencyTable[:, 0])
    for i in range(1, len(frequencyTable)):
        c[i, 0] = frequencyTable[i - 1, 0] + c[i - 1, 0]
    d = np.zeros(c.shape)
    d[:, 1] = c[:, 1]
    d[:, 0] = c[:, 0] + frequencyTable[:, 0]
    c = {c[i, 1]: c[i, 0] for i in range(len(c))}
    d = {d[i, 1]: d[i, 0] for i in range(len(d))}
    a = int()
    b = int(whole)
    s = int()
    outputString = ''
    length = len(zigzagArray)
    for i in range(length):
        w = b - a
        if w == 0 or w == 1:
            print("precision error")
        b = a + int(w * d[round(zigzagArray[i], 2)] / R)
        a = a + int(w * c[round(zigzagArray[i], 2)] / R)
        c, d, R = compute_c_d_R(frequencyTable, zigzagArray[i])
        while b < half or a > half:
            if b < half:
                outputString += '0' + s * '1'
                s = 0
                a = int(2 * a)
                b = int(2 * b)
            elif a > half:
                outputString += '1' + s * '0'
                s = 0
                a = 2 * int(a - half)
                b = 2 * int(b - half)
        while a > quarter and b < 3 * quarter:
            s = s + 1
            a = 2 * (a - quarter)
            b = 2 * (b - quarter)
    s = s + 1
    if a <= quarter:
        outputString += '0' + s * '1'
    else:
        outputString += '1' + s * '0'
    return outputString


# 自适应解码器
def arithmetic_decoder(inputString, length):
    zigzagArray = np.zeros(length)
    # 设置算术编码参数
    c = np.zeros(frequencyTable.shape)
    c[:, 1] = frequencyTable[:, 1]
    c[0, 0] = 0
    R = np.sum(frequencyTable[:, 0])
    for i in range(1, len(frequencyTable)):
        c[i, 0] = frequencyTable[i - 1, 0] + c[i - 1, 0]
    d = np.zeros(c.shape)
    d[:, 1] = c[:, 1]
    d[:, 0] = c[:, 0] + frequencyTable[:, 0]
    c = {c[i, 1]: c[i, 0] for i in range(len(c))}
    d = {d[i, 1]: d[i, 0] for i in range(len(d))}
    a = int(0)
    b = int(whole)
    M = len(inputString)
    z = 0
    i = 1
    zk = 0
    while i <= precision and i <= M:
        if inputString[i - 1] == '1':
            z = z + 2**(precision - i)
        i += 1
    while zk < length:
        for j in frequencyTable[:, 1]:
            w = b - a
            if w == 0 or w == 1:
                print("precision error")
            b0 = a + int(w * d[j] / R)
            a0 = a + int(w * c[j] / R)
            if a0 <= z < b0:
                zigzagArray[zk] = j
                zk += 1

                a = a0
                b = b0
                c, d, R = compute_c_d_R(frequencyTable, j)
                break
        while b < half or a > half:
            if b < half:
                a = 2 * a
                b = 2 * b
                z = 2 * z
            elif a > half:
                a = 2 * (a - half)
                b = 2 * (b - half)
                z = 2 * (z - half)
            if i <= M and inputString[i - 1] == '1':
                z = z + 1
            i = i + 1
        while a > quarter and b < 3 * quarter:
            a = 2 * (a - quarter)
            b = 2 * (b - quarter)
            z = 2 * (z - quarter)
            if i <= M and inputString[i - 1] == '1':
                z = z + 1
            i = i + 1
    return zigzagArray


# 预测编码器，mode为传入预测编码模式编号
def predict(inputImage, mode, meanValue):
    assert numberOfModes <= 9
    height, width = inputImage.shape
    errorImage = np.zeros((height, width))
    if mode == 1:
        # print('mode 1')
        errorImage[0, :] = inputImage[0, :] - meanValue
        errorImage[1:, :] = inputImage[1:, :] - inputImage[0, :]
    elif mode == 2:
        # print('mode 2')
        errorImage[:, 0] = inputImage[:, 0] - meanValue
        errorImage[:, 1:] = (inputImage[:, 1:].T - inputImage[:, 0]).T
    elif mode == 3:
        # print('mode 3')
        errorImage = inputImage[:, :] - meanValue
    elif mode == 4:
        # print('mode 4')
        errorImage[0, :] = inputImage[0, :] - meanValue
        errorImage[:, width - 1] = inputImage[:, width - 1] - meanValue
        for i in range(1, height, 1):
            for j in range(0, width - 1, 1):
                if int((i + j)/(width - 1)) >= 1:
                    errorImage[i, j] = inputImage[i, j] - inputImage[i + j - (width - 1), width - 1]
                else:
                    errorImage[i, j] = inputImage[i, j] - inputImage[0, i + j]
    elif mode == 5:
        errorImage[0, :] = inputImage[0, :] - meanValue
        errorImage[:, : 2] = inputImage[:, : 2] - meanValue
        for i in range(1, height, 1):
            for j in range(2, width, 1):
                predict_i = max(i - j >> 1, 0)
                predict_j = j - 2 * (i - predict_i)
                errorImage[i, j] = inputImage[i, j] - inputImage[predict_i, predict_j]
    elif mode == 6:
        errorImage[0, :] = inputImage[0, :] - meanValue
        errorImage[:, 0] = inputImage[:, 0] - meanValue
        for i in range(1, height, 1):
            for j in range(1, width, 1):
                predict_i = max(i - j, 0)
                predict_j = j - i + predict_i
                errorImage[i, j] = inputImage[i, j] - inputImage[predict_i, predict_j]
    elif mode == 7:
        errorImage[:, width - 1] = inputImage[:, width - 1] - meanValue
        errorImage[: 2, :] = inputImage[: 2, :] - meanValue
        for i in range(2, height, 1):
            for j in range(width - 2, -1, -1):
                if int((i + 1) / 2) + j < width - 1:
                    predict_i = i % 2
                    predict_j = int((i + 1) / 2) + j - int((predict_i + 1) / 2)
                else:
                    predict_j = width - 1
                    predict_i = 2 * (int((i + 1) / 2) + j - predict_j) - i % 2
                errorImage[i, j] = inputImage[i, j] - inputImage[predict_i, predict_j]
    elif mode == 8:
        errorImage[:, :2] = inputImage[:, :2] - meanValue
        errorImage[height - 1, :] = inputImage[height - 1, :] - meanValue
        for j in range(2, width, 1):
            for i in range(0, height - 1, 1):
                if int((j + 1) / 2) + i < height - 1:
                    predict_j = j % 2
                    predict_i = int((j + 1) / 2) + i - int((predict_j + 1) / 2)
                else:
                    predict_i = height - 1
                    predict_j = 2 * (int((j + 1) / 2) + i - predict_i) - j % 2
                errorImage[i, j] = inputImage[i, j] - inputImage[predict_i, predict_j]
    elif mode == 9:
        errorImage[:2, :] = inputImage[:2, :] - meanValue
        errorImage[:, 0] = inputImage[:, 0] - meanValue
        for i in range(2, height, 1):
            for j in range(1, width, 1):
                if j - i >> 1 > 0:
                    predict_j = j - i >> 1
                    predict_i = i % 2
                else:
                    predict_j = 0
                    predict_i = 2 * (predict_j + int((i + 1) / 2) - j) - i % 2
                errorImage[i, j] = inputImage[i, j] - inputImage[predict_i, predict_j]
    return errorImage


# 预测解码器，mode为输入预测编码模式
def de_predict(errorImage, mode, meanValue):
    assert numberOfModes <= 9
    inputHeight, inputWidth = errorImage.shape
    outputImage = np.zeros((inputHeight, inputWidth))
    if mode == 1:
        # print('mode 1')
        outputImage[0, :] = errorImage[0, :inputWidth] + meanValue
        outputImage[1: inputHeight, :] = errorImage[1: inputHeight, :] + outputImage[0, :]
    elif mode == 2:
        # print('mode 2')
        outputImage[:, 0] = errorImage[:inputHeight, 0] + meanValue
        outputImage[:, 1: inputWidth] = (errorImage[:, 1: inputWidth].T + outputImage[:, 0]).T
    elif mode == 3:
        # print('mode 3')
        outputImage = errorImage + meanValue
    elif mode == 4:
        # print('mode 4')
        outputImage[0, :] = errorImage[0, :] + meanValue
        outputImage[:, inputWidth - 1] = errorImage[:, inputWidth - 1] + meanValue
        for i in range(1, inputHeight, 1):
            for j in range(0, inputWidth - 1, 1):
                if int((i + j) / (inputWidth - 1)) >= 1:
                    outputImage[i, j] = errorImage[i, j] + outputImage[i + j - (inputWidth - 1), inputWidth - 1]
                else:
                    outputImage[i, j] = errorImage[i, j] + outputImage[0, i + j]
    elif mode == 5:
        outputImage[0, :] = errorImage[0, :inputWidth] + meanValue
        outputImage[:, : 2] = errorImage[:inputHeight, : 2] + meanValue
        for i in range(1, inputHeight, 1):
            for j in range(2, inputWidth, 1):
                predict_i = max(i - j >> 1, 0)
                predict_j = j - 2 * (i - predict_i)
                outputImage[i, j] = errorImage[i, j] + outputImage[predict_i, predict_j]
    elif mode == 6:
        outputImage[0, :] = errorImage[0, :inputWidth] + meanValue
        outputImage[:, 0] = errorImage[:inputHeight, 0] + meanValue
        for i in range(1, inputHeight, 1):
            for j in range(1, inputWidth, 1):
                predict_i = max(i - j, 0)
                predict_j = j - i + predict_i
                outputImage[i, j] = errorImage[i, j] + outputImage[predict_i, predict_j]
    elif mode == 7:
        outputImage[:, inputWidth - 1] = errorImage[:inputHeight, inputWidth - 1] + meanValue
        outputImage[: 2, :] = errorImage[: 2, :inputWidth] + meanValue
        for i in range(2, inputHeight, 1):
            for j in range(inputWidth - 2, -1, -1):
                if int((i + 1) / 2) + j < inputWidth - 1:
                    predict_i = i % 2
                    predict_j = int((i + 1) / 2) + j - int((predict_i + 1) / 2)
                else:
                    predict_j = inputWidth - 1
                    predict_i = 2 * (int((i + 1) / 2) + j - predict_j) - i % 2
                outputImage[i, j] = errorImage[i, j] + outputImage[predict_i, predict_j]
    elif mode == 8:
        outputImage[:, :2] = errorImage[:inputHeight, :2] + meanValue
        outputImage[inputHeight - 1, :] = errorImage[inputHeight - 1, :inputWidth] + meanValue
        for j in range(2, inputWidth, 1):
            for i in range(0, inputHeight - 1, 1):
                if int((j + 1) / 2) + i < inputHeight - 1:
                    predict_j = j % 2
                    predict_i = int((j + 1) / 2) + i - int((predict_j + 1) / 2)
                else:
                    predict_i = inputHeight - 1
                    predict_j = 2 * (int((j + 1) / 2) + i - predict_i) - j % 2
                outputImage[i, j] = errorImage[i, j] + outputImage[predict_i, predict_j]
    elif mode == 9:
        outputImage[:2, :] = errorImage[:2, :inputWidth] + meanValue
        outputImage[:, 0] = errorImage[:inputHeight, 0] + meanValue
        for i in range(2, inputHeight, 1):
            for j in range(1, inputWidth, 1):
                if j - i >> 1 > 0:
                    predict_j = j - i >> 1
                    predict_i = i % 2
                else:
                    predict_j = 0
                    predict_i = 2 * (predict_j + int((i + 1) / 2) - j) - i % 2
                outputImage[i, j] = errorImage[i, j] + outputImage[predict_i, predict_j]
    return outputImage


# DCT变换函数
def dct(matrix):
    m, n = matrix.shape
    N = n
    C_temp = np.zeros(matrix.shape)
    C_temp[0, :] = 1 * np.sqrt(1 / N)
    for i in range(1, m):
        for j in range(n):
            C_temp[i, j] = np.cos(np.pi * i * (2 * j + 1) / (2 * N)) * np.sqrt(2 / N)
    dst = np.dot(np.dot(C_temp, matrix), np.transpose(C_temp))
    return dst


# 反DCT变换函数
def idct(dst):
    m, n = dst.shape
    N = n
    C_temp = np.zeros(dst.shape)
    C_temp[0, :] = 1 * np.sqrt(1 / N)
    for i in range(1, m):
        for j in range(n):
            C_temp[i, j] = np.cos(np.pi * i * (2 * j + 1) / (2 * N)) * np.sqrt(2 / N)
    img_recor = np.dot(np.transpose(C_temp), dst)
    img_recor1 = np.dot(img_recor, C_temp)
    return img_recor1


# 遍历哈夫曼树，并产生码表
def walk_tree(node, prefix="", code={}):
    if node[1].left is not None and isinstance(node[1].left[1], HuffmanNode):
        walk_tree(node[1].left, prefix + "0", code)
    else:
        code[node[1].left[1]] = prefix + "0"
    if node[1].right is not None and isinstance(node[1].right[1], HuffmanNode):
        walk_tree(node[1].right, prefix + "1", code)
    else:
        code[node[1].right[1]] = prefix + "1"
    return code


# 构造哈夫曼树
def create_tree(numberOfModes):
    # numberOfModes = 4
    frequencies = [(400, 1), (400, 2), (3000, 3), (100, 4), (50, 5), (100, 6), (60, 7), (80, 8), (20, 9)]
    tuples = []
    for value in frequencies:
        tuples.append(value)
        tuples.sort(key=lambda t: t[0], reverse=True)
    while len(tuples) > 1:
        l, r = tuples.pop(), tuples.pop()
        node = HuffmanNode(l, r)
        tuples.append((l[0] + r[0], node))
    return tuples.pop()


# 将8x8矩阵转为zigzag数组
def zigzag(M):
    height, width = M.shape
    zz = np.zeros(height * width)
    i = 0
    u = 0
    v = 0
    zz[i] = M[u, v]
    i += 1
    while u < height and v < width:
        v += 1
        if u < height and v < width:
            zz[i] = M[u, v]
            i += 1
        elif u < height - 1 and v > 0:
            v -= 1
            u += 1
            zz[i] = M[u, v]
            i += 1
        else:
            break
        while u < height - 1 and 0 < v < width:
            v -= 1
            u += 1
            zz[i] = M[u, v]
            i += 1
        u += 1
        if u < height and v < width:
            zz[i] = M[u, v]
            i += 1
        elif u > 0 and v < width - 1:
            u -= 1
            v += 1
            zz[i] = M[u, v]
            i += 1
        while 0 < u < height and v < width - 1:
            u -= 1
            v += 1
            zz[i] = M[u, v]
            i += 1
    return zz


# 反zigzag，输出8x8矩阵
def de_zigzag(zz, height, width):
    u = 0
    v = 0
    M = np.zeros((height, width))
    i = 0
    M[u, v] = zz[i]
    i += 1
    while u < height and v < width:
        v += 1
        if u < height and v < width:
            M[u, v] = zz[i]
            i += 1
        elif u < height - 1 and v > 0:
            v -= 1
            u += 1
            M[u, v] = zz[i]
            i += 1
        else:
            break
        while u < height - 1 and 0 < v < width:
            v -= 1
            u += 1
            M[u, v] = zz[i]
            i += 1
        u += 1
        if u < height and v < width:
            M[u, v] = zz[i]
            i += 1
        elif u > 0 and v < width - 1:
            u -= 1
            v += 1
            M[u, v] = zz[i]
            i += 1
        while 0 < u < height and v < width - 1:
            u -= 1
            v += 1
            M[u, v] = zz[i]
            i += 1
    return M
