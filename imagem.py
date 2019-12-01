import cv2
import numpy as np
from matplotlib import pyplot as plt

cinza = cv2.imread('bola.jpeg', cv2.IMREAD_GRAYSCALE)
_, binaria = cv2.threshold(cinza, 220, 255, cv2.THRESH_BINARY_INV)

kernel = np.ones((3, 3), np.uint8)

erosao = cv2.erode(binaria, kernel, iterations=1)
dilatacao = cv2.dilate(binaria, kernel, iterations=1)
abertura = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel)
fechamento = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel)
gradiente = cv2.morphologyEx(binaria, cv2.MORPH_GRADIENT, kernel)
tophat = cv2.morphologyEx(binaria, cv2.MORPH_TOPHAT, kernel)
blackhat = cv2.morphologyEx(binaria, cv2.MORPH_BLACKHAT, kernel)

titles = ['Original', 'Cinza', 'Binária', 'Dilatação', 'Erosão', 'Abertura', 'Fechamento', 'Gradiente', 'TopHat', 'BlackHat']
images = [cv2.imread('bola.jpeg'), cinza, binaria, dilatacao, erosao, abertura, fechamento, gradiente, tophat, blackhat]

for i in range(10):
    plt.subplot(2, 5, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

plt.show()

cv2.waitKey(0)