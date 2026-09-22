import cv2
import numpy as np
import os
import math

print('OpenCV verzió:', cv2.__version__)
print('numpy: ', np.__version__)

# főbb változók
images = []
imgPath = []
currentImgIndex = 0
windowName = 'Palcikak'

# zajos képek
noisyImgs = {}
# alap
tackbarValues = {'S': -1, 'P': -1, 'Gauss': -1}

'''
def arrays_are_equal(arr1, arr2):
    if len(arr1) != len(arr2):
        return False
    for index in range(len(arr1)):
        if arr1[index] != arr2[index]:
            return False
    return True
'''
# összes kép fájl a mappában és main.py fájl abszolút útvonala
BASEDIR = os.path.dirname(os.path.abspath(__file__))

pathToImages = os.path.join(BASEDIR, 'testimg')

if not os.path.exists(pathToImages):
    raise Exception(f"A megadott mappa nem létezik: {pathToImages}")

for i in os.listdir(pathToImages):
    if i.lower().endswith(('.jpg', '.jpeg', '.png')):
        path = os.path.join(pathToImages, i)
        print("reading current:", path)
        placeHolderImg = cv2.imread(path)

        if placeHolderImg is not None:
            images.append(placeHolderImg)
            imgPath.append(path)
        else:
            raise Exception(f'Nem sikerült beolvasni a képet: {path}')

if len(images) == 0:
    raise Exception('Nincs érvényes képfájl a mappában!')


def trackbarvaluechange(value):
    s = cv2.getTrackbarPos('S', windowName)
    p = cv2.getTrackbarPos('P', windowName)
    g = cv2.getTrackbarPos('Gauss', windowName)
    print(f'értékek: s: {s}, p: {p}, g: {g}')
    pass


# ablak
cv2.namedWindow(windowName)

# csúszkák
cv2.createTrackbar('S', windowName, 0, 100, trackbarvaluechange)
cv2.createTrackbar('P', windowName, 0, 100, trackbarvaluechange)
cv2.createTrackbar('Gauss', windowName, 0, 100, trackbarvaluechange)

while True:
    # X-el bezárásra
    if cv2.getWindowProperty(windowName, cv2.WND_PROP_VISIBLE) < 1:
        break

    # csúszka értékek
    noiseSalt = cv2.getTrackbarPos('S', windowName)
    noisePepper = cv2.getTrackbarPos('P', windowName)
    noiseGauss = cv2.getTrackbarPos('Gauss', windowName)

    # változás van-e valamelyikben
    if (noiseSalt != tackbarValues['S'] or
            noisePepper != tackbarValues['P'] or
            noiseGauss != tackbarValues['Gauss']):
        noisyImgs.clear()  # reset
        tackbarValues = {'S': noiseSalt, 'P': noisePepper, 'Gauss': noiseGauss}

    # kép tárolása ha nem létezik
    if currentImgIndex not in noisyImgs:
        # ne az eredtit piszkáljam
        placeholderImg = images[currentImgIndex].copy()

        # salt & pepper
        probabilitySalt = noiseSalt / 200.0
        maskSalt = np.random.rand(placeholderImg.shape[0], placeholderImg.shape[1])
        placeholderImg[maskSalt < probabilitySalt] = 255
        placeholderImg = cv2.medianBlur(placeholderImg, 3)

        probabilityPepper = noisePepper / 200.0
        maskPepper = np.random.rand(placeholderImg.shape[0], placeholderImg.shape[1])
        placeholderImg[maskPepper < probabilityPepper] = 0
        placeholderImg = cv2.medianBlur(placeholderImg, 3)

        # gauss
        if noiseGauss > 0:
            # olvashatóbb zaj, 1-nél
            probabilityGauss = noiseGauss * 0.9
            gaussNoise = np.random.normal(0, probabilityGauss, placeholderImg.shape)
            placeholderImg = np.clip(placeholderImg.astype(float) + gaussNoise, 0, 255).astype(np.uint8)
            placeholderImg = cv2.GaussianBlur(placeholderImg, (5, 5), 2.0)

        noisyImgs[currentImgIndex] = placeholderImg

        currentImg = noisyImgs[currentImgIndex]
        # szürke hogy lehessen éldetektálást csinálni
        gray = cv2.cvtColor(currentImg, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 190)

        kernel = np.ones((14, 14), np.uint8)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        lines = cv2.HoughLinesP(closed, 1, np.pi / 180, 120, np.array([]), 200, 20)

        lineListFinal = lines
        if lines is not None:

            cleanLines = [line[0] for line in lines]
            cleanLines.sort(key=lambda l: (l[2] - l[0]) ** 2 + (l[3] - l[1]) ** 2, reverse=True)

            lineListFinal = []

            while len(cleanLines) > 0:
                currentLine = cleanLines.pop(0)
                lineListFinal.append(currentLine)

                x1, y1, x2, y2 = currentLine
                lineLength = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                if lineLength == 0:
                    continue

                remaining = []

                for check in cleanLines:
                    cx1, cy1, cx2, cy2 = check

                    d1 = abs((y2 - y1) * cx1 - (x2 - x1) * cy1 + x2 * y1 - y2 * x1) / lineLength
                    d2 = abs((y2 - y1) * cx2 - (x2 - x1) * cy2 + x2 * y1 - y2 * x1) / lineLength

                    if d1 < 30 and d2 < 30:
                        continue
                    else:
                        remaining.append(check)

                cleanLines = remaining

            for line in lineListFinal:
                x1, y1, x2, y2 = line
                cv2.line(currentImg, (x1, y1), (x2, y2), (0, 0, 255), 3)

        print(len(lineListFinal))
        cv2.putText(currentImg, ("palcikak: " + str(len(lineListFinal))), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2)
        cv2.imshow(windowName, currentImg)

    key = cv2.waitKey(10) & 0xFF

    # következő kép
    if key == ord('d'):
        currentImgIndex += 1
        if currentImgIndex >= len(images):
            currentImgIndex = 0
            noisyImgs.clear()  # törlöm hogy ne akadjon el
    # előző kép
    elif key == ord('a'):
        currentImgIndex -= 1
        if currentImgIndex < 0:
            currentImgIndex = len(images) - 1
            noisyImgs.clear()
    # kilépés (q vagy Esc)
    elif (key == ord('q')) or (key == 27):
        break

cv2.destroyAllWindows()
