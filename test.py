import cv2
import numpy as np


def main():
    font = cv2.FONT_HERSHEY_SIMPLEX
    canvas = np.zeros((640, 640, 3), dtype=np.uint8)
    cv2.putText(canvas, str('Y'), (200, 500), font, 18, (255, 255, 255), 20)
    canvas = canvas[:320]
    

    # contours
    contours, _ = cv2.findContours(cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(contour)
    points = cv2.boxPoints(rect)
    points = np.int0(points)
    cv2.drawContours(canvas, [points], -1, (0, 255, 0), 3)
    cv2.imwrite('output.png', canvas)

if __name__ == "__main__":
    main()