import cv2
import numpy as np

class hand_processer():
    def __init__(self, img):
        self.img = img

    def is_circle(self, contour, epsilon=0.1):
        area = cv2.contourArea(contour)
        if area == 0:
            return False
        (_, _), radius = cv2.minEnclosingCircle(contour)
        circle_area = np.pi * (radius ** 2)
        return abs(area - circle_area) / circle_area < epsilon

    def get_contours(self):
        ycbcr = cv2.cvtColor(self.img, cv2.COLOR_BGR2YCrCb)
        _, cb, _ = cv2.split(ycbcr)
        _, thresh1 = cv2.threshold(cb, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  
        
        CNTS, _ = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hand = max(CNTS, key=cv2.contourArea)

        circles = [cnt for cnt in CNTS if self.is_circle(cnt)]
        # Assume the target circle is the largest one
        target_circle = max(circles, key=cv2.contourArea)
        return hand, target_circle

    def get_size(self, hand, target_circle):
        actual_area = 2.5 * 2.5 * 3.14  # radius = 2.5cm
        pixel_ratio = actual_area / cv2.contourArea(target_circle)
        return cv2.contourArea(hand) * pixel_ratio

    def process(self):
        if self.img is None:
            print("Image not found")
            return None
        hand, target_circle = self.get_contours()
        return round(self.get_size(hand, target_circle),2)

if __name__ == "__main__":
    img = cv2.imread('./images/image2.jpg')
    hand = hand_processer(img)
    print("The hand area in the image is {} squared centimeters".format(hand.process()))