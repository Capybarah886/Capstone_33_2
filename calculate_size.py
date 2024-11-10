import cv2
import numpy as np

class HandProcesser:
    def __init__(self, img):
        if img is None:
            raise ValueError("Image cannot be None")
        self.img = img

    def is_circle(self, contour, epsilon=0.15):
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
        if not CNTS:
            raise ValueError("No contours found in the image")

        hand = max(CNTS, key=cv2.contourArea)
        if cv2.contourArea(hand) == 0:
            raise ValueError("Hand contour area is zero")

        circles = [cnt for cnt in CNTS if self.is_circle(cnt)]
        if not circles:
            return hand, None

        target_circle = max(circles, key=cv2.contourArea)
        return hand, target_circle

    def get_size(self, hand, target_circle):
        if target_circle is None:
            raise ValueError("Target circle is None")
        actual_area = 1.6*1.6 * 3.14  # radius = 1.8cm
        pixel_ratio = actual_area / cv2.contourArea(target_circle)
        return cv2.contourArea(hand) * pixel_ratio

    def process(self):
        try:
            hand, target_circle = self.get_contours()
            return round(self.get_size(hand, target_circle), 2)
        except ValueError as e:
            print(f"Error: {e}")
            return None

if __name__ == "__main__":
    img = cv2.imread('./images/image3.jpg')
    try:
        hand = HandProcesser(img)
        result = hand.process()
        if result is not None:
            print(f"The hand area in the image is {result} squared centimeters")
    except ValueError as e:
        print(f"Initialization Error: {e}")