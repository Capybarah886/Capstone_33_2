import base64
import json
import cv2
import numpy as np
import sys
from skimage.filters import threshold_multiotsu

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
    def detect_background(self, image):
        # find the background region based on its greatest frequency
        unique, counts = np.unique(image, return_counts=True)
        for i in range(len(unique)):
            print(f"Value: {unique[i]}, Frequency: {counts[i]}")
        background_value = unique[np.argmax(counts)]
        print(f"Background value: {background_value}")
        masked_image = np.where(image == background_value, 0, 255)
        return masked_image

    def get_contours(self):
        ycbcr = cv2.cvtColor(self.img, cv2.COLOR_BGR2YCrCb)
        _, cb, _ = cv2.split(ycbcr)
        thresholds=threshold_multiotsu(cb, classes=3)
        thresh1 = np.digitize(cb, bins=thresholds)
        masked_image = self.detect_background(thresh1)
        masked_image_8uc1 = masked_image.astype('uint8')

        # _, thresh1 = cv2.threshold(cb, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  
        
        CNTS, _ = cv2.findContours(masked_image_8uc1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

    def main(self):
        try:
            hand, target_circle = self.get_contours()
            return round(self.get_size(hand, target_circle), 2)
        except ValueError as e:
            print(f"Error: {e}")
            return None

def b64_to_img(b64):
    img = base64.b64decode(b64)
    npimg = np.fromstring(img, dtype=np.uint8)
    return cv2.imdecode(npimg, cv2.IMREAD_COLOR)

def lambda_handler(event, context):
    img = event['body']
    img_b64=b64_to_img(img)
    try:
        hand = HandProcesser(img_b64)
        result = hand.main()
        if result is not None:
            return {
                'statusCode': 200,
                 'body': json.dumps(f"The hand area in the image is {result} squared centimeters"),
    }
    except ValueError as e:
        print(f"Initialization Error: {e}")
        return {
            'statusCode': 400,
            'body': json.dumps('No data return')
        }
if __name__ == "__main__":
    img_path = ".\images\image5.jpg"
    img = cv2.imread(img_path)

    # Create an instance of HandProcesser
    hand = HandProcesser(img)

    # Call the process method
    result = hand.main()

    # Assert that the result is not None

    # Print the result
    print(f"The hand area in the image is {result} squared centimeters")

