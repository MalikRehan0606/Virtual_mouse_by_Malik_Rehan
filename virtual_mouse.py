import cv2
import numpy as np
import time
import HandTracking as ht  # Assuming HandTracking module is defined elsewhere
import autopy   # Install using "pip install autopy"
import pyautogui  # Install using "pip install pyautogui"

### Variables Declaration
pTime = 0               # Used to calculate frame rate
width = 640             # Width of Camera
height = 480            # Height of Camera
frameR = 100            # Frame Reduction
smoothening = 7         # Smoothening Factor (tune this value as needed)
prev_x, prev_y = 0, 0   # Previous coordinates
curr_x, curr_y = 0, 0   # Current coordinates

cap = cv2.VideoCapture(0)   # Getting video feed from the webcam
cap.set(3, width)           # Adjusting size
cap.set(4, height)

detector = ht.handDetector(maxHands=1)  # Initialize hand detector (assuming from HandTracking module)
screen_width, screen_height = autopy.screen.size()  # Getting the screen size

# Function to take screenshot
def take_screenshot():
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    screenshot_path = f"screenshot_{timestamp}.png"
    screenshot = pyautogui.screenshot()
    screenshot.save(screenshot_path)
    print(f"Screenshot saved as {screenshot_path}")

while True:
    success, img = cap.read()
    img = detector.findHands(img)  # Finding the hand
    lmlist, bbox = detector.findPosition(img)  # Getting position of hand

    if len(lmlist) != 0:
        x1, y1 = lmlist[8][1:]   # Index finger tip coordinates
        x5, y5 = lmlist[20][1:]  # Pinky finger tip coordinates

        fingers = detector.fingersUp()  # Checking if fingers are upwards
        cv2.rectangle(img, (frameR, frameR), (width - frameR, height - frameR), (255, 0, 255), 2)  # Creating boundary box

        # Movement Mode: Index finger up, pinky finger down
        if fingers[1] == 1 and fingers[4] == 0:
            x1_mapped = np.interp(x1, (frameR, width - frameR), (0, screen_width))
            y1_mapped = np.interp(y1, (frameR, height - frameR), (0, screen_height))

            curr_x = prev_x + (x1_mapped - prev_x) / smoothening
            curr_y = prev_y + (y1_mapped - prev_y) / smoothening

            autopy.mouse.move(screen_width - curr_x, curr_y)  # Moving the cursor
            prev_x, prev_y = curr_x, curr_y
            cv2.circle(img, (x1, y1), 7, (255, 0, 255), cv2.FILLED)  # Visual feedback for cursor movement

        # Click Mode: Index finger and middle finger both up and close to each other
        if fingers[1] == 1 and fingers[2] == 1:
            length, img, lineInfo = detector.findDistance(8, 12, img)

            if length < 40:  # If both fingers are really close to each other
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()  # Perform click

        # Screenshot Mode: Index finger and pinky finger both up
        if fingers[1] == 1 and fingers[4] == 1:
            length, img, lineInfo = detector.findDistance(8, 20, img)

            if length < 40:  # If both fingers are really close to each other
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                take_screenshot()  # Perform screenshot capture
                time.sleep(1)  # Avoid multiple screenshots in quick succession

        # Scrolling Mode: Index finger and ring finger up (scroll up)
        if fingers[1] == 1 and fingers[3] == 1 and fingers[2] == 0 and fingers[4] == 0:
            pyautogui.scroll(20)  # Scroll up

        # Scrolling Mode: Middle finger and pinky finger up (scroll down)
        if fingers[2] == 1 and fingers[4] == 1 and fingers[1] == 0 and fingers[3] == 0:
            pyautogui.scroll(-20)  # Scroll down

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
