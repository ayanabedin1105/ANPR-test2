import cv2

# Path to the pre-trained Haar cascade classifier for license plate detection
harcascade = "model/haarcascade_russian_plate_number.xml"

# Load the cascade classifier
plate_cascade = cv2.CascadeClassifier(harcascade)

# Initialize video capture
cap = cv2.VideoCapture("sample.mp4")


cap.set(3, 640) #width camera
cap.set(4, 480) #camera height

# Minimum area of the plate
min_area = 500
count = 0

while True:
    #Read a frame from the video
    success, img = cap.read()
    
    if not success:
        break

    # Convert the frame to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect license plates
    plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

    for (x, y, w, h) in plates:
        area = w * h

        if area > min_area:
            # Draw a rectangle around the detected plate
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, "Number plate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)

            # Display the region of interest (ROI)
            img_roi = img[y:y + h, x:x + w]
            cv2.imshow("ROI", img_roi)

    # Display the result
    cv2.imshow("Result", img)

    # Save the frame containing the detected plate
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("plate_folder/scanned_img_" + str(count) + ".jpg", img_roi)
        cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, "Plate Saved", (150, 150), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
        cv2.imshow("Results", img)
        cv2.waitKey(500)
        count += 1

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()