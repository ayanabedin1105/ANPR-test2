import cv2

cascade_files = ["model/haarcascade_russian_plate_number.xml",
                 "model/indian_license_plate.xml"]


def my_function():
    cap = cv2.VideoCapture(0)

    cap.set(3, 640)  # width camera
    cap.set(4, 480)  # camera height

    min_plate_width = 80
    min_plate_height = 20
    min_area = 500
    count = 0

    while True:
        success, img = cap.read()

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Loop through each cascade classifier
        for cascade_file in cascade_files:
            plate_cascade = cv2.CascadeClassifier(cascade_file)

            plates = plate_cascade.detectMultiScale(
                img_gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_plate_width, min_plate_height))

            for (x, y, w, h) in plates:
                area = w * h

                if area > min_area:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(img, "Number plate", (x, y - 5),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)

                    img_roi = img[y:y + h, x:x + w]
                    cv2.imshow("ROI", img_roi)

        cv2.imshow("Result", img)

        if cv2.waitKey(1) & 0xFF == ord('s'):
            if 'img_roi' in locals():
                cv2.imwrite("plate_folder/scanned_img_" +
                            str(count) + ".jpg", img_roi)
                cv2.rectangle(img, (0, 200), (640, 300),
                              (0, 255, 0), cv2.FILLED)
                cv2.putText(img, "Plate Saved", (150, 150),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
                cv2.imshow("Results", img)
                cv2.waitKey(500)
                count += 1

    return "Python script executed!"


# Call the function
my_function()
