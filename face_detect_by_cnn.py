import cv2
import dlib


def write_to_disk(image, face_coordinates):

    # this function will save the cropped image from original photo on disk
    for (x1, y1, w, h) in face_coordinates:
        cropped_face = image[y1:y1 + h, x1:x1 + w]
        cv2.imwrite(str(y1) + ".jpg", cropped_face)


def draw_fancy_box(img, pt1, pt2, color, thickness, r, d):
    # To draw some fancy box around founded faces in stream
    x1, y1 = pt1
    x2, y2 = pt2

    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)


# Realtime face detection using dlib cnn trained model is very slow but good performance over frontal face


def face_detection_realtime():
    cap = cv2.VideoCapture(0)

    while True:

        # Getting out image by webcam
        _, image = cap.read()

        # Converting the image to gray scale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Load the CNN face detector model
        cnn_face_detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")

        # Get faces into webcam's image
        rects = cnn_face_detector(gray, 0)

        face_coordinates = []
        # For each detected face
        for (i, rect) in enumerate(rects):
            # Finding points for rectangle to draw on face
            x1, y1, x2, y2, w, h = rect.rect.left(), rect.rect.top(), rect.rect.right() + \
                                                                      1, rect.rect.bottom() + 1, rect.rect.width(), rect.rect.height()

            # drawing rectangle around face
            draw_fancy_box(image, (x1, y1), (x2, y2), (127, 255, 255), 2, 10, 20)

            # Drawing simple rectangle around found faces
            # cv2.rectangle(image, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)

            face_coordinates.append((x1, y1, w, h))

            # show the face number
            cv2.putText(image, "Face #{}".format(i + 1), (x1 - 20, y1 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (51, 51, 255), 2)

        # Show the image
        cv2.imshow("Output", image)

        # To capture found faces from camera
        if cv2.waitKey(30) & 0xFF == ord('s'):
            write_to_disk(image, face_coordinates)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()
