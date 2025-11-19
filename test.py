import cv2

for i in [1, 2]:
    cap = cv2.VideoCapture(i)
    ret, frame = cap.read()
    if ret:
        print(f"Index {i}: Working")
        cv2.imshow(f"Camera {i}", frame)
        cv2.waitKey(3000)
    else:
        print(f"Index {i}: Not usable")
    cap.release()

cv2.destroyAllWindows()
