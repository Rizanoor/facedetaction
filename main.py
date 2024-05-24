import cv2

faceRef = cv2.CascadeClassifier("faceRef.xml")
camera = cv2.VideoCapture(0)

def faceDetection(frame):
    optimizeFrame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = faceRef.detectMultiScale(optimizeFrame, scaleFactor=1.1,
                                     minSize=(500, 500))
    return faces

def drawerBox(frame):
    for x, y, w, h, in faceDetection(frame):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 4)
        cv2.putText(frame, 'Riza', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    
def closeWindow():
    camera.release()
    cv2.destroyAllWindows()
    exit

def main():
    while True:
        _, frame = camera.read()
        drawerBox(frame)
        cv2.imshow("face detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            closeWindow()

if __name__ == '__main__':
    main()