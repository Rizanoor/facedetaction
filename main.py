import cv2

# Load face detection model
faceRef = cv2.CascadeClassifier("faceRef.xml")

# Load age detection model
ageNet = cv2.dnn.readNetFromCaffe("deploy_age.prototxt", "age_net.caffemodel")

# Define age ranges
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

camera = cv2.VideoCapture(0)

def faceDetection(frame):
    optimizeFrame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = faceRef.detectMultiScale(optimizeFrame, scaleFactor=1.1, minSize=(500, 500))
    return faces

def agePrediction(face):
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
    ageNet.setInput(blob)
    agePreds = ageNet.forward()
    age = ageList[agePreds[0].argmax()]
    return age

def drawerBox(frame):
    for x, y, w, h in faceDetection(frame):
        face = frame[y:y+h, x:x+w]
        age = agePrediction(face)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 4)
        cv2.putText(frame, f'Riza, {age}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

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
