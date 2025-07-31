import cv2
import face_recognition
import pickle

# Load known encodings
print("[INFO] Loading encodings...")
with open("encodings.pickle", "rb") as f:
    data = pickle.load(f)

# Initialize video stream
print("[INFO] Starting video stream... Press 'q' to quit.")
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("[ERROR] Could not open webcam.")
    exit()

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("[WARNING] Failed to grab frame")
        break

    # Resize frame for faster processing (optional)
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect face locations and encodings
    boxes = face_recognition.face_locations(rgb_small)
    encodings = face_recognition.face_encodings(rgb_small, boxes)
    names = []

    for face_encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"], face_encoding)
        name = "Unknown"

        if True in matches:
            matched_idxs = [i for i, match in enumerate(matches) if match]
            counts = {}

            for i in matched_idxs:
                matched_name = data["names"][i]
                counts[matched_name] = counts.get(matched_name, 0) + 1

            name = max(counts, key=counts.get)

        names.append(name)

    # Scale back face locations since the frame was resized
    for ((top, right, bottom, left), name) in zip(boxes, names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Face Recognition", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

