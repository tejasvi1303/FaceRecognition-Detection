import os
import face_recognition
import pickle

dataset_dir = "dataset"
encoding_file = "encodings.pickle"

known_encodings = []
known_names = []

for image_name in os.listdir(dataset_dir):
    if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue  # skip non-image files

    name = os.path.splitext(image_name)[0]  # filename without extension
    image_path = os.path.join(dataset_dir, image_name)

    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)

    if encodings:
        known_encodings.append(encodings[0])
        known_names.append(name)
        print(f"[INFO] Encoded {image_name} as {name}")
    else:
        print(f"[WARNING] No face found in {image_name}")

# Save encoding data
data = {"encodings": known_encodings, "names": known_names}
with open(encoding_file, "wb") as f:
    pickle.dump(data, f)

print("[INFO] Encoding complete and saved to encodings.pickle.")
