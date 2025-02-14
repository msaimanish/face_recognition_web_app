from flask import Flask, request, jsonify, render_template
import os
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from database import insert_face, get_all_faces
from datetime import datetime
import pandas as pd
import cv2

app = Flask(__name__)

# Load ResNet Model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove last layer
model.eval()

# Load OpenCV's pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Transform Pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_embedding_dnn(image_path):
    prototxt_path = r"deploy.prototxt"
    model_path = r"res10_300x300_ssd_iter_140000.caffemodel"

    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

    image = cv2.imread(image_path)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Adjust threshold if needed
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x2, y2) = box.astype("int")

            face_roi = image[y:y2, x:x2]

            # Convert to PIL for PyTorch transformations
            face_pil = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
            preprocessed_image = transform(face_pil).unsqueeze(0)

            with torch.no_grad():
                embedding = model(preprocessed_image)

            return embedding.squeeze().cpu().numpy().astype(np.float32)

    print("❌ No face detected using DNN!")
    return None


def extract_embedding(image_path):
    try:
        if not os.path.exists(image_path) or os.path.getsize(image_path) == 0:
            print(f"❌ Image {image_path} is missing or empty!")
            return None

        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Enhance contrast for better face detection
        gray = cv2.equalizeHist(gray)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        if len(faces) == 0:
            print("❌ No face detected! Trying deep learning model...")
            return extract_embedding_dnn(image_path)  # Try DNN-based detection as fallback

        # Crop the largest face
        x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
        face_roi = image[y:y+h, x:x+w]

        # Debug: Show detected face
        cv2.imshow("Detected Face", face_roi)
        cv2.waitKey(500)  # Show for 500ms
        cv2.destroyAllWindows()

        # Convert to PIL for PyTorch transformations
        face_pil = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
        preprocessed_image = transform(face_pil).unsqueeze(0)

        with torch.no_grad():
            embedding = model(preprocessed_image)

        return embedding.squeeze().cpu().numpy().astype(np.float32)

    except Exception as e:
        print(f"❌ Error processing image: {e}")
        return None


@app.route('/')
def index():
    return render_template("index.html")

DATASET_DIR = "dataset"

@app.route("/register", methods=["POST"])
def register():
    student_id = request.form.get("student_id")
    name = request.form.get("name")

    if not student_id or not name:
        return jsonify({"message": "Missing student ID or name"}), 400

    student_dir = os.path.join(DATASET_DIR, student_id)
    os.makedirs(student_dir, exist_ok=True)

    images = [request.files[key] for key in request.files]  # Get all uploaded images

    if not images:
        return jsonify({"message": "No images received"}), 400

    all_embeddings = []

    for idx, img in enumerate(images):
        img_path = os.path.join(student_dir, f"{name}_{idx}.jpg")
        img.save(img_path)

        embedding = extract_embedding(img_path)
        if embedding is not None:
            all_embeddings.append(embedding)

    if not all_embeddings:
        return jsonify({"message": "No valid face embeddings extracted"}), 400

    # Average embeddings for multiple images
    final_embedding = np.mean(all_embeddings, axis=0).tobytes()

    # Store in database
    result = insert_face(student_id, name, final_embedding)
    
    if result == "Student ID already exists":
        return jsonify({"message": result}), 400

    return jsonify({"message": f"Registered {name} successfully"}), 200


@app.route('/verify', methods=['POST'])
def verify():
    if 'image' not in request.files:
        print("❌ No image received for verification")
        return jsonify({"verified": False, "message": "No image uploaded"}), 400

    file = request.files['image']
    image_path = "temp.jpg"
    file.save(image_path)
    print(f"📸 Image saved: {image_path}")

    embedding = extract_embedding(image_path)
    if embedding is None:
        print("❌ Failed to extract face embedding")
        return jsonify({"verified": False, "message": "No face detected"}), 400

    print(f"🛠 Extracted embedding shape: {embedding.shape}")

    faces = get_all_faces()
    if not faces:
        print("❌ No registered faces in the database")
        return jsonify({"verified": False, "message": "No registered faces"}), 400

    known_embeddings = np.array([np.frombuffer(face[2], dtype=np.float32) for face in faces], dtype=np.float32)
    known_ids = [face[0] for face in faces]
    known_names = [face[1] for face in faces]

    # Compute cosine similarity
    known_embeddings = np.array([np.frombuffer(face[2], dtype=np.float32) for face in faces], dtype=np.float32)

    # 🚨 Prevent division by zero
    if known_embeddings.shape[0] == 0 or np.linalg.norm(embedding) == 0:
        print("⚠️ Embeddings are empty or zero vector!")
        return jsonify({"verified": False, "message": "Face embedding issue"}), 400

    # ✅ Ensure all embeddings are normalized
    known_embeddings = known_embeddings / np.linalg.norm(known_embeddings, axis=1, keepdims=True)
    embedding = embedding / np.linalg.norm(embedding)

    # ✅ Compute cosine similarity
    similarities = np.dot(known_embeddings, embedding)

    print(f"📊 Similarity scores: {similarities}")  # Debugging print

    
    match_index = np.argmax(similarities)
    confidence = similarities[match_index]
    print(f"🔍 Best match index: {match_index}, Confidence: {confidence}")

    if confidence > 0.60:
        return jsonify({"verified": True, "name": known_ids[match_index]})
    else:
        return jsonify({"verified": False, "message": "Face not recognized"})


@app.route('/mark-attendance', methods=['POST'])
def mark_attendance():
    file = request.files['image']
    image_path = "temp.jpg"
    file.save(image_path)

    embedding = extract_embedding(image_path)
    if embedding is None or embedding.shape[0] != 512:
        return jsonify({"message": "Face not recognized"}), 400  

    faces = get_all_faces()
    if not faces:
        return jsonify({"message": "No registered faces"}), 400

    known_embeddings = np.array([np.frombuffer(face[2], dtype=np.float32) for face in faces], dtype=np.float32)
    known_ids = [face[0] for face in faces]
    known_names = [face[1] for face in faces]

    # Normalize embeddings for cosine similarity
    known_embeddings = known_embeddings / np.linalg.norm(known_embeddings, axis=1, keepdims=True)
    embedding = embedding / np.linalg.norm(embedding)
    similarities = np.dot(known_embeddings, embedding)

    match_index = np.argmax(similarities)
    confidence = similarities[match_index]

    if confidence > 0.60:
        student_id = known_ids[match_index]
        name = known_names[match_index]
        today_date = datetime.now().strftime("%Y-%m-%d")
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Ensure attendance.csv exists and has correct columns
        csv_path = "attendance.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, dtype={"ID": str})  

            # **Ensure expected columns exist**
            required_columns = {"ID", "Name", "Date", "Time"}
            if not required_columns.issubset(df.columns):
                df = pd.DataFrame(columns=["ID", "Name", "Date", "Time"])  # Reset file if structure is wrong
                df.to_csv(csv_path, index=False)

            # **Check if attendance already exists for today**
            existing_records = df[(df["ID"] == student_id) & (df["Date"] == today_date)]
            if not existing_records.empty:
                recorded_time = existing_records.iloc[0].get('Time', 'Unknown Time')  # Avoid KeyError
                return jsonify({"message": f"Attendance already marked for {name} at {recorded_time}"}), 200

        # **If no attendance recorded today, add the entry**
        new_entry = pd.DataFrame([[student_id, name, today_date, timestamp]], columns=["ID", "Name", "Date", "Time"])
        new_entry.to_csv(csv_path, mode="a", header=not os.path.exists(csv_path), index=False)

        return jsonify({"message": f"Attendance marked for {name} at {timestamp}"}), 200
    else:
        return jsonify({"verified": False, "message": "Face not recognized"}), 200




@app.route('/get-attendance')
def get_attendance():
    date_filter = request.args.get("date")  # Get the selected date from frontend

    try:
        df = pd.read_csv("attendance.csv")

        if date_filter:
            df = df[df["Date"] == date_filter]  # Filter attendance by date

        records = df.to_dict(orient="records")
        return jsonify(records)
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/attendance')
def attendance():
    return render_template("attendance.html")


if __name__ == '__main__':
    app.run(debug=True)
