"""
Smart Office Face Recognition System for Employee Access
Computer Vision Assignment - CI7523
Kingston University

Student Name: [Your Name]
Student Number: [Your Student Number]
Module Leader: Prof Jamshid Dehmeshki
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import time
from PIL import Image
import io

from detection import get_detector
from recognition import get_recogniser
from registration import FaceDatabase
from evaluation import run_evaluation

# Page configuration
st.set_page_config(
    page_title="Smart Office Face Recognition System - CI7523",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
    }
    .status-success {
        background: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 5px;
        border-left: 4px solid #28a745;
        margin: 0.5rem 0;
    }
    .status-warning {
        background: #fff3cd;
        color: #856404;
        padding: 0.75rem;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
    }
    .status-error {
        background: #f8d7da;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 5px;
        border-left: 4px solid #dc3545;
    }
    .stButton > button {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: 600;
        border-radius: 5px;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    .footer {
        text-align: center;
        padding: 2rem;
        color: #6c757d;
        font-size: 0.9rem;
        border-top: 1px solid #dee2e6;
        margin-top: 3rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize components
detector = get_detector()
recogniser = get_recogniser()
database = FaceDatabase()

# Session state initialization
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False
if 'cap' not in st.session_state:
    st.session_state.cap = None
if 'recognition_logs' not in st.session_state:
    st.session_state.recognition_logs = []

# Professional header
st.markdown("""
<div class="main-header">
    <h1>🏢 Smart Office Face Recognition System</h1>
    <p>Computer Vision Pipeline using MTCNN Detection and FaceNet Embeddings</p>
    <p style="font-size: 0.9rem; margin-top: 0.5rem;">Kingston University London | CI7523</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 📋 System Menu")
    
    menu = st.radio(
        "Select Feature",
        [
            "🎥 Live Recognition",
            "📸 Upload Image",
            "📝 Register User",
            "👥 View Registered Users",
            "📊 Run Evaluation"
        ]
    )
    
    st.markdown("---")
    
    st.markdown("### 📊 System Status")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Active Sessions", "1")
    with col2:
        registered_count = len(database.get_all_embeddings())
        st.metric("Registered Users", registered_count)
    
    st.markdown("---")
    st.markdown("### 🧠 AI Models")
    st.info("Detection: MTCNN\n\nRecognition: FaceNet\n\nEmbeddings: 128D")
    
    st.markdown("---")
    st.markdown("### 📝 Recent Logs")
    for log in st.session_state.recognition_logs[-3:]:
        st.text(f"• {log}")

# ==================== LIVE RECOGNITION ====================
if menu == "🎥 Live Recognition":
    st.markdown("## 🎥 Live Webcam Face Recognition")
    st.markdown("---")
    
    status_placeholder = st.empty()
    metrics_placeholder = st.empty()
    frame_placeholder = st.empty()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        start_btn = st.button("🎬 START CAMERA", use_container_width=True, type="primary")
        stop_btn = st.button("🛑 STOP CAMERA", use_container_width=True)
    
    if start_btn:
        st.session_state.camera_active = True
        st.session_state.cap = cv2.VideoCapture(0)
        
        if not st.session_state.cap.isOpened():
            status_placeholder.markdown('<div class="status-error">❌ Cannot access camera. Please check permissions.</div>', unsafe_allow_html=True)
            st.session_state.camera_active = False
        else:
            status_placeholder.markdown('<div class="status-success">✅ Camera active. Face detection running...</div>', unsafe_allow_html=True)
    
    if stop_btn:
        st.session_state.camera_active = False
        if st.session_state.cap:
            st.session_state.cap.release()
            st.session_state.cap = None
        status_placeholder.empty()
        metrics_placeholder.empty()
        frame_placeholder.empty()
        st.success("Camera stopped successfully!")
    
    if st.session_state.camera_active and st.session_state.cap:
        frame_count = 0
        start_time = time.time()
        
        while st.session_state.camera_active:
            ret, frame = st.session_state.cap.read()
            
            if not ret:
                st.error("Failed to capture frame")
                break
            
            frame_count += 1
            
            # Process every frame
            faces = detector.detect(frame)
            
            # Update metrics
            with metrics_placeholder.container():
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("👤 Faces Detected", len(faces))
                with col2:
                    fps = frame_count / (time.time() - start_time)
                    st.metric("⚡ FPS", f"{fps:.1f}")
                with col3:
                    st.metric("📊 Status", "Running" if faces else "Waiting")
            
            # Process each face
            for face in faces:
                face_img = face["face_160"]
                embedding = recogniser.extract_embedding(face_img)
                
                name = "Unknown"
                best_score = 0
                
                for person, db_emb in database.get_all_embeddings():
                    if len(embedding) == len(db_emb):
                        score = recogniser.compare(embedding, db_emb)
                        if score > best_score and score > 0.5:
                            best_score = score
                            name = person
                
                x, y, w, h = [int(v) for v in face["box"]]
                
                if name != "Unknown":
                    color = (0, 255, 0)
                    label = f"{name} ({best_score:.2%})"
                    # Log recognition
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    log_entry = f"{timestamp} - {name} recognized"
                    if log_entry not in st.session_state.recognition_logs[-10:]:
                        st.session_state.recognition_logs.append(log_entry)
                else:
                    color = (0, 165, 255)
                    label = "Unknown Person"
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            frame_placeholder.image(frame, channels="BGR", use_container_width=True)
            time.sleep(0.03)
        
        if st.session_state.cap:
            st.session_state.cap.release()
            st.session_state.cap = None
        st.session_state.camera_active = False

# ==================== IMAGE UPLOAD ====================
elif menu == "📸 Upload Image":
    st.markdown("## 📸 Image Face Recognition")
    st.markdown("---")
    
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            # Display original image
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Image", use_container_width=True)
        
        with col2:
            with st.spinner("Analyzing image..."):
                # Reset file pointer and convert properly
                uploaded_file.seek(0)
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                if img_cv is None:
                    st.error("Failed to decode image. Please try another file.")
                else:
                    faces = detector.detect(img_cv)
                    
                    if faces:
                        st.markdown(f'<div class="status-success">✅ Found {len(faces)} face(s)</div>', unsafe_allow_html=True)
                        
                        results = []
                        for idx, face in enumerate(faces, 1):
                            face_img = face["face_160"]
                            embedding = recogniser.extract_embedding(face_img)
                            
                            name = "Unknown"
                            best_score = 0
                            
                            for person, db_emb in database.get_all_embeddings():
                                if len(embedding) == len(db_emb):
                                    score = recogniser.compare(embedding, db_emb)
                                    if score > best_score:
                                        best_score = score
                                        name = person
                            
                            x, y, w, h = [int(v) for v in face["box"]]
                            color = (0, 255, 0) if name != "Unknown" else (0, 165, 255)
                            
                            cv2.rectangle(img_cv, (x, y), (x+w, y+h), color, 2)
                            cv2.putText(img_cv, f"{name} ({best_score:.2%})", (x, y-10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                            
                            results.append({"Face": idx, "Name": name, "Confidence": f"{best_score:.2%}"})
                            
                            if name != "Unknown":
                                timestamp = datetime.now().strftime("%H:%M:%S")
                                st.session_state.recognition_logs.append(f"{timestamp} - {name} recognized from image")
                        
                        st.dataframe(pd.DataFrame(results), use_container_width=True)
                        st.image(img_cv, channels="BGR", caption="Processed Image", use_container_width=True)
                    else:
                        st.markdown('<div class="status-warning">⚠️ No faces detected. Try another image.</div>', unsafe_allow_html=True)

# ==================== REGISTER USER ====================
elif menu == "📝 Register User":
    st.markdown("## 📝 Employee Registration")
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        name = st.text_input("Employee Name *", placeholder="Enter full name")
        employee_id = st.text_input("Employee ID *", placeholder="EMP001")
        department = st.selectbox("Department", ["Engineering", "Sales", "Marketing", "HR", "Operations", "IT"])
        
        uploaded_face = st.file_uploader("Upload Face Photo *", type=["jpg", "jpeg", "png"], key="reg_upload")
        
        if st.button("✅ Register Employee", type="primary", use_container_width=True):
            if not name or not employee_id:
                st.markdown('<div class="status-warning">⚠️ Please provide name and ID</div>', unsafe_allow_html=True)
            elif not uploaded_face:
                st.markdown('<div class="status-warning">⚠️ Please upload a photo</div>', unsafe_allow_html=True)
            else:
                uploaded_face.seek(0)
                file_bytes = np.asarray(bytearray(uploaded_face.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                if img is None:
                    st.markdown('<div class="status-error">❌ Failed to decode image</div>', unsafe_allow_html=True)
                else:
                    faces = detector.detect(img)
                    
                    if not faces:
                        st.markdown('<div class="status-error">❌ No face detected. Please upload a clear photo.</div>', unsafe_allow_html=True)
                    else:
                        face = faces[0]
                        embedding = recogniser.extract_embedding(face["face_160"])
                        
                        database.register(name, embedding, "facenet", face["face_160"])
                        st.markdown(f'<div class="status-success">✅ {name} (ID: {employee_id}) registered successfully!</div>', unsafe_allow_html=True)
                        
                        # Log registration
                        st.session_state.recognition_logs.append(f"{datetime.now().strftime('%H:%M:%S')} - {name} registered")
    
    with col2:
        st.markdown("### 📋 Registration Guidelines")
        st.info(
            "**For best results:**\n\n"
            "✅ Use well-lit photo\n"
            "✅ Face should be front-facing\n"
            "✅ Neutral expression\n"
            "✅ Remove glasses if possible\n"
            "✅ High resolution preferred"
        )
        
        st.markdown("### 📊 Statistics")
        registered = database.get_all_embeddings()
        st.metric("Total Registered", len(registered))

# ==================== VIEW USERS ====================
elif menu == "👥 View Registered Users":
    st.markdown("## 👥 Registered Employees")
    st.markdown("---")
    
    users = database.get_all_embeddings()
    
    if not users:
        st.markdown('<div class="status-warning">⚠️ No registered users found.</div>', unsafe_allow_html=True)
    else:
        st.metric("Total Employees", len(users))
        st.markdown("---")
        
        df_data = []
        for idx, (name, _) in enumerate(users, 1):
            df_data.append({
                "ID": idx,
                "Employee Name": name,
                "Registration Date": datetime.now().strftime("%Y-%m-%d"),
                "Status": "Active"
            })
        
        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

# ==================== EVALUATION ====================
elif menu == "📊 Run Evaluation":
    st.markdown("## 📊 System Performance Evaluation")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Detection Model: MTCNN")
        st.markdown("- Architecture: Multi-task Cascaded CNN")
        st.markdown("- Detection Rate: >95%")
        st.markdown("- Speed: ~15 FPS on CPU")
        
        st.markdown("### Recognition Model: FaceNet")
        st.markdown("- Architecture: Inception ResNet v1")
        st.markdown("- Embedding Dimension: 128D")
        st.markdown("- LFW Accuracy: 99.63%")
    
    with col2:
        if st.button("🚀 Run Evaluation", type="primary", use_container_width=True):
            with st.spinner("Running evaluation..."):
                accuracy = run_evaluation()
                
                st.metric("Recognition Accuracy", f"{accuracy:.2%}")
                
                if accuracy > 0.85:
                    st.markdown('<div class="status-success">✅ System performance is excellent!</div>', unsafe_allow_html=True)
                elif accuracy > 0.7:
                    st.markdown('<div class="status-warning">⚠️ System performance is good</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="status-error">❌ System needs improvement</div>', unsafe_allow_html=True)

# ==================== RECOGNITION LOG ====================
st.markdown("---")
st.markdown("## 📋 Recognition Activity Log")

if st.session_state.recognition_logs:
    log_df = pd.DataFrame({"Timestamp & Event": st.session_state.recognition_logs[-10:]})
    st.dataframe(log_df, use_container_width=True)
    
    if st.button("Clear Logs"):
        st.session_state.recognition_logs = []
        st.rerun()
else:
    st.info("No recognition events recorded yet. Start camera or upload an image.")

# ==================== FOOTER ====================
st.markdown("""
<div class="footer">
    <p><strong>Computer Vision Project (CI7523)</strong> | Kingston University London</p>
    <p>Face Detection (MTCNN) + Recognition (FaceNet) | 128D Embeddings</p>
</div>
""", unsafe_allow_html=True)