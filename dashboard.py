"""
Smart Office Face Recognition System for Employee Access
Computer Vision Assignment - CI7523
Kingston University

Student Name: [Your Name]
Student Number: [Your Student Number]
Module Leader: Prof Jamshid Dehmeshki

This system marks attendance by identifying a person's face in real-time.
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import time
from PIL import Image
import json
import os

from detection import get_detector
from recognition import get_recogniser
from registration import FaceDatabase
from evaluation import run_evaluation

# Page configuration
st.set_page_config(
    page_title="Office Attendance System - CI7523",
    page_icon="📋",
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
    .attendance-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 1rem;
        color: white;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize components
detector = get_detector()
recogniser = get_recogniser()
database = FaceDatabase()

# Attendance file path
ATTENDANCE_FILE = "attendance_records.json"

# Session state initialization
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False
if 'cap' not in st.session_state:
    st.session_state.cap = None
if 'recognition_logs' not in st.session_state:
    st.session_state.recognition_logs = []
if 'attendance_today' not in st.session_state:
    st.session_state.attendance_today = []
if 'marked_today' not in st.session_state:
    st.session_state.marked_today = set()

# Load attendance records
def load_attendance():
    if os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, 'r') as f:
            return json.load(f)
    return []

# Save attendance records
def save_attendance(records):
    with open(ATTENDANCE_FILE, 'w') as f:
        json.dump(records, f, indent=2)

# Mark attendance
def mark_attendance(name, confidence):
    today = datetime.now().strftime("%Y-%m-%d")
    now = datetime.now().strftime("%H:%M:%S")
    
    attendance = load_attendance()
    
    # Check if already marked today
    for record in attendance:
        if record['name'] == name and record['date'] == today:
            return False, "Already marked today"
    
    # Add new attendance record
    attendance.append({
        'name': name,
        'date': today,
        'time': now,
        'confidence': f"{confidence:.2%}",
        'status': 'Present'
    })
    
    save_attendance(attendance)
    return True, f"Attendance marked at {now}"

# Get today's attendance
def get_today_attendance():
    today = datetime.now().strftime("%Y-%m-%d")
    attendance = load_attendance()
    return [r for r in attendance if r['date'] == today]

# Professional header
st.markdown("""
<div class="main-header">
    <h1>📋 Office Attendance System</h1>
    <p>Face Recognition Based Employee Attendance Tracking</p>
    <p style="font-size: 0.9rem; margin-top: 0.5rem;">Kingston University London | CI7523</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 📋 System Menu")
    
    menu = st.radio(
        "Select Feature",
        [
            "📋 Mark Attendance",
            "📊 View Attendance Report",
            "📝 Register Employee",
            "👥 Employee Database",
            "📸 Test Recognition",
            "📊 System Evaluation"
        ]
    )
    
    st.markdown("---")
    
    st.markdown("### 📊 Today's Statistics")
    today_attendance = get_today_attendance()
    total_employees = len(database.get_all_embeddings())
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Present Today", len(today_attendance))
    with col2:
        st.metric("Total Employees", total_employees)
    
    if total_employees > 0:
        attendance_rate = (len(today_attendance) / total_employees) * 100
        st.progress(attendance_rate / 100, text=f"Attendance Rate: {attendance_rate:.1f}%")
    
    st.markdown("---")
    st.markdown("### 🧠 AI Models")
    st.info("Detection: MTCNN\n\nRecognition: FaceNet\n\nEmbeddings: 128D")
    
    st.markdown("---")
    st.markdown("### 📝 Recent Activity")
    for log in st.session_state.recognition_logs[-3:]:
        st.text(f"• {log}")

# ==================== MARK ATTENDANCE ====================
if menu == "📋 Mark Attendance":
    st.markdown("## 📋 Mark Your Attendance")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🎥 Look at the Camera to Mark Attendance")
        
        status_placeholder = st.empty()
        metrics_placeholder = st.empty()
        frame_placeholder = st.empty()
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        with col_btn2:
            start_btn = st.button("🎬 START CAMERA", use_container_width=True, type="primary")
            stop_btn = st.button("🛑 STOP CAMERA", use_container_width=True)
        
        if start_btn:
            st.session_state.camera_active = True
            st.session_state.cap = cv2.VideoCapture(0)
            
            if not st.session_state.cap.isOpened():
                status_placeholder.markdown('<div class="status-error">❌ Cannot access camera. Please check permissions.</div>', unsafe_allow_html=True)
                st.session_state.camera_active = False
            else:
                status_placeholder.markdown('<div class="status-success">✅ Camera active. Looking for face to mark attendance...</div>', unsafe_allow_html=True)
        
        if stop_btn:
            st.session_state.camera_active = False
            if st.session_state.cap:
                st.session_state.cap.release()
                st.session_state.cap = None
            status_placeholder.empty()
            metrics_placeholder.empty()
            frame_placeholder.empty()
            st.success("Camera stopped!")
        
        if st.session_state.camera_active and st.session_state.cap:
            frame_count = 0
            start_time = time.time()
            last_marked_time = 0
            
            while st.session_state.camera_active:
                ret, frame = st.session_state.cap.read()
                
                if not ret:
                    break
                
                frame_count += 1
                
                if frame_count % 2 == 0:
                    faces = detector.detect(frame)
                    
                    with metrics_placeholder.container():
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("👤 Faces Detected", len(faces))
                        with col_b:
                            fps = frame_count / (time.time() - start_time)
                            st.metric("⚡ FPS", f"{fps:.1f}")
                        with col_c:
                            st.metric("📊 Status", "Scanning" if faces else "Waiting")
                    
                    for face in faces:
                        face_img = face["face_160"]
                        embedding = recogniser.extract_embedding(face_img)
                        
                        name = "Unknown"
                        best_score = 0
                        
                        for person, db_emb in database.get_all_embeddings():
                            if len(embedding) == len(db_emb):
                                score = recogniser.compare(embedding, db_emb)
                                if score > best_score and score > 0.55:
                                    best_score = score
                                    name = person
                        
                        x, y, w, h = [int(v) for v in face["box"]]
                        
                        if name != "Unknown" and best_score > 0.6:
                            color = (0, 255, 0)
                            label = f"{name} ({best_score:.2%})"
                            
                            # Mark attendance (once per session per person)
                            current_time = time.time()
                            if name not in st.session_state.marked_today and (current_time - last_marked_time) > 2:
                                success, message = mark_attendance(name, best_score)
                                if success:
                                    st.session_state.marked_today.add(name)
                                    last_marked_time = current_time
                                    status_placeholder.markdown(f'<div class="status-success">✅ {message}</div>', unsafe_allow_html=True)
                                    st.session_state.recognition_logs.append(f"{datetime.now().strftime('%H:%M:%S')} - {name} attendance marked")
                                    # Refresh attendance display
                                    st.rerun()
                                else:
                                    status_placeholder.markdown(f'<div class="status-warning">⚠️ {message}</div>', unsafe_allow_html=True)
                        else:
                            color = (0, 165, 255)
                            label = "Unknown Person - Not Registered"
                        
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    frame_placeholder.image(frame, channels="BGR", use_container_width=True)
                
                time.sleep(0.03)
            
            if st.session_state.cap:
                st.session_state.cap.release()
                st.session_state.cap = None
            st.session_state.camera_active = False
    
    with col2:
        st.markdown("### 📋 Today's Attendance")
        today_attendance = get_today_attendance()
        
        if today_attendance:
            for record in today_attendance:
                st.markdown(f"""
                <div style="background: #f0f2f6; padding: 0.8rem; border-radius: 10px; margin-bottom: 0.5rem;">
                    <strong>👤 {record['name']}</strong><br>
                    ⏰ Time: {record['time']}<br>
                    📊 Confidence: {record['confidence']}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No attendance marked yet today. Look at the camera to mark your attendance.")
        
        st.markdown("---")
        st.markdown("### 📋 Instructions")
        st.info(
            "**How to Mark Attendance:**\n\n"
            "1️⃣ Click 'START CAMERA'\n"
            "2️⃣ Look directly at the camera\n"
            "3️⃣ Wait for face detection\n"
            "4️⃣ System will automatically mark attendance\n"
            "5️⃣ Green box indicates successful recognition\n\n"
            "**Note:** Each employee can only mark attendance once per day."
        )

# ==================== VIEW ATTENDANCE REPORT ====================
elif menu == "📊 View Attendance Report":
    st.markdown("## 📊 Attendance Report")
    st.markdown("---")
    
    attendance = load_attendance()
    
    if not attendance:
        st.markdown('<div class="status-warning">⚠️ No attendance records found.</div>', unsafe_allow_html=True)
    else:
        # Date filter
        dates = sorted(set([r['date'] for r in attendance]), reverse=True)
        selected_date = st.selectbox("Select Date", dates)
        
        # Filter by date
        filtered = [r for r in attendance if r['date'] == selected_date]
        
        # Statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Present", len(filtered))
        with col2:
            total_employees = len(database.get_all_embeddings())
            st.metric("Total Employees", total_employees)
        with col3:
            rate = (len(filtered) / total_employees * 100) if total_employees > 0 else 0
            st.metric("Attendance Rate", f"{rate:.1f}%")
        
        st.markdown("---")
        
        # Display attendance table
        df_data = []
        for idx, record in enumerate(filtered, 1):
            df_data.append({
                "S.No": idx,
                "Employee Name": record['name'],
                "Date": record['date'],
                "Time": record['time'],
                "Confidence": record['confidence'],
                "Status": record['status']
            })
        
        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Export option
        if st.button("📥 Export to CSV", use_container_width=True):
            csv = df.to_csv(index=False)
            st.download_button(
                label="💾 Download CSV",
                data=csv,
                file_name=f"attendance_{selected_date}.csv",
                mime="text/csv"
            )
        
        # Summary chart
        st.markdown("### 📊 Monthly Summary")
        monthly_data = {}
        for record in attendance:
            month = record['date'][:7]
            if month not in monthly_data:
                monthly_data[month] = 0
            monthly_data[month] += 1
        
        if monthly_data:
            summary_df = pd.DataFrame([
                {"Month": m, "Total Attendance": c} for m, c in monthly_data.items()
            ])
            st.bar_chart(summary_df.set_index("Month"))

# ==================== REGISTER EMPLOYEE ====================
elif menu == "📝 Register Employee":
    st.markdown("## 📝 Register New Employee")
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        name = st.text_input("Employee Name *", placeholder="Enter full name")
        employee_id = st.text_input("Employee ID *", placeholder="EMP001")
        department = st.selectbox("Department", ["Engineering", "Sales", "Marketing", "HR", "Operations", "IT", "Finance"])
        position = st.selectbox("Position", ["Employee", "Team Lead", "Manager", "Executive"])
        
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

# ==================== EMPLOYEE DATABASE ====================
elif menu == "👥 Employee Database":
    st.markdown("## 👥 Employee Database")
    st.markdown("---")
    
    users = database.get_all_embeddings()
    
    if not users:
        st.markdown('<div class="status-warning">⚠️ No registered employees found.</div>', unsafe_allow_html=True)
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

# ==================== TEST RECOGNITION ====================
elif menu == "📸 Test Recognition":
    st.markdown("## 📸 Test Face Recognition")
    st.markdown("---")
    
    uploaded_file = st.file_uploader("Upload an image to test recognition", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="Test Image", use_container_width=True)
        
        with col2:
            with st.spinner("Analyzing..."):
                uploaded_file.seek(0)
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                if img_cv is None:
                    st.error("Failed to decode image.")
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
                        
                        st.dataframe(pd.DataFrame(results), use_container_width=True)
                        st.image(img_cv, channels="BGR", caption="Recognition Result", use_container_width=True)
                    else:
                        st.markdown('<div class="status-warning">⚠️ No faces detected.</div>', unsafe_allow_html=True)

# ==================== SYSTEM EVALUATION ====================
elif menu == "📊 System Evaluation":
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

# ==================== ACTIVITY LOG ====================
st.markdown("---")
st.markdown("## 📋 System Activity Log")

if st.session_state.recognition_logs:
    log_df = pd.DataFrame({"Timestamp & Event": st.session_state.recognition_logs[-10:]})
    st.dataframe(log_df, use_container_width=True)
    
    if st.button("Clear Logs"):
        st.session_state.recognition_logs = []
        st.rerun()
else:
    st.info("No activity recorded yet.")

# ==================== FOOTER ====================
st.markdown("""
<div class="footer">
    <p><strong>Office Attendance System (CI7523)</strong> | Kingston University London</p>
    <p>Face Recognition Based Attendance Tracking | MTCNN + FaceNet</p>
</div>
""", unsafe_allow_html=True)