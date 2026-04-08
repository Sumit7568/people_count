import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import smtplib
from email.mime.text import MIMEText

# ==============================
# CONFIG
# ==============================
THRESHOLD = 1

SENDER_EMAIL = ""
APP_PASSWORD = ""
RECEIVER_EMAIL = ""

# ==============================
# LOAD MODEL
# ==============================
model = YOLO("yolov8n.pt")

# ==============================
# EMAIL FUNCTION
# ==============================
def send_email_alert(count):
    try:
        msg = MIMEText(f"🚨 ALERT!\nPeople count exceeded.\nCurrent Count: {count}")
        msg["Subject"] = "People Count Alert"
        msg["From"] = SENDER_EMAIL
        msg["To"] = RECEIVER_EMAIL

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(SENDER_EMAIL, APP_PASSWORD)
        server.send_message(msg)
        server.quit()

    except Exception as e:
        print("Email Error:", e)

# ==============================
# SESSION STATE (for email control)
# ==============================
if "email_sent" not in st.session_state:
    st.session_state.email_sent = False

# ==============================
# UI
# ==============================
st.title("👥 People Counting from Video")

uploaded_file = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])

frame_placeholder = st.empty()
status_placeholder = st.empty()

# ==============================
# PROCESS VIDEO
# ==============================
if uploaded_file is not None:

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, stream=True)

        count = 0

        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == 0:
                    count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # ==============================
        # FRAME TEXT
        # ==============================
        cv2.putText(frame, f"People Count: {count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # ==============================
        # STATUS + EMAIL ALERT
        # ==============================
        if count > THRESHOLD:
            status_text = f"People Count: {count} | Status: ALERT 🚨"
            status_placeholder.error(status_text)

            # Send email only once
            if not st.session_state.email_sent:
                send_email_alert(count)
                st.session_state.email_sent = True

        else:
            status_text = f"People Count: {count} | Status: Normal ✅"
            status_placeholder.success(status_text)
            st.session_state.email_sent = False  # reset when normal

        # Convert BGR → RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_placeholder.image(frame, channels="RGB")

    cap.release()