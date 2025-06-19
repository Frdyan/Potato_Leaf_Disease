# Import modules
from pathlib import Path
import PIL
import tempfile
import cv2
import base64
import streamlit as st

# Local Modules
import settings
import helper
import asyncio
import sys
import os

# Fix untuk event loop di Windows (Python >= 3.8)
if sys.platform.startswith('win') and sys.version_info >= (3, 8):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Matikan Streamlit file watcher yang menyebabkan error dengan torch.classes
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

# Setting page layout
st.set_page_config(
    page_title="Potato Leaf Disease Detection using YOLOv11",
    page_icon="🥔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Customizing the sidebar with background color
st.markdown("""
    <style>
        .custom-header {
            background-color: #986b4c;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }
        .custom-header h2 {
            color: black;
        }
        .history-image {
            width: 80%;
            max-width: 300px;
            height: 50%;
        }
    </style>
    """, unsafe_allow_html=True)

# Sidebar with custom header     and different background color for the main content
st.sidebar.markdown('<div class="custom-header"><h2>🥔 | Potato Leaf Disease Detection</h2></div>', unsafe_allow_html=True)

page = st.sidebar.selectbox("Select Page", ["🏠 | Home", "🔎 | Detection", "⌛ | History", "ℹ️ | About"], index=0, key='page_selector')

# Home Page
if page == "🏠 | Home":
    st.title("Potato Leaf Disease Project")
    
    st.write("""
        Aplikasi berbasis web ini menggunakan teknologi AI canggih dengan model YOLOv11 untuk mendeteksi 
        dan mengidentifikasi penyakit pada daun kentang secara akurat dan real-time. Dengan sistem deteksi 
        otomatis yang dapat memproses gambar, video, maupun webcam langsung, aplikasi ini membantu petani dan 
        peneliti dalam memantau kesehatan tanaman kentang untuk meningkatkan produktivitas hasil panen.
    """)

    # Gambar utama dengan ukuran besar
    default_image_path = str(settings.DEFAULT_IMAGE)
    default_image = PIL.Image.open(default_image_path)
    st.image(default_image_path, caption="Potato Plant Image by PngTree.com", use_container_width=True)
    
    st.subheader("Deskripsi Halaman:")
    st.write("""
        - **Home**: Halaman utama yang berisi gambaran umum aplikasi dan deskripsi proyek.
        - **Detection**: Halaman untuk mengunggah gambar, video, atau menggunakan webcam untuk mendeteksi penyakit daun kentang menggunakan model YOLOv11.
        - **History**: Halaman untuk melihat riwayat deteksi sebelumnya, termasuk jenis sumber, jalur file, dan hasil gambar yang terdeteksi.
        - **About**: Halaman yang berisi informasi detail tentang proyek, teknologi yang digunakan, dan tim pengembang.
    """)


# Detection Page
elif page == "🔎 | Detection":
    st.title("Corn Disease Detection using YOLOv8")

    st.sidebar.header("ML Model Config")
    confidence = float(st.sidebar.slider("Select Model Confidence (%)", 25, 100, 40)) / 100

    model_path = Path(settings.DETECTION_MODEL)
    try:
        model = helper.load_model(model_path)
    except Exception as ex:
        st.error(f"Unable to load model. Check the specified path: {model_path}")
        st.error(ex)

    st.sidebar.subheader("Image/Video Config")
    source_radio = st.sidebar.radio("Select Source", settings.SOURCES_LIST)

    source_img = None
    source_vid = None

    # If image is selected
    if source_radio == settings.IMAGE:
        source_img = st.sidebar.file_uploader("Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

        col1, col2 = st.columns(2)

        with col1:
            try:
                if source_img is None:
                    default_image_path = str(settings.DEFAULT_UPLOAD_IMAGE)
                    default_image = PIL.Image.open(default_image_path)
                    st.image(default_image_path, caption="Default Image", use_container_width=True)
                else:
                    uploaded_image = PIL.Image.open(source_img)
                    st.image(source_img, caption="Uploaded Image", use_container_width=True)
            except Exception as ex:
                st.error("Error occurred while opening the image.")
                st.error(ex)

        with col2:
            if source_img is None:
                default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
                default_detected_image = PIL.Image.open(default_detected_image_path)
                st.image(default_detected_image_path, caption='Detected Image', use_container_width=True)
            else:
                if st.sidebar.button('Detect Objects'):
                    try:
                        res = model.predict(uploaded_image, conf=confidence)
                        boxes = res[0].boxes
                        res_plotted = res[0].plot()[:, :, ::-1]
                        st.image(res_plotted, caption='Detected Image', use_container_width=True)

                        # Save detection result
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                            PIL.Image.fromarray(res_plotted).save(tmpfile.name)
                            with open(tmpfile.name, "rb") as file:
                                detected_image = file.read()
                                helper.save_detection("Image", source_img.name, detected_image)

                        try:
                            with st.expander("Detection Results"):
                                if boxes:
                                    for box in boxes:
                                        class_id = int(box.cls)
                                        class_name = model.names[class_id]
                                        conf_value = float(box.conf)
                                        st.write(f"Class: {class_name}, Confidence: {conf_value:.2f}")
                                else:
                                    st.write("No objects detected.")
                        except Exception as ex:
                            st.error("Error processing detection results.")
                            st.error(ex)
                    except Exception as ex:
                        st.error("Error running detection.")
                        st.error(ex)

    elif source_radio == settings.VIDEO:
        helper.play_stored_video(confidence, model)

    elif source_radio == settings.WEBCAM:
        helper.play_webcam(confidence, model)

    elif source_radio == settings.YOUTUBE:
        helper.play_youtube_video(confidence, model)

    else:
        st.error("Please select a valid source type!")

# History Page
elif page == "⌛ | History":
    st.title("Detection History")
    history = helper.get_detection_history()

    if not history:
        st.warning("No Detection History")
    else:
        for record in history:
            st.write(f"Source Type: {record.source_type}")
            st.write(f"Source Path: {record.source_path}")
            image_data = base64.b64encode(record.detected_image).decode('utf-8')
            st.markdown(f'<img class="history-image" src="data:image/png;base64,{image_data}" alt="Detected Image">', unsafe_allow_html=True)
            if st.button('Delete', key=f'delete_{record.id}'):
                helper.delete_detection_record(record.id)
                st.rerun()

# About Page
elif page == "ℹ️ | About":
    st.title("About This Project")
    st.write("""
        This project is a web-based application that uses AI technology with the YOLOv11 model to detect and identify diseases on potato leaves accurately and in real-time. 
        The automatic detection system can process images, videos, or webcam feeds directly, helping farmers and researchers monitor the health of potato plants to improve crop productivity.
    """)
    
    st.subheader("Technologies Used:")
    st.write("""
        - **Streamlit**: For building the web application interface.
        - **YOLOv11**: For object detection and disease identification.
        - **OpenCV**: For image processing tasks.
        - **PIL (Pillow)**: For image handling and manipulation.
        - **SQLite**: For storing detection history.
    """)