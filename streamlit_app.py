import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
import matplotlib.pyplot as plt
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import io
import av

# ============================================================================
# CONFIGURATION & BRANDING
# ============================================================================
APP_TITLE = "DIP & Emotion Detection Laboratory"
APP_SUBTITLE = "Advanced Digital Image Processing & AI-Powered Emotion Recognition"
APP_ICON = "🔬"
APP_AUTHOR = "Ayush"
BRAND_URL = "https://github.com/Ayush-Raj189"
EMAIL = "ayushashish1111@gmail.com"

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# PROFESSIONAL STYLING
# ============================================================================
st.markdown("""
<style>
    /* Global Theme */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * {
        font-family: 'Inter', sans-serif;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 100%);
    }

    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
        animation: fadeInDown 0.8s ease-out;
    }

    .main-header h1 {
        color: white;
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }

    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        margin-top: 0.5rem;
        font-weight: 300;
    }

    /* Feature Cards */
    .feature-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border: 2px solid rgba(102, 126, 234, 0.3);
        border-radius: 16px;
        padding: 1.5rem;
        height: 100%;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }

    .feature-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.3);
        border-color: rgba(102, 126, 234, 0.6);
    }

    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        display: block;
    }

    .feature-title {
        color: #667eea;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    .feature-desc {
        color: rgba(255, 255, 255, 0.7);
        font-size: 0.95rem;
        line-height: 1.6;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1e2f 0%, #2d2d44 100%);
        border-right: 1px solid rgba(102, 126, 234, 0.2);
    }

    section[data-testid="stSidebar"] > div {
        background: transparent;
    }

    .sidebar-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
    }

    .sidebar-header h2 {
        color: white;
        font-size: 1.5rem;
        margin: 0;
        font-weight: 600;
    }

    /* Info Cards */
    .info-card {
        background: rgba(102, 126, 234, 0.1);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }

    .info-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.2);
        border-color: rgba(102, 126, 234, 0.5);
    }

    /* Hero Image Section */
    .hero-section {
        text-align: center;
        margin: 2rem 0;
        padding: 2rem;
        background: rgba(102, 126, 234, 0.05);
        border-radius: 16px;
        border: 2px solid rgba(102, 126, 234, 0.2);
    }

    .hero-section img {
        max-width: 100%;
        border-radius: 12px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }

    .stDownloadButton > button {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        transition: all 0.3s ease;
    }

    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(245, 87, 108, 0.4);
    }

    /* Radio Buttons */
    .stRadio > label {
        color: #667eea;
        font-weight: 600;
        font-size: 1.1rem;
    }

    .stRadio > div {
        background: rgba(102, 126, 234, 0.05);
        padding: 1rem;
        border-radius: 8px;
    }

    /* Select Box */
    .stSelectbox > label {
        color: #667eea;
        font-weight: 600;
        font-size: 1.1rem;
    }

    /* Slider */
    .stSlider > label {
        color: #667eea;
        font-weight: 600;
        font-size: 1.1rem;
    }

    .stSlider > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }

    /* File Uploader */
    .stFileUploader > label {
        color: #667eea;
        font-weight: 600;
        font-size: 1.1rem;
    }

    section[data-testid="stFileUploadDropzone"] {
        background: rgba(102, 126, 234, 0.05);
        border: 2px dashed rgba(102, 126, 234, 0.3);
        border-radius: 12px;
        transition: all 0.3s ease;
    }

    section[data-testid="stFileUploadDropzone"]:hover {
        border-color: rgba(102, 126, 234, 0.6);
        background: rgba(102, 126, 234, 0.1);
    }

    /* Messages */
    .stSuccess, .stInfo, .stWarning, .stError {
        border-radius: 8px;
        padding: 1rem;
        font-weight: 500;
    }

    .stSuccess {
        background: rgba(46, 213, 115, 0.1);
        border-left: 4px solid #2ed573;
    }

    .stInfo {
        background: rgba(102, 126, 234, 0.1);
        border-left: 4px solid #667eea;
    }

    .stWarning {
        background: rgba(255, 159, 64, 0.1);
        border-left: 4px solid #ff9f40;
    }

    /* Image Styling */
    .stImage > div {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
    }

    .stImage > div:hover {
        transform: scale(1.02);
        box-shadow: 0 12px 35px rgba(0,0,0,0.4);
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        margin-top: 3rem;
        background: rgba(102, 126, 234, 0.05);
        border-radius: 12px;
        border-top: 2px solid rgba(102, 126, 234, 0.2);
    }

    .footer a {
        color: #667eea;
        text-decoration: none;
        font-weight: 600;
        transition: all 0.3s ease;
        margin: 0 0.5rem;
    }

    .footer a:hover {
        color: #764ba2;
        text-shadow: 0 0 10px rgba(102, 126, 234, 0.5);
    }

    /* Animations */
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes pulse {
        0%, 100% {
            opacity: 1;
        }
        50% {
            opacity: 0.7;
        }
    }

    /* Checkbox */
    .stCheckbox > label {
        color: #667eea;
        font-weight: 500;
    }

    /* Divider */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.5), transparent);
        margin: 2rem 0;
    }

    /* Responsive */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        .main-header p {
            font-size: 0.95rem;
        }
        .feature-card {
            margin-bottom: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HEADER
# ============================================================================
st.markdown(f"""
<div class="main-header">
    <h1>🔬 {APP_TITLE}</h1>
    <p>{APP_SUBTITLE}</p>
</div>
""", unsafe_allow_html=True)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def convert_image(uploaded_file):
    """Convert uploaded file to numpy array."""
    image = Image.open(uploaded_file)
    return np.array(image.convert("RGB"))


def display_hist(image, title):
    """Display histogram with professional styling."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

    fig, ax = plt.subplots(figsize=(10, 4), facecolor='#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    ax.plot(hist, color='#667eea', linewidth=2)
    ax.fill_between(range(256), hist.flatten(), alpha=0.3, color='#667eea')
    ax.set_title(f"📊 {title} Histogram", color='white', fontsize=14, fontweight='bold')
    ax.set_xlabel('Pixel Intensity', color='white')
    ax.set_ylabel('Frequency', color='white')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('#667eea')
    ax.spines['left'].set_color('#667eea')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2, color='#667eea')

    st.pyplot(fig)
    plt.close()


def side_by_side(*images, titles=[]):
    """Display images side by side."""
    cols = st.columns(len(images))
    for i, img in enumerate(images):
        with cols[i]:
            caption = titles[i] if i < len(titles) else None
            st.image(img, caption=f"🖼️ {caption}" if caption else None, width='stretch')


def thresholding(image, thr=100):
    """Apply binary thresholding."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)


def filtering(image):
    """Apply Gaussian blur and sharpening filters."""
    blurred = cv2.GaussianBlur(image, (7, 7), 1.5)
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return blurred, sharpened


def edge_detection(image):
    """Apply edge detection using Canny and Sobel."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Canny Edge Detection
    canny = cv2.Canny(gray, 100, 200)
    canny_rgb = cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB)

    # Sobel Edge Detection
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobelx ** 2 + sobely ** 2)
    sobel = np.uint8(sobel / sobel.max() * 255)
    sobel_rgb = cv2.cvtColor(sobel, cv2.COLOR_GRAY2RGB)

    return canny_rgb, sobel_rgb


def morphological_operations(image):
    """Apply morphological operations (erosion, dilation, opening, closing)."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)

    erosion = cv2.erode(binary, kernel, iterations=1)
    dilation = cv2.dilate(binary, kernel, iterations=1)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return (cv2.cvtColor(erosion, cv2.COLOR_GRAY2RGB),
            cv2.cvtColor(dilation, cv2.COLOR_GRAY2RGB),
            cv2.cvtColor(opening, cv2.COLOR_GRAY2RGB),
            cv2.cvtColor(closing, cv2.COLOR_GRAY2RGB))


def histogram_equalization(image):
    """Apply histogram equalization."""
    # Convert to YCrCb color space
    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    equalized = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
    return equalized


def face_identification(image, face_cascade):
    """Detect and mark faces in the image."""
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (102, 126, 234), 3)
        cv2.putText(img_bgr, f"{w}x{h}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (102, 126, 234), 2)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb, len(faces)


def emotion_deepface(image, face_cascade):
    """Detect emotions using DeepFace AI."""
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    img_copy = img_bgr.copy()
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    result_list = []

    for (x, y, w, h) in faces:
        face = img_bgr[y:y + h, x:x + w]
        try:
            result = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)
            if isinstance(result, list) and len(result) > 0:
                result = result[0]
            emotion = result.get('dominant_emotion', "Unknown")

            cv2.rectangle(img_copy, (x, y), (x + w, y + h), (46, 213, 115), 3)
            cv2.putText(img_copy, emotion.upper(), (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (46, 213, 115), 2)
            result_list.append((emotion, (x, y, w, h)))
        except Exception as e:
            st.error(f"⚠️ DeepFace Analysis Error: {str(e)}")

    img_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
    return img_rgb, result_list


class EmotionDetector(VideoProcessorBase):
    """Real-time emotion detection for webcam stream."""

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in faces:
            face = img[y:y + h, x:x + w]
            try:
                result = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)
                if isinstance(result, list) and len(result) > 0:
                    result = result[0]
                emotion = result.get('dominant_emotion', "Unknown")

                cv2.rectangle(img, (x, y), (x + w, y + h), (46, 213, 115), 3)
                cv2.putText(img, emotion.upper(), (x, y - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (46, 213, 115), 2)
            except Exception:
                pass

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <h2>⚡ Control Panel</h2>
    </div>
    """, unsafe_allow_html=True)

    mode = st.radio(
        "Select Operation Mode:",
        ["🏠 Home", "📸 Image Processing", "🎥 Live Webcam Detection"],
        help="Choose between home view, static image analysis, or real-time webcam emotion detection"
    )

    st.markdown("---")

    st.markdown("""
    <div class="info-card">
        <h3 style="color: #667eea; margin-top: 0;">ℹ️ About</h3>
        <p style="color: rgba(255,255,255,0.8); font-size: 0.9rem;">
        This application combines advanced Digital Image Processing techniques 
        with state-of-the-art AI models for emotion recognition.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <h3 style="color: #667eea; margin-top: 0;">🛠️ Technologies</h3>
        <p style="color: rgba(255,255,255,0.8); font-size: 0.9rem;">
        • OpenCV for image processing<br>
        • DeepFace for emotion AI<br>
        • Streamlit for web interface<br>
        • Real-time video processing
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# HOME PAGE
# ============================================================================
if mode == "🏠 Home":
    # Hero Section with Image
    st.markdown("""
    <div class="hero-section">
        <h2 style="color: #667eea; margin-bottom: 1rem;">Welcome to DIP Laboratory</h2>
        <p style="color: rgba(255,255,255,0.8); font-size: 1.1rem; max-width: 800px; margin: 0 auto;">
        Explore the power of Digital Image Processing and Artificial Intelligence. 
        Our platform provides cutting-edge tools for image analysis and real-time emotion detection.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Feature Cards
    st.markdown("### 🎯 Key Features")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <span class="feature-icon">🖼️</span>
            <h3 class="feature-title">Image Processing</h3>
            <p class="feature-desc">
                Apply advanced filters, thresholding, edge detection, and morphological 
                operations to enhance and analyze images with precision.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <span class="feature-icon">🤖</span>
            <h3 class="feature-title">AI-Powered Emotions</h3>
            <p class="feature-desc">
                Leverage deep learning models to detect and classify human emotions 
                from facial expressions with high accuracy.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-card">
            <span class="feature-icon">⚡</span>
            <h3 class="feature-title">Real-Time Analysis</h3>
            <p class="feature-desc">
                Process live webcam feeds for instant emotion recognition and 
                face detection with minimal latency.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Available Techniques
    st.markdown("### 🔧 Available Techniques")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3 class="feature-title">📊 Image Processing Methods</h3>
            <p class="feature-desc">
                • <b>Thresholding:</b> Binary image segmentation<br>
                • <b>Filtering:</b> Blur and sharpening effects<br>
                • <b>Edge Detection:</b> Canny and Sobel operators<br>
                • <b>Morphology:</b> Erosion, dilation, opening, closing<br>
                • <b>Histogram Equalization:</b> Contrast enhancement
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3 class="feature-title">🎭 Emotion Detection</h3>
            <p class="feature-desc">
                • <b>Face Detection:</b> Haar Cascade classifier<br>
                • <b>Emotion AI:</b> DeepFace neural network<br>
                • <b>7 Emotions:</b> Happy, Sad, Angry, Surprise, Fear, Disgust, Neutral<br>
                • <b>Real-time:</b> Webcam emotion tracking<br>
                • <b>Batch Processing:</b> Multiple faces support
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Getting Started
    st.markdown("### 🚀 Getting Started")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="feature-card" style="text-align: center;">
            <span class="feature-icon">1️⃣</span>
            <h3 class="feature-title">Select Mode</h3>
            <p class="feature-desc">
                Choose between Image Processing or Live Webcam Detection from the sidebar.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card" style="text-align: center;">
            <span class="feature-icon">2️⃣</span>
            <h3 class="feature-title">Upload/Start</h3>
            <p class="feature-desc">
                Upload an image or start your webcam to begin the analysis process.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-card" style="text-align: center;">
            <span class="feature-icon">3️⃣</span>
            <h3 class="feature-title">Analyze</h3>
            <p class="feature-desc">
                Apply techniques and view results. Download processed images when done.
            </p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# IMAGE PROCESSING MODE
# ============================================================================
elif mode == "📸 Image Processing":
    st.markdown("### 📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a portrait image (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear portrait image for best results"
    )

    if uploaded_file:
        col1, col2 = st.columns([2, 1])

        with col1:
            method = st.selectbox(
                "🔧 Select Processing Method:",
                [
                    'Show Original',
                    'Threshold',
                    'Filtering',
                    'Edge Detection',
                    'Morphological Operations',
                    'Histogram Equalization',
                    'Face Identification',
                    'Emotion Detection (AI)'
                ],
                help="Choose the image processing technique to apply"
            )

        with col2:
            save_results = st.checkbox("💾 Enable Download", help="Show download buttons for results")

        st.markdown("---")

        # Load image and cascade classifier
        image = convert_image(uploaded_file)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        show_list = [image]
        title_list = ["Original Image"]

        # Process based on selected method
        if method == 'Threshold':
            st.markdown("### 🎚️ Threshold Configuration")
            val = st.slider(
                "Threshold Value:",
                0, 255, 127,
                help="Adjust the binary threshold value (0-255)"
            )

            processed = thresholding(image, thr=val)
            show_list.append(processed)
            title_list.append("Thresholded Result")

            st.markdown("### 📊 Histogram Analysis")
            col1, col2 = st.columns(2)
            with col1:
                display_hist(image, "Original")
            with col2:
                display_hist(processed, "Thresholded")

        elif method == 'Filtering':
            blurred, sharp = filtering(image)
            show_list += [blurred, sharp]
            title_list += ["Gaussian Blur", "Sharpened"]

            st.markdown("### 📊 Histogram Analysis")
            display_hist(image, "Original")

        elif method == 'Edge Detection':
            canny, sobel = edge_detection(image)
            show_list += [canny, sobel]
            title_list += ["Canny Edge", "Sobel Edge"]

            st.markdown("### 📊 Histogram Analysis")
            display_hist(image, "Original")

        elif method == 'Morphological Operations':
            erosion, dilation, opening, closing = morphological_operations(image)
            show_list += [erosion, dilation, opening, closing]
            title_list += ["Erosion", "Dilation", "Opening", "Closing"]

            st.info("ℹ️ Morphological operations are used for image preprocessing and feature extraction.")

        elif method == 'Histogram Equalization':
            equalized = histogram_equalization(image)
            show_list.append(equalized)
            title_list.append("Equalized")

            st.markdown("### 📊 Histogram Comparison")
            col1, col2 = st.columns(2)
            with col1:
                display_hist(image, "Original")
            with col2:
                display_hist(equalized, "Equalized")

        elif method == 'Face Identification':
            identified, nfaces = face_identification(image, face_cascade)
            show_list.append(identified)
            title_list.append(f"Detected: {nfaces} Face(s)")

            if nfaces > 0:
                st.success(f"✅ Successfully detected {nfaces} face(s) in the image!")
            else:
                st.warning("⚠️ No faces detected. Try another image.")

        elif method == 'Emotion Detection (AI)':
            with st.spinner('🤖 AI is analyzing emotions...'):
                ai_emotion_img, results = emotion_deepface(image, face_cascade)
                show_list.append(ai_emotion_img)
                title_list.append("AI Emotion Analysis")

                if results:
                    st.success(f"✅ Detected emotions for {len(results)} face(s):")

                    cols = st.columns(min(len(results), 4))
                    for idx, (emotion, box) in enumerate(results):
                        with cols[idx % 4]:
                            emoji_map = {
                                'happy': '😊',
                                'sad': '😢',
                                'angry': '😠',
                                'surprise': '😮',
                                'fear': '😨',
                                'disgust': '🤢',
                                'neutral': '😐'
                            }
                            emoji = emoji_map.get(emotion.lower(), '😐')
                            st.info(f"{emoji} **{emotion.upper()}**")
                else:
                    st.warning("⚠️ No faces detected for emotion analysis.")

        # Display results
        st.markdown("### 🎨 Results")
        side_by_side(*show_list, titles=title_list)

        # Download section
        if save_results and len(show_list) > 1:
            st.markdown("### 💾 Download Results")
            cols = st.columns(len(show_list))
            for i, img in enumerate(show_list):
                with cols[i]:
                    filename = f"{method.replace(' ', '_')}_{title_list[i].replace(' ', '_')}.png"
                    pil_img = Image.fromarray(img)
                    buf = io.BytesIO()
                    pil_img.save(buf, format="PNG")
                    st.download_button(
                        f"📥 {title_list[i]}",
                        buf.getvalue(),
                        file_name=filename,
                        mime="image/png",
                        use_container_width=True
                    )
    else:
        st.info("👆 Please upload an image to begin processing")

# ============================================================================
# LIVE WEBCAM MODE
# ============================================================================
elif mode == "🎥 Live Webcam Detection":
    st.markdown("### 🎥 Real-Time Emotion Detection")
    st.warning("⚠️ Allow camera access when prompted. Processing may be slower on older devices.")

    with st.expander("ℹ️ Privacy Notice", expanded=False):
        st.markdown("""
        - All processing happens **locally** on your device
        - No video data is stored or transmitted
        - Camera access is only used for real-time analysis
        - You can stop the stream at any time
        """)

    st.markdown("---")

    webrtc_streamer(
        key="emotion-live-detection",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=EmotionDetector,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown(f"""
<div class="footer">
    <p style="font-size: 1.1rem; color: rgba(255,255,255,0.9); font-weight: 600;">
        Made with ❤️ by {APP_AUTHOR}
    </p>
    <p style="color: rgba(255,255,255,0.7);">
        <a href="{BRAND_URL}" target="_blank">🔗 GitHub</a> •
        <a href="mailto:{EMAIL}">📧 Email</a>
    </p>
    <p style="font-size: 0.85rem; color: rgba(255,255,255,0.5); margin-top: 1rem;">
        Digital Image Processing & AI Emotion Recognition Laboratory
    </p>
</div>
""", unsafe_allow_html=True)