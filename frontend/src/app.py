"""
Creative Studio Frontend - Streamlit UI
"""
import streamlit as st
import requests
import os
from PIL import Image, ImageDraw
import io
import json

# Page configuration
st.set_page_config(
    page_title="Creative Studio",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling
st.markdown("""
    <style>
        .main { padding: 2rem; }
        h1 { color: #6F4FF2; text-align: center; }
        .result-box { 
            border: 2px solid #6F4FF2;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
    </style>
""", unsafe_allow_html=True)

# API endpoint
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Initialize session state
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None
if "file_id" not in st.session_state:
    st.session_state.file_id = None
if "detections" not in st.session_state:
    st.session_state.detections = None
if "selected_objects" not in st.session_state:
    st.session_state.selected_objects = []

# Title
st.title("🎨 Creative Studio")
st.markdown("AI-powered photo editor with selective neural style transfer")

st.markdown("Choose an image, detect objects within the image, then apply artistic styles to selected objects!")

# Check API health
try:
    health = requests.get(f"{API_URL}/health", timeout=5).json()
    if health["status"] == "healthy":
        st.success("✅ Connected to Creative Studio API")
    else:
        st.error("❌ API not ready")
except:
    st.error("❌ Cannot connect to API. Make sure backend is running.")
    st.stop()
    
# File uploader
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"]
)
    
# Main layout
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📤 Upload & Detect")
    
    image_placeholder = st.empty()
    
    if uploaded_file:
        # Display uploaded image
        image = Image.open(uploaded_file)
        image_placeholder.image(image, caption="Uploaded Image")
        
        # Upload to backend
        if st.button("🔍 Upload & Detect Objects", use_container_width=True):
            with st.spinner("Uploading image..."):
                try:
                    files = {"file": uploaded_file.getbuffer()}
                    response = requests.post(
                        f"{API_URL}/upload",
                        files=files,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.session_state.file_id = result["file_id"]
                        st.session_state.uploaded_image = image
                        
                        st.success("✅ Image uploaded successfully")
                        
                        # Run detection
                        with st.spinner("Detecting objects..."):
                            det_response = requests.post(
                                f"{API_URL}/detect-objects",
                                params={"file_id": st.session_state.file_id},
                                timeout=60
                            )
                            
                            if det_response.status_code == 200:
                                st.session_state.detections = det_response.json()["objects"]
                                st.success(f"✅ Found {len(st.session_state.detections)} objects")
                            else:
                                st.error(f"Detection failed: {det_response.text}")
                    else:
                        st.error(f"Upload failed: {response.text}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    
        st.divider()

    NST_image = st.empty()
    NST_download = st.empty()

with col2:
    st.subheader("🎭 Select & Stylize")
    
    if st.session_state.detections:
        st.markdown("Select objects below to visualize and apply styles")        
        st.markdown("**Detected Objects:**")
        
        # Display detections
        for i, det in enumerate(st.session_state.detections):
            col_check, col_info = st.columns([0.3, 0.7], vertical_alignment="top")
            
            with col_check:
                selected = st.checkbox(
                    f"Select #{i}",
                    key=f"det_{i}"
                )
                if selected and i not in st.session_state.selected_objects:
                    st.session_state.selected_objects.append(i)
                elif not selected and i in st.session_state.selected_objects:
                    st.session_state.selected_objects.remove(i)
            
            with col_info:
                st.markdown(
                    f"**{det['class']}** (conf: {det['confidence']:.2f}) | "
                    f"Size: {det['bbox']['width']:.0f}x{det['bbox']['height']:.0f}px"
                )
        
        if st.session_state.selected_objects:
            # Draw bounding boxes on a copy of the original image
            # from PIL import ImageDraw
            
            viz_image = st.session_state.uploaded_image.copy()
            draw = ImageDraw.Draw(viz_image)
            
            # Color map for different objects
            colors = [
                "#FF0000", "#00FF00", "#0000FF",
                "#FFFF00", "#FF00FF", "#00FFFF",
                "#FF8000", "#FF0080", "#80FF00",
                "#8000FF", "#00FF80", "#FF8080"
            ]
            
            # Draw boxes for selected objects
            for idx in st.session_state.selected_objects:
                if idx >= len(st.session_state.detections):
                    continue
                
                detection = st.session_state.detections[idx]
                color = colors[idx % len(colors)]
                bbox = detection["bbox"]
                
                # Draw rectangle
                x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
                draw.rectangle(
                    [(x1, y1), (x2, y2)],
                    outline=color,
                    width=3
                )
                
                # Draw label
                label = f"{detection['class']} ({idx})"
                draw.text(
                    (int(x1), max(int(y1) - 20, 10)),
                    label,
                    fill=color
                )
            
            image_placeholder.image(
                viz_image,
                caption=f"Selected Objects ({len(st.session_state.selected_objects)})"
            )
        else:
            st.info("Select objects above to visualize")
        
        st.divider()
        
        # Style selection
        st.markdown("**Choose Style:**")
        
        try:
            styles_response = requests.get(f"{API_URL}/styles", timeout=10)
            if styles_response.status_code == 200:
                available_styles = styles_response.json()["styles"]
            else:
                available_styles = ["mosaic", "impressionist", "cubist", "oil_painting"]
        except:
            available_styles = ["mosaic", "impressionist", "cubist", "oil_painting"]
        
        style_name = st.selectbox(
            "Style",
            available_styles,
            index=0
        )
        
        strength = st.slider(
            "Style Strength",
            0.0, 1.0, 0.8, 0.1
        )
        
        # Apply style transfer
        if st.button("✨ Apply Style Transfer", use_container_width=True, type="primary"):
            if not st.session_state.selected_objects:
                st.warning("Please select at least one object")
            else:
                with st.spinner("Applying neural style transfer..."):
                    try:
                        response = requests.post(
                            f"{API_URL}/apply-style-transfer",
                            params={
                                "file_id": st.session_state.file_id,
                                "style_name": style_name,
                                "object_indices": st.session_state.selected_objects,
                                "strength": strength
                            },
                            timeout=120
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.success(f"✅ Style transfer completed!")
                            st.info(
                                f"Applied **{style_name}** to "
                                f"{result['objects_stylized']} object(s)"
                            )
                            
                            # Download result
                            result_id = result["result_id"]
                            
                            try:
                                img_response = requests.get(
                                    f"{API_URL}/result/{result_id}",
                                    timeout=30
                                )
                                
                                if img_response.status_code == 200:
                                    result_image = Image.open(io.BytesIO(img_response.content))
                                    NST_image.image(
                                        result_image,
                                        caption="Stylized Result"
                                    )
                                    
                                    # Download button
                                    NST_download.download_button(
                                        label="📥 Download Result",
                                        data=img_response.content,
                                        file_name=f"result_{style_name}.png",
                                        mime="image/png",
                                        use_container_width=True
                                    )
                            except Exception as e:
                                st.error(f"Error downloading result: {str(e)}")
                        else:
                            st.error(f"Style transfer failed: {response.text}")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

# Settings sidebar
with st.sidebar:
    with st.expander("📊 Session State"):
        st.json(st.session_state)
    
    st.markdown("### 📚 About")
    st.markdown("""
    **Creative Studio** is an AI-powered photo editor that uses:
    - **YOLOv8m-seg** for object detection and image segmentation
    - **Neural Style Transfer** for artistic effects
    - **Google Cloud Storage** for reliable storage
    
    Selective style transfer lets you apply styles to specific objects in your images.
    """)
    
    st.divider()
    
    st.markdown("### ⚙️ Settings")
    
    st.markdown("**API Configuration**")
    st.markdown(f"Endpoint: {API_URL}") #, st.code(f"Endpoint: {API_URL}", language="text")
    
    # Get API settings
    try:
        settings = requests.get(f"{API_URL}/settings", timeout=10).json()
        
        st.markdown("**Capabilities**")
        st.markdown(f"- GPU Enabled: {'✅' if settings['gpu_enabled'] else '❌'}")
        st.markdown(f"- GCS Enabled: {'✅' if settings['gcs_enabled'] else '❌'}")
        st.markdown(f"- YOLO Model: {settings['yolo_model']}")
        st.markdown(f"- Max Image Size: {settings['max_image_size']}MB")
    except:
        st.warning("Could not load API settings")
    
