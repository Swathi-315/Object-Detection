
import streamlit as st
import cv2
import numpy as np
import tempfile
import torch
import torchvision
from torchvision import transforms
from ultralytics import YOLO
from collections import Counter
import time
import os

# ---------------------------
# CSS Styling
# ---------------------------

def load_css(file_name):
    """Loads a CSS file and injects it into the Streamlit app."""
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ---------------------------
# Model & Data Loading
# ---------------------------

def get_torchvision_labels():
    """Returns the list of 80-class COCO labels for torchvision models."""
    return [
        "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
        "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
        "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
        "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
        "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
        "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
        "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
        "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
        "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ]

@st.cache_resource
def load_model(model_name, yolo_size="n", custom_path=None):
    """Loads a detection model with caching to avoid reloading."""
    try:
        if model_name == "YOLO":
            if custom_path and os.path.exists(custom_path):
                return YOLO(custom_path)
            return YOLO(f"yolov8{yolo_size}.pt")
        elif model_name == "SSD":
            model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights="DEFAULT")
            model.eval()
            return model
        elif model_name == "Faster R-CNN":
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
            model.eval()
            return model
    except Exception as e:
        st.error(f"❌ Failed to load {model_name}: {e}")
        return None

# ---------------------------
# Object Detection & Tracking Logic
# ---------------------------

def process_frame(frame, model, model_name, conf_thresh, use_tracking):
    """
    Processes a single frame for object detection or tracking.
    Returns the annotated frame and lists of detected objects and their details.
    """
    output_frame = frame.copy()

    if model_name == "YOLO" and use_tracking:
        results = model.track(output_frame, verbose=False, conf=conf_thresh, persist=True)
        tracked_items = []
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            class_names = results[0].names

            for box, track_id, cls_id in zip(boxes, track_ids, cls_ids):
                x1, y1, x2, y2 = box
                label = class_names[cls_id]
                tracked_items.append((label, track_id))

                track_info = f"ID {track_id}: {label}"
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(output_frame, track_info, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        current_objects = [item[0] for item in tracked_items]
        return output_frame, current_objects, [], tracked_items

    else:
        objects, scores = [], []
        if model_name == "YOLO":
            results = model(output_frame, verbose=False, conf=conf_thresh)
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    cls_id, score = int(box.cls[0]), float(box.conf[0])
                    label = r.names[cls_id]
                    objects.append(label)
                    scores.append(score)
                    cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(output_frame, f"{label} {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else: # Torchvision models
            labels_list = get_torchvision_labels()
            img_rgb = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
            tensor_img = transforms.ToTensor()(img_rgb)
            with torch.no_grad():
                preds = model([tensor_img])[0]
            for i, score_val in enumerate(preds["scores"]):
                if score_val > conf_thresh:
                    box = preds["boxes"][i].cpu().numpy().astype(int)
                    label = labels_list[preds["labels"][i]]
                    objects.append(label)
                    scores.append(float(score_val))
                    cv2.rectangle(output_frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                    cv2.putText(output_frame, f"{label} {score_val:.2f}", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return output_frame, objects, scores, []

# ---------------------------
# UI Rendering Functions
# ---------------------------

def display_sidebar():
    """Configures and displays the sidebar settings, returning user selections."""
    st.sidebar.header("⚙️ Settings")
    model_name = st.sidebar.selectbox("Choose Model", ["YOLO", "SSD", "Faster R-CNN"])
    conf_thresh = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
    frame_skip = st.sidebar.slider("Frame Skip (Video & Webcam)", 1, 10, 1)

    custom_model_path, use_tracking, yolo_size = None, False, "n"
    if model_name == "YOLO":
        st.sidebar.markdown("---")
        use_tracking = st.sidebar.checkbox("Enable Object Tracking", value=True)
        st.sidebar.info("Tracking assigns a unique ID to each object, counting it only once.")

        use_custom = st.sidebar.checkbox("Use Custom YOLO Model?")
        if use_custom:
            custom_model_file = st.sidebar.file_uploader("Upload YOLO Weights (.pt)", type=["pt"])
            if custom_model_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as temp_pt:
                    temp_pt.write(custom_model_file.read())
                    custom_model_path = temp_pt.name
        else:
            yolo_size = st.sidebar.selectbox("YOLO Model Size", ["n (Nano)", "s (Small)", "m (Medium)", "l (Large)", "x (Extra Large)"])[0]

    st.sidebar.markdown("---")
    st.sidebar.info("💡 Tip: Use Frame Skip to improve performance on slower devices.")

    return model_name, conf_thresh, frame_skip, yolo_size, custom_model_path, use_tracking

def render_image_tab(model, model_name, conf_thresh):
    """Handles the logic and UI for the Image Detection tab."""
    st.header("Object Detection for Images")
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"], key="image_uploader")
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        with st.spinner("🔍 Detecting objects..."):
            output_frame, objects, scores, _ = process_frame(img, model, model_name, conf_thresh, use_tracking=False)

        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(output_frame, channels="BGR", caption="✅ Detection Result")
        with col2:
            if objects:
                st.subheader("📊 Detected Objects")
                counts = Counter(objects)
                for name, count in counts.items():
                    st.write(f"**{name.title()}:** {count}")
                if scores:
                    avg_confidence = np.mean(scores)
                    st.subheader("📈 Performance")
                    st.metric(label="Average Confidence", value=f"{avg_confidence:.2f}")
            else:
                st.warning("No objects detected.")
        _, buf = cv2.imencode(".jpg", output_frame)
        st.download_button("💾 Save Result", buf.tobytes(), file_name="detection.jpg", mime="image/jpeg")

def render_video_tab(model, model_name, conf_thresh, frame_skip, use_tracking):
    """Handles the logic and UI for the Video Detection tab."""
    st.header("Object Detection for Videos")
    uploaded_file = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"], key="video_uploader")
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)

        out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (int(cap.get(3)), int(cap.get(4))))

        stframe = st.empty()
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress_bar, status_text = st.progress(0), st.empty()
        total_counts, counted_track_ids = Counter(), set()
        prev_time, frame_count = time.time(), 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            if frame_count % frame_skip == 0:
                output_frame, current_objs, _, tracked_items = process_frame(frame, model, model_name, conf_thresh, use_tracking)

                if use_tracking and model_name == "YOLO":
                    for label, track_id in tracked_items:
                        if track_id not in counted_track_ids:
                            total_counts[label] += 1
                            counted_track_ids.add(track_id)
                else:
                    total_counts.update(current_objs)

                curr_time = time.time()
                fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
                prev_time = curr_time
                cv2.putText(output_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                progress_bar.progress(min(frame_count / total_frames, 1.0))
                status_text.text(f"Processing frame {frame_count}/{total_frames}...")
                stframe.image(output_frame, channels="BGR")
                out.write(output_frame)

            frame_count += 1

        cap.release()
        out.release()
        progress_bar.empty()
        status_text.empty()
        st.success("✅ Video Processing Complete!")

        col1, col2 = st.columns([2, 1])
        with col1:
            st.video(out_path)
            with open(out_path, "rb") as f:
                st.download_button("💾 Save Processed Video", f.read(), file_name="video_output.mp4")
        with col2:
            st.subheader("📊 Total Objects Detected")
            if total_counts:
                for name, count in sorted(total_counts.items()):
                    st.write(f"**{name.title()}:** {count}")
            else: st.warning("No objects were detected.")

def render_webcam_tab(model, model_name, conf_thresh, frame_skip, use_tracking):
    """Handles the logic and UI for the Webcam Detection tab."""
    st.header("🔴 Live Webcam Detection")

    if 'webcam_running' not in st.session_state:
        st.session_state.webcam_running = False

    if not st.session_state.webcam_running:
        if st.button("Start Webcam"):
            st.session_state.webcam_running = True
            st.session_state.total_object_counts = Counter()
            st.session_state.counted_track_ids = set()
            st.rerun()
    else:
        if st.button("Stop Webcam"):
            st.session_state.webcam_running = False
            st.rerun()

        col1, col2 = st.columns([3, 1])
        with col1: FRAME_WINDOW = st.image([])
        with col2:
            st.subheader("📊 Total Detected Objects")
            total_counts_placeholder = st.empty()

        cap = cv2.VideoCapture(0)
        prev_time, frame_count = time.time(), 0

        while st.session_state.webcam_running:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture image from webcam.")
                st.session_state.webcam_running = False
                break

            if frame_count % frame_skip == 0:
                output_frame, current_objs, _, tracked_items = process_frame(frame, model, model_name, conf_thresh, use_tracking)

                if use_tracking and model_name == "YOLO":
                    for label, track_id in tracked_items:
                        if track_id not in st.session_state.counted_track_ids:
                            st.session_state.total_object_counts[label] += 1
                            st.session_state.counted_track_ids.add(track_id)
                else:
                    st.session_state.total_object_counts.update(current_objs)

                curr_time = time.time()
                fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
                prev_time = curr_time
                cv2.putText(output_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                y_offset = 60
                for obj, count in sorted(Counter(current_objs).items()):
                    text = f"{obj.title()}: {count}"
                    cv2.putText(output_frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    y_offset += 30

                with total_counts_placeholder.container():
                    for name, count in sorted(st.session_state.total_object_counts.items()):
                        st.write(f"**{name.title()}:** {count}")

                FRAME_WINDOW.image(output_frame, channels="BGR")

            frame_count += 1

        cap.release()
        if not st.session_state.webcam_running:
             st.info("Webcam stopped.")

# ---------------------------
# Main Application
# ---------------------------

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="AI Object Detection", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")

    # Load external CSS file
    if os.path.exists("style.css"):
        load_css("style.css")

    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🚀 Object Detection Dashboard</h1>", unsafe_allow_html=True)

    model_name, conf_thresh, frame_skip, yolo_size, custom_model_path, use_tracking = display_sidebar()

    with st.spinner(f"🔄 Loading {model_name} model..."):
        model = load_model(model_name, yolo_size, custom_model_path)
    if model is None:
        st.stop()

    tab1, tab2, tab3 = st.tabs(["📷 Image", "🎞️ Video", "📹 Webcam"])

    with tab1:
        render_image_tab(model, model_name, conf_thresh)

    with tab2:
        render_video_tab(model, model_name, conf_thresh, frame_skip, use_tracking)

    with tab3:
        render_webcam_tab(model, model_name, conf_thresh, frame_skip, use_tracking)


if __name__ == "__main__":
    main()