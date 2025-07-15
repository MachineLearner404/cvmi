import streamlit as st
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
from ultralytics import YOLO
import os

# ----------  HARD-CODED LOGIN USERS ----------
USERS = {
    "daksh":  {"password": "cvmi123",    "name": "Daksh Vyawhare", "role": "Deep Learning Engineer", "image": "assets/daksh.jpg"},
    "khushi": {"password": "research456","name": "Dr. Khushi Rathod", "role": "Researcher", "image": "assets/khushi.jpg"},
    "alka":   {"password": "mentor1",    "name": "Dr. Alka Banker", "role": "Mentor", "image": "assets/alka.jpg"},
    "rahul":  {"password": "mentor2",    "name": "Dr. Rahul Kumar", "role": "Mentor", "image": "assets/rahul.jpg"},
}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username  = ""

def login_page():
    st.title("CVMI Vertebra Tool – Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in USERS and USERS[username]["password"] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.rerun()
        else:
            st.error("Invalid username or password.")

if not st.session_state.logged_in:
    login_page()
    st.stop()

# ---------- Load Models ----------
roi_model = YOLO("roi_best.pt")
seg_model = YOLO("seg_best.pt")

# ---------- Color Map ----------
class_color_map = {
    "C2": [139, 0, 0],
    "C3": [0, 100, 0],
    "C4": [0, 0, 139],
}

# ---------- Session State Defaults ----------
for key in ["original_image", "cropped_roi", "segmented_output", "masks"]:
    st.session_state.setdefault(key, None)

# ---------- Sidebar ----------
st.sidebar.markdown(f"**Welcome, {USERS[st.session_state.username]['name']}!**")
page = st.sidebar.radio("Navigation", ["Main Page", "Combined Masks Page", "Dashboard"])

# Add logout button
with st.sidebar:
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()

# ---------- Main Page ----------
def main_page():
    st.title("CVMI Vertebra Segmentation")
    st.write("Upload a cephalometric X-ray image to detect and segment vertebrae (C2–C4).")

    if st.button("Clear session"):
        for key in ["original_image", "cropped_roi", "segmented_output", "masks"]:
            st.session_state[key] = None
        st.rerun()

    uploader_locked = st.session_state.original_image is not None
    if uploader_locked:
        st.info("Uploader is locked. Click 'Clear session' to upload a new image.")

    uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"], disabled=uploader_locked)

    if uploaded_file and not uploader_locked:
        image = np.array(Image.open(uploaded_file).convert("RGB"))
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        st.session_state.original_image = image_bgr

        roi_results = roi_model(image_bgr)
        boxes = roi_results[0].boxes.xyxy.cpu().numpy()

        if len(boxes) == 0:
            st.warning("No ROI detected.")
            return

        x1, y1, x2, y2 = map(int, boxes[0])
        cropped_roi = image_bgr[y1:y2, x1:x2]
        st.session_state.cropped_roi = cropped_roi

        seg_results = seg_model.predict(source=cropped_roi, imgsz=512, save=False)
        masks = seg_results[0].masks

        if masks is None or len(masks.data) == 0:
            st.error("No vertebrae detected.")
            return

        classes = seg_results[0].boxes.cls.cpu().numpy()
        seg_boxes = seg_results[0].boxes.xyxy.cpu().numpy()
        confs = seg_results[0].boxes.conf.cpu().numpy()
        names = seg_model.names
        base_img = cropped_roi.copy()
        overlay = np.zeros_like(base_img, dtype=np.uint8)
        mask_data = masks.data.cpu().numpy()
        target_shape = (base_img.shape[1], base_img.shape[0])

        vertebra_masks = []

        for i, mask in enumerate(mask_data):
            class_id = int(classes[i])
            class_name = names[class_id]
            conf = float(confs[i])
            box = seg_boxes[i]
            color = class_color_map.get(class_name, [50, 50, 50])
            resized_mask = cv2.resize(mask, target_shape, interpolation=cv2.INTER_NEAREST)
            binary_mask = (resized_mask > 0.5).astype(np.uint8)

            colored_mask = np.zeros_like(overlay)
            for c in range(3):
                colored_mask[:, :, c] = binary_mask * color[c]

            overlay = cv2.addWeighted(overlay, 1.0, colored_mask, 0.5, 0)

            x1b, y1b = int(box[0]), int(box[1])
            text_position = (x1b + 40, y1b)
            cv2.putText(base_img, f"{class_name} ({conf:.2f})", text_position, cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2, cv2.LINE_AA)

            vertebra_masks.append({
                "class_name": class_name,
                "conf": conf,
                "box": list(map(int, box)),
                "mask": binary_mask
            })

        final_img = cv2.addWeighted(base_img, 1.0, overlay, 0.5, 0)
        final_img_rgb = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)

        st.session_state.segmented_output = final_img_rgb
        st.session_state.masks = vertebra_masks

    if st.session_state.original_image is not None and st.session_state.cropped_roi is not None and st.session_state.segmented_output is not None:
        st.subheader("Uploaded Image | ROI | Segmented Output")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(cv2.cvtColor(st.session_state.original_image, cv2.COLOR_BGR2RGB), caption="Original X-ray", use_container_width=True)
        with col2:
            st.image(cv2.cvtColor(st.session_state.cropped_roi, cv2.COLOR_BGR2RGB), caption="Detected ROI", use_container_width=True)
        with col3:
            st.image(st.session_state.segmented_output, caption="Segmented Output", use_container_width=True)

# ---------- Combined Mask Page ----------
def combined_mask_page():
    st.title("Combined Vertebrae Mask View")
    if st.button("Clear session"):
        for key in ["original_image", "cropped_roi", "segmented_output", "masks"]:
            st.session_state[key] = None
        st.rerun()

    if st.session_state.masks is None:
        st.warning("No masks available. Use the main page to segment.")
        return

    vertebra_order = {"C2": 0, "C3": 1, "C4": 2}
    sorted_masks = sorted(st.session_state.masks, key=lambda x: vertebra_order.get(x["class_name"], 999))

    panels = []
    for entry in sorted_masks:
        class_name = entry["class_name"]
        conf = entry["conf"]
        x1b, y1b, x2b, y2b = entry["box"]
        binary_mask = entry["mask"]
        vertebra_mask = binary_mask[y1b:y2b, x1b:x2b]
        h, w = vertebra_mask.shape
        white_bg = np.ones((h, w, 3), dtype=np.uint8) * 255

        color = class_color_map.get(class_name, [128, 128, 128])
        for c in range(3):
            white_bg[:, :, c] = np.where(vertebra_mask == 1, color[c], 255)

        contours, _ = cv2.findContours(vertebra_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(white_bg, contours, -1, (0, 0, 0), thickness=2)

        cv2.putText(white_bg, f"{class_name} ({conf:.2f})", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 0), 2, cv2.LINE_AA)

        panels.append(white_bg)

    if panels:
        max_width = max(p.shape[1] for p in panels)
        padded_panels = []
        for p in panels:
            h, w = p.shape[:2]
            if w < max_width:
                pad = max_width - w
                pad_left = pad // 2
                pad_right = pad - pad_left
                p = cv2.copyMakeBorder(p, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            padded_panels.append(p)

        combined = np.vstack(padded_panels)
        combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)

        chart_path = "assets/cvmi_chart.jpg"
        if os.path.exists(chart_path):
            chart_img = Image.open(chart_path)
            chart_array = np.array(chart_img.convert("RGB"))
            ch = combined_rgb.shape[0]
            cw = int(chart_array.shape[1] * (ch / chart_array.shape[0]))
            chart_resized = cv2.resize(chart_array, (cw, ch))

            delta_w = abs(combined_rgb.shape[1] - chart_resized.shape[1])
            if chart_resized.shape[1] < combined_rgb.shape[1]:
                pad_left = delta_w // 2
                pad_right = delta_w - pad_left
                chart_resized = cv2.copyMakeBorder(chart_resized, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            elif chart_resized.shape[1] > combined_rgb.shape[1]:
                combined_rgb = cv2.copyMakeBorder(combined_rgb, 0, 0, delta_w // 2, delta_w - delta_w // 2, cv2.BORDER_CONSTANT, value=(255, 255, 255))

            side_by_side = np.hstack([chart_resized, combined_rgb])
            st.image(side_by_side, caption="CVMI Chart and Combined Vertebrae Mask", use_container_width=True)
        else:
            st.image(combined_rgb, caption="Combined Vertebrae Mask (CVMI Chart Missing)", use_container_width=True)
    else:
        st.warning("No panels to display.")

# ---------- Dashboard Page ----------
def dashboard_page():
    st.title("Dashboard")
    st.header("What We Do")
    st.write("This tool detects, segments, and presents CVMI stages from lateral cephalograms for clinical and academic applications.")

    st.header("About Us")
    cols = st.columns(4)
    for i, user_key in enumerate(USERS):
        user = USERS[user_key]
        with cols[i % 4]:
            if os.path.exists(user["image"]):
                st.image(user["image"], width=120)
            st.markdown(f"**{user['name']}**")
            st.caption(user['role'])

    st.markdown("---")
    st.caption("Inspired by Doctors. Designed by Engineers.")

# ---------- Page Routing ----------
if page == "Main Page":
    main_page()
elif page == "Combined Masks Page":
    combined_mask_page()
elif page == "Dashboard":
    dashboard_page()