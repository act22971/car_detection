import os
import cv2
import streamlit as st
from ultralytics import YOLO
import tempfile
import time

# Load the YOLOv8 model
model = YOLO('yolo8n.pt')

def video_process_page():

    # Streamlit app title
    st.title("Car Crash Detection and Tracking")

    # Video file uploader
    video_file = st.file_uploader("Upload a Video", type=['mp4', 'mov', 'avi', 'mkv'])

    # Initialize a session state variable for the temporary video file path
    if "temp_video_path" not in st.session_state:
        st.session_state.temp_video_path = None

    if video_file is not None:
        # Save uploaded video to a unique temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            st.session_state.temp_video_path = temp_file.name
            temp_file.write(video_file.read())

        # Initialize counts and display placeholders
        car_count = 0
        crossed_car_count = 0

        crossed_car_count_text = st.empty()
        crossed_car_count_text.markdown(f"**Cars crossed the line:** {crossed_car_count}")

        # Define a vertical line position for detecting crossings
        line_position = 240

        # Load video and initialize placeholders
        cap = cv2.VideoCapture(st.session_state.temp_video_path)
        stframe = st.empty()
        frame_counter = 0
        frame_skip = 1  # Process every nth frame
        tracked_cars = {}
        crossed_car_ids = set()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_counter % frame_skip == 0:
                    frame = cv2.resize(frame, (640, 480))
                    result = model.track(frame, persist=True)
                    frame_result = result[0].plot()

                    # Check if detections exist
                    detections = result[0].boxes
                    if detections is None or len(detections) == 0:
                        continue  # No detections, skip this frame

                    labels = detections.cls
                    confidences = detections.conf
                    bboxes = detections.xyxy
                    ids = detections.id

                    car_count = 0  # Reset car count for this frame

                    for i in range(len(labels)):
                        if labels[i] == 1 and confidences[i] > 0.35:  # Only for car detections
                            car_count += 1

                            # Handle bounding boxes (bboxes) safely
                            bbox = bboxes[i]
                            if bbox.is_cuda:  # Move to CPU if needed
                                x1, y1, x2, y2 = bbox.cpu().numpy()
                            else:
                                x1, y1, x2, y2 = bbox.numpy()

                            # Ensure that 'ids' is valid and has values
                            if ids is not None and len(ids) > 0:
                                car_id_tensor = ids[i]
                                if car_id_tensor.is_cuda:  # Move to CPU if needed
                                    car_id = int(car_id_tensor.cpu().item())
                                else:
                                    car_id = int(car_id_tensor.item())

                                if car_id not in crossed_car_ids:
                                    if car_id not in tracked_cars:
                                        tracked_cars[car_id] = y1
                                    else:
                                        previous_y = tracked_cars[car_id]
                                        if (previous_y < line_position <= y1) or (previous_y > line_position >= y1):
                                            crossed_car_count += 1
                                            crossed_car_ids.add(car_id)
                                    tracked_cars[car_id] = y1

                    # Clean up tracked cars
                    if ids is not None:
                        ids_array = ids.cpu().numpy() if ids.is_cuda else ids.numpy()
                        tracked_cars = {car_id: pos for car_id, pos in tracked_cars.items() if car_id in ids_array}
                    else:
                        tracked_cars = {}

                    # Convert frame from BGR to RGB
                    frame_result = cv2.cvtColor(frame_result, cv2.COLOR_BGR2RGB)

                    # Update Streamlit display
                    stframe.image(frame_result, channels='RGB', use_column_width=True)
                    crossed_car_count_text.markdown(f"**Cars drive through:** {crossed_car_count}")

                    time.sleep(0.015)  # Control frame rate

                frame_counter += 1

        finally:
            cap.release()  # Release video capture object
            # Delete the temporary file
            if st.session_state.temp_video_path and os.path.exists(st.session_state.temp_video_path):
                os.remove(st.session_state.temp_video_path)
                st.session_state.temp_video_path = None