```python
import cv2
import numpy as np
import streamlit as st
import tempfile
from pathlib import Path
from ultralytics import YOLO

try:
    import imageio_ffmpeg as ffmpeg
except ImportError:
    ffmpeg = None


MODEL_PRIMARY = "yolo26n-pose.pt"
MODEL_BACKUP = "yolo11n-pose.pt"

PERSON_CONF = 0.25
JOINT_CONF = 0.5

OUTPUT_FILE = Path(__file__).parent / "pose_result.mp4"

VIDEO_FORMATS = [
    "mp4", "mov", "avi", "mkv", "webm",
    "m4v", "mpg", "mpeg", "wmv"
]

WHITE = (255, 255, 255)

BODY_LINKS = [
    (15, 13), (13, 11), (16, 14), (14, 12),
    (11, 12), (5, 11), (6, 12), (5, 6),
    (5, 7), (6, 8), (7, 9), (8, 10),
    (1, 2), (0, 1), (0, 2), (1, 3),
    (2, 4), (3, 5), (4, 6)
]


@st.cache_resource(show_spinner="Loading pose detection model...")
def initialize_detector():

    try:
        model = YOLO(MODEL_PRIMARY)
        return model, MODEL_PRIMARY, None

    except Exception as error:

        model = YOLO(MODEL_BACKUP)

        warning = (
            f"{MODEL_PRIMARY} could not be loaded. "
            f"Using {MODEL_BACKUP} instead. Details: {error}"
        )

        return model, MODEL_BACKUP, warning


def render_skeleton(result, width, height, thickness, radius):

    frame = np.zeros((height, width, 3), dtype=np.uint8)

    keypoints = result.keypoints

    if keypoints is None or len(keypoints) == 0:
        return frame

    coordinates = keypoints.xy.cpu().numpy()

    if keypoints.conf is None:
        confidence = np.ones(coordinates.shape[:2])
    else:
        confidence = keypoints.conf.cpu().numpy()

    for person, scores in zip(coordinates, confidence):

        visible = (
            (scores >= JOINT_CONF) &
            (person[:, 0] > 0) &
            (person[:, 1] > 0)
        )

        joints = [
            (int(round(x)), int(round(y)))
            for x, y in person
        ]

        for first, second in BODY_LINKS:

            if visible[first] and visible[second]:

                cv2.line(
                    frame,
                    joints[first],
                    joints[second],
                    WHITE,
                    thickness,
                    cv2.LINE_AA
                )

        for index in np.flatnonzero(visible):

            cv2.circle(
                frame,
                joints[index],
                radius,
                WHITE,
                -1,
                cv2.LINE_AA
            )

    return frame


class VideoExporter:

    def __init__(self, path, width, height, fps):

        self.encoder = None
        self.writer = None

        if ffmpeg is not None:

            try:

                even_dimensions = (
                    width % 2 == 0 and height % 2 == 0
                )

                self.encoder = ffmpeg.write_frames(
                    str(path),
                    (width, height),
                    pix_fmt_in="bgr24",
                    pix_fmt_out=(
                        "yuv420p" if even_dimensions else "yuv444p"
                    ),
                    fps=fps,
                    codec="libx264",
                    quality=7,
                    macro_block_size=1,
                    output_params=["-movflags", "+faststart"]
                )

                self.encoder.send(None)

                self.backend = "imageio-ffmpeg (H.264)"
                self.browser_support = even_dimensions

                return

            except Exception:

                if self.encoder is not None:
                    try:
                        self.encoder.close()
                    except Exception:
                        pass

                self.encoder = None

        self.writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

        if not self.writer.isOpened():
            raise RuntimeError("Unable to create output video.")

        self.backend = "OpenCV (mp4v)"
        self.browser_support = False

    def write_frame(self, frame):

        if self.encoder is not None:
            self.encoder.send(frame.tobytes())
        else:
            self.writer.write(frame)

    def release(self):

        if self.encoder is not None:
            self.encoder.close()
            self.encoder = None

        if self.writer is not None:
            self.writer.release()
            self.writer = None


def process_video(model, input_file, output_file, progress_callback):

    capture = cv2.VideoCapture(str(input_file))

    if not capture.isOpened():
        raise ValueError("Unable to open the selected video.")

    fps = capture.get(cv2.CAP_PROP_FPS)

    if not fps or fps <= 0 or fps > 1000:
        fps = 30.0

    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    exporter = None
    completed = 0

    try:

        while True:

            success, image = capture.read()

            if not success:
                break

            if exporter is None:

                height, width = image.shape[:2]

                thickness = max(
                    2,
                    round(min(width, height) / 240)
                )

                radius = thickness + 2

                exporter = VideoExporter(
                    output_file,
                    width,
                    height,
                    fps
                )

            detection = model(
                image,
                conf=PERSON_CONF,
                verbose=False
            )[0]

            skeleton = render_skeleton(
                detection,
                width,
                height,
                thickness,
                radius
            )

            exporter.write_frame(skeleton)

            completed += 1

            progress_callback(completed, total)

    finally:

        capture.release()

        if exporter is not None:
            exporter.release()

    if completed == 0:
        raise ValueError("No readable frames found in the video.")

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "frames": completed,
        "encoder": exporter.backend,
        "browser_friendly": exporter.browser_support
    }


def main():

    st.set_page_config(
        page_title="Pose Skeleton Generator",
        page_icon="🦴",
        layout="centered"
    )

    st.title("Human Pose Skeleton Video Generator")

    st.write(
        "Upload a video to detect human poses and generate "
        "a skeleton-only video with white joints and limbs "
        "displayed against a black background."
    )

    uploaded_file = st.file_uploader(
        "Upload Video",
        type=VIDEO_FORMATS
    )

    if uploaded_file is not None:

        if st.session_state.get("file_id") != uploaded_file.file_id:

            st.session_state.file_id = uploaded_file.file_id
            st.session_state.pop("result", None)

    if st.button(
        "Generate Pose Video",
        type="primary",
        disabled=uploaded_file is None
    ):

        detector, model_name, warning = initialize_detector()

        if warning:
            st.warning(warning)

        progress = st.progress(0.0, text="Preparing video...")

        def update_progress(current, total):

            if current % 5 != 0 and current != total:
                return

            if total > 0:

                progress.progress(
                    min(current / total, 1.0),
                    text=f"Processing frame {current} of {total}"
                )

            else:

                progress.progress(
                    0.0,
                    text=f"Processing frame {current}"
                )

        extension = Path(uploaded_file.name).suffix or ".mp4"

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension
        ) as temp:

            temp.write(uploaded_file.getbuffer())
            input_path = Path(temp.name)

        try:

            details = process_video(
                detector,
                input_path,
                OUTPUT_FILE,
                update_progress
            )

            st.session_state.result = {
                **details,
                "model": model_name
            }

        except Exception as error:

            progress.empty()
            st.error(f"Processing failed: {error}")

        else:

            progress.progress(1.0, text="Video generation completed!")

        finally:

            input_path.unlink(missing_ok=True)

    result = st.session_state.get("result")

    if result and OUTPUT_FILE.exists():

        st.success("Pose video generated successfully!")

        st.caption(
            f"Resolution: {result['width']}x{result['height']} | "
            f"FPS: {result['fps']:.2f} | "
            f"Frames: {result['frames']} | "
            f"Duration: {result['frames'] / result['fps']:.1f}s | "
            f"Model: {result['model']} | "
            f"Encoder: {result['encoder']}"
        )

        if not result["browser_friendly"]:

            st.info(
                "The video may not preview correctly in your browser. "
                "Download it and open it using VLC or another "
                "desktop video player."
            )

        video_content = OUTPUT_FILE.read_bytes()

        st.video(
            video_content,
            format="video/mp4"
        )

        st.download_button(
            "Download Pose Video",
            data=video_content,
            file_name="pose_result.mp4",
            mime="video/mp4"
        )


if __name__ == "__main__":
    main()
```
