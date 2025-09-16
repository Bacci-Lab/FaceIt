import cv2

def extract_frames_from_video(video_path, frame_indices):
    frames = []
    cap = cv2.VideoCapture(video_path)
    for frame_no in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        if ret:
            # Resize frame to half size
            frame_half = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            frames.append(frame_half)  # append resized frame
        else:
            print(f"Frame {frame_no} not found.")
    cap.release()
    return frames

# Example usage:
frame_list = [1998, 26, 767, 937]




video_path = r"Y:\users\faezeh.rabbani\faceit\65\faceit_65_230cd_2025-06-16\faceit_65_230cd_2025-06-16.avi"
frame_images = extract_frames_from_video(video_path, frame_list)


output_folder = r"C:\Users\faezeh.rabbani\PycharmProjects\FaceProject\Extrene_test\test_Error_frames\full_frame\65_230cd_2025-06-16"
for idx, img in zip(frame_list, frame_images):
    cv2.imwrite(f"{output_folder}/Error_{idx}.png", img)
