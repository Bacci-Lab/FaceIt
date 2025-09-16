import matplotlib.pyplot as plt
import csv
import pandas as pd
import random
import math
import numpy as np
import cv2
import os

def generate_occluded_pupil_frame(save=True, show=True):
    # === Image and anatomical parameters ===
    width, height = 640, 480
    eye_center = (width // 2, height // 2)
    eye_axes = (220, 160)
    pupil_center = eye_center
    pupil_radius = 50
    pupil_axes = (pupil_radius, pupil_radius)

    # === Create background with eye and pupil ===
    img = np.full((height, width), 80, dtype=np.uint8)  # face color
    cv2.ellipse(img, eye_center, eye_axes, 0, 0, 360, 170, thickness=-1)  # iris
    cv2.ellipse(img, pupil_center, pupil_axes, 0, 0, 360, 90, thickness=-1)  # pupil

    # === Add a smaller, offset reflection ===
    reflection_radius = 20  # smaller than before
    reflection_center = (pupil_center[0] + 30, pupil_center[1] - 25)  # offset more outward

    overlay = img.copy()
    cv2.circle(overlay, reflection_center, reflection_radius, 250, -1)  # white reflection
    blurred = cv2.GaussianBlur(overlay, (13, 13), 0)

    # Combine reflection with original image
    img = np.maximum(img, blurred)

    # === Optional display ===
    if show:
        cv2.imshow("Pupil with Light Reflection (Less Covered)", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # === Optional save ===
    if save:
        out_path = r"C:\Users\faezeh.rabbani\Desktop\FACEIT\compare_both\Condition_5.random_trajectory_dilated_reflection\frame_example.png"
        cv2.imwrite(out_path, img)
        print(f"[✓] Saved frame to: {out_path}")

    return img

def generate_occluded_pupil_frame_with_shadow(save=True, show=False):
    # === Parameters ===
    width, height = 640, 480
    eye_center = (width // 2, height // 2)
    eye_axes = (220, 160)
    pupil_radius = 50
    pupil_axes = (pupil_radius, pupil_radius)
    pupil_center = eye_center

    # === Base image (gray face)
    img = np.full((height, width), 80, dtype=np.uint8)

    # === Draw iris and pupil
    cv2.ellipse(img, eye_center, eye_axes, 0, 0, 360, 170, thickness=-1)  # iris
    cv2.ellipse(img, pupil_center, pupil_axes, 0, 0, 360, 90, thickness=-1)  # pupil

    # === Add shadow to iris (left to center gradient)
    x0 = eye_center[0] - eye_axes[0]
    x1 = eye_center[0]
    gradient = np.clip((x1 - np.arange(width)) / (x1 - x0), 0, 1)
    mask = np.tile(gradient, (height, 1)) * 70  # shadow intensity
    Y, X = np.ogrid[:height, :width]
    ellipse_mask = ((X - eye_center[0])**2 / eye_axes[0]**2 + (Y - eye_center[1])**2 / eye_axes[1]**2) <= 1
    # shadow = (mask * ellipse_mask).astype(np.uint8)
    # img = np.clip(img.astype(np.int16) - shadow.astype(np.int16), 0, 255).astype(np.uint8)

    # === Add reflection (blurred circle)
    reflection_radius = 20
    reflection_center = (pupil_center[0] + 30, pupil_center[1] - 25)
    overlay = img.copy()
    # cv2.circle(overlay, reflection_center, reflection_radius, 250, -1)
    blurred = cv2.GaussianBlur(overlay, (13, 13), 0)
    img = np.maximum(img, blurred)

    # === Optional display
    if show:
        cv2.imshow("Pupil with Reflection and Shadow", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # === Optional save
    if save:
        out_path = r"C:\Users\faezeh.rabbani\Desktop\FACEIT\compare_both\Condition_1_fix_calibration\sample.png"
        cv2.imwrite(out_path, img)
        print(f"[✓] Saved to: {out_path}")

    return img
generate_occluded_pupil_frame_with_shadow()

def show_and_save_static_pupil_frame(dilation=True):
    # === PARAMETERS ===
    width, height = 480, 360
    eye_center = (width // 2, height // 2)
    eye_axes = (250, 200)
    base_radius = 60
    dilation_amplitude = 35

    # === Dilation logic ===
    if dilation:
        t = 0  # time point
        pupil_radius = int(base_radius + dilation_amplitude * np.sin(t))
    else:
        pupil_radius = base_radius

    pupil_axes = (pupil_radius, pupil_radius)

    # === Create frame ===
    frame = np.full((height, width), 255, dtype=np.uint8)
    cv2.ellipse(frame, eye_center, eye_axes, 0, 0, 360, 120, -1)
    cv2.ellipse(frame, eye_center, pupil_axes, 0, 0, 360, 90, -1)

    # === Show frame ===
    cv2.imshow("Static Pupil Frame", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # === Save frame ===
    save_dir = r"C:\Users\faezeh.rabbani\Desktop\FACEIT\compare_both\Condition_1_fix_calibration"
    os.makedirs(save_dir, exist_ok=True)

    suffix = "dilated" if dilation else "fixed"
    filename = os.path.join(save_dir, f"sample_frame_{suffix}.png")
    cv2.imwrite(filename, frame)
    print(f"[✓] Frame saved as: {filename}")
show_and_save_static_pupil_frame()

def module_static_pupil_calibration1(dilation=True):

    # === PARAMETERS ===
    width, height = 480, 360
    fps = 30
    duration_seconds = 10
    num_frames = fps * duration_seconds

    # Eyeball & Pupil parameters
    eye_center = (width // 2, height // 2)
    eye_axes = (250, 200)
    base_radius = 60
    dilation_amplitude = 35  # How much the radius varies if dilation is True

    # === Output setup ===
    out_folder = r"C:\Users\faezeh.rabbani\Desktop\FACEIT\fix_calibration\pupil_calibration_dilation5"
    os.makedirs(out_folder, exist_ok=True)

    suffix = "_dilated" if dilation else "_fixed"
    out_path = os.path.join(out_folder, f'calibration{suffix}.mp4')
    csv_path = os.path.join(out_folder, f'calibration{suffix}.csv')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height), isColor=True)

    # === Data logging ===
    records = []

    for i in range(num_frames):
        # === Dilation logic ===
        if dilation:
            t = 2 * np.pi * i / num_frames
            pupil_radius = int(base_radius + dilation_amplitude * np.sin(t))
        else:
            pupil_radius = base_radius

        pupil_axes = (pupil_radius, pupil_radius)

        # === Draw frame ===
        frame = np.full((height, width), 255, dtype=np.uint8)
        cv2.ellipse(frame, eye_center, eye_axes, 0, 0, 360, 120, -1)
        cv2.ellipse(frame, eye_center, pupil_axes, 0, 0, 360, 90, -1)
        out.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))

        # === Record info ===
        area = np.pi * pupil_radius ** 2
        records.append({
            'frame': i,
            'center_x': eye_center[0],
            'center_y': eye_center[1],
            'radius': pupil_radius,
            'area': area
        })

    # === Save CSV ===
    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)

    # === Finalize ===
    out.release()
    print(f"[✓] Static pupil video saved to: {out_path}")
    print(f"[✓] Data saved to: {csv_path}")


def module_pupil_random_movement_fixations_optional_dilation(
    width=640,
    height=480,
    fps=30,
    duration_seconds=10,
    pupil_base_radius=45,
    dilation_amp=10,
    apply_dilation=True
):
    num_frames = fps * duration_seconds
    eye_center = (width // 2, height // 2)
    eye_axes = (int(width * 0.4), int(height * 0.35))  # visible ellipse
    pupil_axes_func = lambda r: (r, r)
    fixation_min = int(0.4 * fps)
    fixation_max = int(1.5 * fps)
    saccade_min = 2
    saccade_max = 4

    out_folder = r"C:\Users\faezeh.rabbani\Desktop\FACEIT\fix_trajectory_motion\dilation_4"
    os.makedirs(out_folder, exist_ok=True)
    out_path = os.path.join(out_folder, "dilation_trajectory_motion4.mp4")
    csv_path = os.path.join(out_folder, "dilation_trajectory_motion4.csv")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height), isColor=True)

    xs, ys, areas = [], [], []
    pupil_centers = []
    current_frame = 0
    current_center = eye_center

    # Inner ellipse for safe pupil motion
    margin = 1.2
    inner_eye_axes = (
        int(eye_axes[0] - margin * pupil_base_radius),
        int(eye_axes[1] - margin * pupil_base_radius)
    )

    def inside_safe_ellipse(x, y):
        ex, ey = eye_center
        a, b = inner_eye_axes
        return ((x - ex)**2 / a**2) + ((y - ey)**2 / b**2) <= 1.0

    while current_frame < num_frames:
        N_fix = random.randint(fixation_min, fixation_max)
        N_fix = min(N_fix, num_frames - current_frame)
        for _ in range(N_fix):
            pupil_centers.append(current_center)
        current_frame += N_fix
        if current_frame >= num_frames:
            break

        # Choose a new pupil center within inner ellipse
        while True:
            dx = random.uniform(-inner_eye_axes[0], inner_eye_axes[0])
            dy = random.uniform(-inner_eye_axes[1], inner_eye_axes[1])
            tx = eye_center[0] + dx
            ty = eye_center[1] + dy
            if inside_safe_ellipse(tx, ty):
                new_target = (int(tx), int(ty))
                break

        N_sac = random.randint(saccade_min, saccade_max)
        N_sac = min(N_sac, num_frames - current_frame)
        cx, cy = current_center
        tx, ty = new_target
        for f in range(1, N_sac + 1):
            alpha = f / float(N_sac)
            inter_x = int(cx + (tx - cx) * alpha)
            inter_y = int(cy + (ty - cy) * alpha)
            pupil_centers.append((inter_x, inter_y))
        current_frame += N_sac
        current_center = new_target

    pupil_centers = pupil_centers[:num_frames]

    # Radius logic
    counter = 0
    stable_period = random.randint(fixation_min, fixation_max)
    current_radius = pupil_base_radius

    xs, ys, areas = [], [], []
    pupil_radii = []
    min_radius = 10
    max_radius = 60

    for i in range(num_frames):
        if apply_dilation and counter >= stable_period:
            current_radius = int(
                pupil_base_radius + random.uniform(-dilation_amp, dilation_amp)
            )
            current_radius = max(min_radius, min(current_radius, max_radius))
            stable_period = random.randint(fixation_min, fixation_max)
            counter = 0
        pc = pupil_centers[i]

        if apply_dilation and counter >= stable_period:
            current_radius = int(
                pupil_base_radius + random.uniform(-dilation_amp, dilation_amp)
            )
            current_radius = max(min_radius, min(current_radius, max_radius))
            stable_period = random.randint(fixation_min, fixation_max)
            counter = 0
        counter += 1

        r = current_radius

        # Clamp pupil radius based on available space (ONLY if dilation is used)
        if apply_dilation:
            dx = pc[0] - eye_center[0]
            dy = pc[1] - eye_center[1]
            a, b = inner_eye_axes
            if dx == 0 and dy == 0:
                max_r = min(a, b)
            else:
                dist_center = math.hypot(dx, dy)
                t_param = 1.0 / math.sqrt((dx*dx)/(a*a) + (dy*dy)/(b*b))
                boundary_dist = dist_center * t_param
                max_r = int(boundary_dist - dist_center)
            r = min(r, max_r)
        else:
            r = pupil_base_radius  # constant if no dilation

        frame = np.full((height, width), 255, dtype=np.uint8)  # white background
        cv2.ellipse(frame, eye_center, eye_axes, 0, 0, 400, 120, -1)  # light gray eyeball
        cv2.ellipse(frame, pc, pupil_axes_func(r), 0, 0, 360, 90, -1)  # black pupil
        out.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))

        xs.append(pc[0])
        ys.append(pc[1])
        pupil_radii.append(r)
        areas.append(math.pi * r ** 2)
    out.release()

    df = pd.DataFrame({
        "frame_index": range(num_frames),
        "center_x": xs,
        "center_y": ys,
        "pupil_radius": pupil_radii,
        "pupil_area": areas,
    })
    df.to_csv(csv_path, index=False)


    return out_path, csv_path



def module_sin_noise():
    def generate_pupil_frame(shape, eye_center, eye_axes, pupil_center, pupil_axes,
                             face_color=255, eye_color=120, pupil_color=90):
        img = np.full(shape, face_color, dtype=np.uint8)
        cv2.ellipse(img, eye_center, eye_axes, 0, 0, 400, eye_color, thickness=-1)
        cv2.ellipse(img, pupil_center, pupil_axes, 0, 0, 360, pupil_color, thickness=-1)
        return img

    def add_reflection(img, center, radius, intensity=250):
        overlay = img.copy()
        cv2.circle(overlay, center, radius, (intensity,), -1)
        blurred = cv2.GaussianBlur(overlay, (radius * 2 + 1, radius * 2 + 1), 0)
        return np.maximum(img, blurred)

    def add_noise(img, sigma=15):
        noise = np.random.normal(0, sigma, img.shape).astype(np.int16)
        noisy = img.astype(np.int16) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

    # === PARAMETERS ===
    sigma_noise = 15
    width, height = 320, 240
    fps = 30
    duration_seconds = 10
    num_frames = fps * duration_seconds

    eye_center = (width // 2, height // 2)
    eye_axes = (180, 70)

    pupil_base_radius = 30
    dilation_amp = 10
    pupil_axes = lambda r: (r, r)

    t_k = 0.6

    offset = (40, -40)
    stable_ref_center = (eye_center[0] + offset[0], eye_center[1] + offset[1])
    ref_radius = 15

    out_folder = r"C:\Users\faezeh.rabbani\Desktop\FACEIT\sin_noise\15_reflection"
    os.makedirs(out_folder, exist_ok=True)
    out_path = os.path.join(out_folder, 'sin_noise.mp4')
    csv_path = os.path.join(out_folder, 'sin_noise_data.csv')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height), isColor=True)

    records = []

    for i in range(num_frames):
        t = 2 * np.pi * i / num_frames
        r = int(pupil_base_radius + dilation_amp * np.sin(4 * t))

        amp_x = (eye_axes[0] - r) * t_k
        amp_y = (eye_axes[1] - r) * t_k
        dx = amp_x * np.cos(t)
        dy = amp_y * np.sin(t * 1.5)
        cx = int(eye_center[0] + dx)
        cy = int(eye_center[1] + dy)

        frame = generate_pupil_frame((height, width), eye_center, eye_axes, (cx, cy), pupil_axes(r))
        frame = add_reflection(frame, stable_ref_center, ref_radius)
        frame = add_noise(frame, sigma=sigma_noise)

        out.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))

        # Record data
        area = np.pi * r ** 2
        records.append({
            'frame': i,
            'center_x': cx,
            'center_y': cy,
            'radius': r,
            'area': area,
            'noise_sigma': sigma_noise
        })

    out.release()

    # Save CSV
    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)

    print(f"[✓] Video saved: {out_path}")
    print(f"[✓] Data saved: {csv_path}")


def module_pupil_random_movement_fixations_dilation_reflection(shadow = False):
    def generate_pupil_frame(shape, eye_center, eye_axes, pupil_center, pupil_axes,
                             face_color=80, eye_color=170, pupil_color=135):
        """Draws a gray-scale frame with a filled ellipse (iris) and smaller ellipse (pupil)."""
        img = np.full(shape, face_color, dtype=np.uint8)
        cv2.ellipse(img, eye_center, eye_axes, 0, 0, 360, eye_color, thickness=-1)
        cv2.ellipse(img, pupil_center, pupil_axes, 0, 0, 360, pupil_color, thickness=-1)
        return img

    def add_reflections(img, base_center, radius, num=3, intensity=230, jitter_amp=1, t=0):
        """Add multiple small specular reflections that jitter around base_center."""
        out = img.copy()
        for i in range(num):
            angle = t + i * (2 * np.pi / num)
            jitter = (int(jitter_amp * np.cos(angle)), int(jitter_amp * np.sin(angle)))
            center = (base_center[0] + jitter[0], base_center[1] + jitter[1])
            overlay = out.copy()
            cv2.circle(overlay, center, max(1, radius // 2), (intensity,), -1)
            ksize = (11,11)
            blurred = cv2.GaussianBlur(overlay, ksize, 0)
            out = np.maximum(out, blurred)
        return out

    def add_shadow(img, eye_center, eye_axes, max_intensity=80):
        """Add a static gradient shadow from left edge of the iris fading toward center."""
        h, w = img.shape
        xx = np.arange(w)
        x0 = eye_center[0] - eye_axes[0]
        x1 = eye_center[0]
        gradient = np.clip((x1 - xx) / (x1 - x0), 0, 1)
        mask = np.tile(gradient, (h, 1))
        mask = mask * max_intensity

        Y, X = np.ogrid[:h, :w]
        ellipse_mask = ((X - eye_center[0])**2 / eye_axes[0]**2 +
                        (Y - eye_center[1])**2 / eye_axes[1]**2) <= 1
        shadow = (mask * ellipse_mask).astype(np.uint8)

        out = img.astype(np.int16) - shadow.astype(np.int16)
        return np.clip(out, 0, 255).astype(np.uint8)

    # === PARAMETERS ===
    width, height    = 640, 480
    fps              = 30
    duration_seconds = 10
    num_frames       = fps * duration_seconds

    eyec    = (width // 2, height // 2)
    eye_axes = (220, 150)

    pupil_base_radius = 45
    dilation_amp      = 10  # amplitude of random dilation
    pupil_axes = lambda r: (r, r)

    # Reflection parameters

    ref_base_offset = (25, -32)
    ref_base_center = (eyec[0] + ref_base_offset[0], eyec[1] + ref_base_offset[1])
    ref_radius      = 30

    # Output video setup
    out_folder = r"C:\Users\faezeh.rabbani\Desktop\FACEIT\compare_both\5.random_trajectory_dilated_reflection\3"
    os.makedirs(out_folder, exist_ok=True)

    out_path   = os.path.join(out_folder, "3.mp4")
    # Use H.264 ('avc1') instead of 'mp4v'
    fourcc     = cv2.VideoWriter_fourcc(*"avc1")
    out        = cv2.VideoWriter(out_path, fourcc, fps, (width, height), isColor=True)

    # ------------------------------------------
    # STEP A: Precompute a sequence of "pupil centers"
    # ------------------------------------------
    fix_min_frames = int(0.5 * fps)   # 0.5 s → 15 frames
    fix_max_frames = int(2.0 * fps)   # 2.0 s → 60 frames
    sac_min_frames = 2                # ~50 ms
    sac_max_frames = 4                # ~83 ms

    # Helper: check if (x, y) is inside the iris ellipse
    def inside_iris(x, y):
        ex, ey = eyec
        a, b = eye_axes
        return ((x - ex)**2 / a**2) + ((y - ey)**2 / b**2) <= 1.0

    pupil_centers = []
    current_frame = 0

    # Start at exact center (no offset)
    current_center = eyec

    while current_frame < num_frames:
        # 1) Fixation interval: stay at current_center for N_fix frames
        N_fix = random.randint(fix_min_frames, fix_max_frames)
        N_fix = min(N_fix, num_frames - current_frame)
        for _ in range(N_fix):
            pupil_centers.append(current_center)
        current_frame += N_fix
        if current_frame >= num_frames:
            break

        # 2) Choose a new random target inside the iris ellipse
        a_lim = eye_axes[0] - pupil_base_radius
        b_lim = eye_axes[1] - pupil_base_radius
        while True:
            dx = random.uniform(-a_lim, a_lim)
            dy = random.uniform(-b_lim, b_lim)
            tx = eyec[0] + dx
            ty = eyec[1] + dy
            if inside_iris(tx, ty):
                new_target = (int(tx), int(ty))
                break

        # 3) Saccade interval: linear interpolation over N_sac frames
        N_sac = random.randint(sac_min_frames, sac_max_frames)
        N_sac = min(N_sac, num_frames - current_frame)
        cx, cy = current_center
        tx, ty = new_target
        for f in range(1, N_sac + 1):
            alpha = f / float(N_sac)
            inter_x = int(cx + (tx - cx) * alpha)
            inter_y = int(cy + (ty - cy) * alpha)
            pupil_centers.append((inter_x, inter_y))
        current_frame += N_sac

        # 4) Update fixation center
        current_center = new_target

    pupil_centers = pupil_centers[:num_frames]

    # Initialize variables for intermittent random dilation and clamping
    counter = 0
    stable_period = random.randint(fix_min_frames, fix_max_frames)
    current_radius = pupil_base_radius

    # Prepare lists for Excel export
    xs = []
    ys = []
    areas = []
    radii = []

    # ------------------------------------------------------
    # STEP B: Write out each frame, save to video + PNG, record data
    # ------------------------------------------------------
    min_r = 20
    max_r = 30

    for i in range(num_frames):
        t = 2 * np.pi * i / float(num_frames)

        # 1) intermittent random dilation
        if counter >= stable_period:
            current_radius = int(
                pupil_base_radius + random.uniform(-dilation_amp, dilation_amp)
            )
            if current_radius > max_r:
                current_radius = max_r
            elif current_radius < min_r:
                current_radius = min_r
            # clamp it to [min_r, max_r]:
            # current_radius = max(min(current_radius, max_r), min_r)
            stable_period = random.randint(fix_min_frames, fix_max_frames)
            counter = 0
        r = current_radius
        counter += 1

        pc = pupil_centers[i]

        # 2) Clamp r so the pupil stays fully inside the iris ellipse
        dx = pc[0] - eyec[0]
        dy = pc[1] - eyec[1]
        a, b = eye_axes
        if dx == 0 and dy == 0:
            max_r = min(a, b)
        else:
            dist_center = math.hypot(dx, dy)
            # distance from center to iris boundary along direction (dx,dy)
            # Distance from pupil center to iris edge in the same direction
            if dx == 0 and dy == 0:
                max_r = min(a, b)
            else:
                dist_center = math.hypot(dx, dy)
                t_param = 1.0 / math.sqrt((dx * dx) / (a * a) + (dy * dy) / (b * b))
                boundary_dist = dist_center * t_param
                margin = 10  # pixels to ensure gap
                max_r = int(boundary_dist - dist_center - margin)

        r = min(r, max_r)

        # 3) Record data for this frame
        xs.append(pc[0])
        ys.append(pc[1])
        areas.append(math.pi * (r ** 2))
        radii.append(r)

        # 4) Generate and save the frame as PNG
        frame = generate_pupil_frame((height, width),
                                     eyec, eye_axes,
                                     pc, pupil_axes(r))
        if shadow:
            frame = add_shadow(frame, eyec, eye_axes, max_intensity=80)
        frame = add_reflections(frame, ref_base_center, ref_radius,
                                num=1, intensity=230, jitter_amp=1, t=t)


        # Convert to BGR, write to video
        out.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))

    out.release()

    # ------------------------------------------------------
    # STEP C: Export xs, ys, areas to Excel
    # ------------------------------------------------------
    # …after you finish filling xs, ys, areas…

    csv_path = os.path.join(out_folder, "output_data.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_index", "center_x", "center_y", "radius", "pupil_area"])
        for i in range(num_frames):
            writer.writerow([i, xs[i], ys[i], radii[i], areas[i]])

    print(f"Saved data to CSV: {csv_path}")

    plt.figure()
    plt.plot(np.arange(num_frames), areas, label='Iris Area')
    plt.xlabel('Frame')
    plt.ylabel('Area (pixels²)')
    plt.title('Iris Area Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_folder, "iris_area_over_time.png"))
    plt.show()

    plt.figure()
    plt.plot(np.arange(num_frames), xs, label='Center X')
    plt.plot(np.arange(num_frames), ys, label='Center Y')
    plt.xlabel('Frame')
    plt.ylabel('Position (pixels)')
    plt.title('Iris Center Coordinates Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_folder, "iris_center_coordinates.png"))
    plt.show()


def model_pupil_color_change(shadow = False):
    def generate_pupil_frame(shape, eye_center, eye_axes, pupil_center, pupil_axes,
                             face_color=120, eye_color=170, pupil_color=100):
        """Draws a gray-scale frame with a filled ellipse (iris) and smaller ellipse (pupil)."""
        img = np.full(shape, face_color, dtype=np.uint8)
        cv2.ellipse(img, eye_center, eye_axes, 0, 0, 360, eye_color, thickness=-1)
        cv2.ellipse(img, pupil_center, pupil_axes, 0, 0, 360, pupil_color, thickness=-1)
        return img

    def add_reflections(img, base_center, radius, num=3, intensity=230, jitter_amp=1, t=0):
        """Add multiple small specular reflections that jitter around base_center."""
        out = img.copy()
        for i in range(num):
            angle = t + i * (2 * np.pi / num)
            jitter = (int(jitter_amp * np.cos(angle)), int(jitter_amp * np.sin(angle)))
            center = (base_center[0] + jitter[0], base_center[1] + jitter[1])
            overlay = out.copy()
            cv2.circle(overlay, center, max(1, radius // 2), (intensity,), -1)
            ksize = (11,11)
            blurred = cv2.GaussianBlur(overlay, ksize, 0)
            out = np.maximum(out, blurred)
        return out

    def add_shadow(img, eye_center, eye_axes, max_intensity=80):
        """Add a static gradient shadow from left edge of the iris fading toward center."""
        h, w = img.shape
        xx = np.arange(w)
        x0 = eye_center[0] - eye_axes[0]
        x1 = eye_center[0]
        gradient = np.clip((x1 - xx) / (x1 - x0), 0, 1)
        mask = np.tile(gradient, (h, 1))
        mask = mask * max_intensity

        Y, X = np.ogrid[:h, :w]
        ellipse_mask = ((X - eye_center[0])**2 / eye_axes[0]**2 +
                        (Y - eye_center[1])**2 / eye_axes[1]**2) <= 1
        shadow = (mask * ellipse_mask).astype(np.uint8)

        out = img.astype(np.int16) - shadow.astype(np.int16)
        return np.clip(out, 0, 255).astype(np.uint8)

    # === PARAMETERS ===
    width, height    = 640, 480
    fps              = 30
    duration_seconds = 10
    num_frames       = fps * duration_seconds

    eyec    = (width // 2, height // 2)
    eye_axes = (220, 150)

    pupil_base_radius = 45
    dilation_amp      = 20  # amplitude of random dilation
    pupil_axes = lambda r: (r, r)

    # Reflection parameters
    ref_base_offset = (25, -32)
    ref_base_center = (eyec[0] + ref_base_offset[0], eyec[1] + ref_base_offset[1])
    ref_radius      = 30

    # Output video setup
    out_folder = r"C:\Users\faezeh.rabbani\Desktop\FACEIT\contrast_change_shadow\5"
    os.makedirs(out_folder, exist_ok=True)

    out_path   = os.path.join(out_folder, "contrast_change_shadow5.mp4")
    # Use H.264 ('avc1') instead of 'mp4v'
    fourcc     = cv2.VideoWriter_fourcc(*"avc1")
    out        = cv2.VideoWriter(out_path, fourcc, fps, (width, height), isColor=True)

    # ------------------------------------------
    # STEP A: Precompute a sequence of "pupil centers"
    # ------------------------------------------
    fix_min_frames = int(0.5 * fps)   # 0.5 s → 15 frames
    fix_max_frames = int(2.0 * fps)   # 2.0 s → 60 frames
    sac_min_frames = 2                # ~50 ms
    sac_max_frames = 4                # ~83 ms

    # Helper: check if (x, y) is inside the iris ellipse
    def inside_iris(x, y):
        ex, ey = eyec
        a, b = eye_axes
        return ((x - ex)**2 / a**2) + ((y - ey)**2 / b**2) <= 1.0

    pupil_centers = []
    current_frame = 0

    # Start at exact center (no offset)
    current_center = eyec

    while current_frame < num_frames:
        # 1) Fixation interval: stay at current_center for N_fix frames
        N_fix = random.randint(fix_min_frames, fix_max_frames)
        N_fix = min(N_fix, num_frames - current_frame)
        for _ in range(N_fix):
            pupil_centers.append(current_center)
        current_frame += N_fix
        if current_frame >= num_frames:
            break

        margin = 8  # or 10
        a_lim = eye_axes[0] - pupil_base_radius - margin
        b_lim = eye_axes[1] - pupil_base_radius - margin

        while True:
            dx = random.uniform(-a_lim, a_lim)
            dy = random.uniform(-b_lim, b_lim)
            tx = eyec[0] + dx
            ty = eyec[1] + dy
            if inside_iris(tx, ty):
                new_target = (int(tx), int(ty))
                break

        # 3) Saccade interval: linear interpolation over N_sac frames
        N_sac = random.randint(sac_min_frames, sac_max_frames)
        N_sac = min(N_sac, num_frames - current_frame)
        cx, cy = current_center
        tx, ty = new_target
        for f in range(1, N_sac + 1):
            alpha = f / float(N_sac)
            inter_x = int(cx + (tx - cx) * alpha)
            inter_y = int(cy + (ty - cy) * alpha)
            pupil_centers.append((inter_x, inter_y))
        current_frame += N_sac

        # 4) Update fixation center
        current_center = new_target

    pupil_centers = pupil_centers[:num_frames]

    # Initialize variables for intermittent random dilation and clamping
    counter = 0
    stable_period = random.randint(fix_min_frames, fix_max_frames)
    current_radius = pupil_base_radius

    # Prepare lists for Excel export
    xs = []
    ys = []
    areas = []
    eye_colors = []
    pupil_colors = []
    iris_amps = []

    # ------------------------------------------------------
    # STEP B: Write out each frame, save to video + PNG, record data
    # ------------------------------------------------------
    min_r = 10
    max_r = 12
    for i in range(num_frames):
        t = 2 * np.pi * i / float(num_frames)

        # 1) intermittent random dilation
        # Redetermine radius periodically
        if counter >= stable_period:
            current_radius = int(pupil_base_radius + random.uniform(-dilation_amp, dilation_amp))
            current_radius = max(min_r, current_radius)  # enforce lower limit
            stable_period = random.randint(fix_min_frames, fix_max_frames)
            counter = 0
        r = current_radius
        counter += 1

        pc = pupil_centers[i]

        # 2) Clamp r so the pupil stays fully inside the iris ellipse
        dx = pc[0] - eyec[0]
        dy = pc[1] - eyec[1]
        a, b = eye_axes
        # Keep pupil margin from edge (e.g., 5 pixels)
        margin = 10

        if dx == 0 and dy == 0:
            max_safe_r = min(a, b) - margin
        else:
            dist_center = math.hypot(dx, dy)
            t_param = 1.0 / math.sqrt((dx * dx) / (a * a) + (dy * dy) / (b * b))
            boundary_dist = dist_center * t_param
            max_safe_r = int(boundary_dist - dist_center - margin)

        r = min(r, max_safe_r)

        # 3) Record data for this frame

        xs.append(pc[0])
        ys.append(pc[1])
        areas.append(math.pi * (r ** 2))

        # 4) Generate and save the frame as PNG
        # … inside your loop, replace the eye_color calculation with:

        iris_base =120
        iris_amplitude = 25
        raw_color = iris_base + iris_amplitude * np.sin(t)
        eye_color = int(raw_color)
        ####################
        pupil_color = 75
        pupil_color_raw = int(pupil_color + iris_amplitude * np.sin(t))

        # Save for export
        eye_colors.append(eye_color)
        pupil_colors.append(pupil_color_raw)
        iris_amps.append(iris_amplitude)


        r = max(1, r)  # Ensure ellipse axes are positive
        face_color = 200
        frame = generate_pupil_frame(
            (height, width),
            eyec, eye_axes,
            pc, pupil_axes(r),
            face_color=face_color,
            eye_color=eye_color,
            pupil_color=pupil_color_raw
        )

        if shadow:

            frame = add_shadow(frame, eyec, eye_axes, max_intensity=80)

        frame = add_reflections(frame, ref_base_center, ref_radius,
                                num=1, intensity=230, jitter_amp=1, t=t)


        # Convert to BGR,pupil_axes(r) write to video
        out.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))

    out.release()

    # ------------------------------------------------------
    # STEP C: Export xs, ys, areas to Excel
    # ------------------------------------------------------
    # …after you finish filling xs, ys, areas…

    csv_path = os.path.join(out_folder, "output_data.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame_index", "center_x", "center_y", "pupil_area",
            "eye_color", "pupil_color", "iris_color_change_amplitude"
        ])
        for i in range(num_frames):
            writer.writerow([
                i, xs[i], ys[i], areas[i],
                eye_colors[i], pupil_colors[i], iris_amps[i]
            ])

    print(f"Saved data to CSV: {csv_path}")

    plt.figure()
    plt.plot(np.arange(num_frames), areas, label='Iris Area')
    plt.xlabel('Frame')
    plt.ylabel('Area (pixels²)')
    plt.title('Iris Area Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_folder, "iris_area_over_time.png"))
    plt.show()

    plt.figure()
    plt.plot(np.arange(num_frames), xs, label='Center X')
    plt.plot(np.arange(num_frames), ys, label='Center Y')
    plt.xlabel('Frame')
    plt.ylabel('Position (pixels)')
    plt.title('Iris Center Coordinates Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_folder, "iris_center_coordinates.png"))
    plt.show()
# model_pupil_color_change(shadow=True)