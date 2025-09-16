import numpy as np
import matplotlib.pyplot as plt
import cv2

class SaturationSettings:
    def __init__(self,
                 primary_direction="Right",
                 brightness_curve=1.0,
                 brightness=1.0,
                 secondary_direction="Horizontal",
                 brightness_concave_power=1.5,
                 secondary_BrightGain=1.0,
                 saturation_ununiform=1):
        self.primary_direction = primary_direction
        self.brightness_curve = brightness_curve
        self.brightness = brightness
        self.secondary_direction = secondary_direction
        self.brightness_concave_power = brightness_concave_power
        self.secondary_BrightGain = secondary_BrightGain
        self.saturation_ununiform = saturation_ununiform


image_bgr = cv2.imread(r"C:\Users\faezeh.rabbani\FACEIT_DATA\Frames\output_image.png")
#Load frame
# === Load a frame from a video ===


def change_Gradual_saturation(image_bgr: np.ndarray, settings: SaturationSettings):

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, w = hsv.shape[:2]
    gradient = np.ones((h, w), dtype=np.float32)

    def get_direction_mask(direction, curve, strength):
        if direction == "Right":
            return np.tile(np.linspace(1.0, strength, w) ** curve, (h, 1))
        elif direction == "Left":
            return np.tile(np.linspace(strength, 1.0, w) ** curve, (h, 1))
        elif direction == "Down":
            return np.tile(np.linspace(1.0, strength, h) ** curve, (w, 1)).T
        elif direction == "UP":
            return np.tile(np.linspace(strength, 1.0, h) ** curve, (w, 1)).T
        else:
            raise ValueError(f"Unknown direction: {direction}")

    # Apply primary direction
    if settings.primary_direction is not None:
        gradient *= get_direction_mask(settings.primary_direction,
         settings.brightness_curve, settings.brightness)


    def get_symmetric_concave_mask(h: int, w: int, direction="Horizontal", strength=2.0, power = 1.5):
        size = w if direction in ["Horizontal"] else h
        x = np.linspace(-1, 1, size)
        curve = np.abs(x) ** power
        scaled = 1 + (strength - 1) * curve
        if direction == "Horizontal":
            return np.tile(scaled, (h, 1))
        elif direction == "Vertical":
            return np.tile(scaled, (w, 1)).T

    if settings.secondary_direction is not None:
        gradient *= get_symmetric_concave_mask(h, w, "Vertical",
          settings.secondary_BrightGain, settings.brightness_concave_power)

    # Apply gradient to brightness channel
    hsv[..., 2] *= gradient
    hsv[..., 2] = np.clip(hsv[..., 2], 0, 255)

    # change saturation channel
    hsv[..., 1] *= settings.saturation_ununiform
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)

    # Convert back to RGB
    bgr_result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    rgb_result = cv2.cvtColor(bgr_result, cv2.COLOR_BGR2RGB)
    return rgb_result

def get_direction_mask(direction, curve, strength):
    if direction == "Right":
        return np.tile(np.linspace(1.0, strength, width) ** curve, (height, 1))
    elif direction == "Left":
        return np.tile(np.linspace(strength, 1.0, width) ** curve, (height, 1))
    elif direction == "Down":
        return np.tile(np.linspace(1.0, strength, height) ** curve, (width, 1)).T
    elif direction == "UP":
        return np.tile(np.linspace(strength, 1.0, height) ** curve, (width, 1)).T
    else:
        raise ValueError(f"Unknown direction: {direction}")
width = 360
height = 240

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Original Image
axes[0,0].imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
axes[0,0].set_title("A. Original Image")
axes[0,0].axis('off')

# HSV Components
hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
axes[0,1].imshow(hsv[...,0], cmap='hsv')
axes[0,1].set_title("B. Hue Channel")
axes[0,2].imshow(hsv[...,1], cmap='gray')
axes[0,2].set_title("C. Saturation Channel")
axes[1,0].imshow(hsv[...,2], cmap='gray')
axes[1,0].set_title("D. Value Channel")

# Gradient Visualization
gradient = get_direction_mask("Right", 1.0, 2.0)
axes[1,1].imshow(gradient, cmap='viridis')
axes[1,1].set_title("E. Primary Gradient Mask\n(Linear Right)")

# Final Result
result = change_Gradual_saturation(image_bgr, SaturationSettings())

axes[1,2].imshow(result)
axes[1,2].set_title("F. Processed Result")

plt.tight_layout()
plt.show()



def get_symmetric_concave_mask(h: int, w: int, direction="Horizontal", strength=2.0, power=1.5):
    size = w if direction in ["Horizontal"] else h
    x = np.linspace(-1, 1, size)
    curve = np.abs(x) ** power
    scaled = 1 + (strength - 1) * curve
    if direction == "Horizontal":
        return np.tile(scaled, (h, 1))
    elif direction == "Vertical":
        return np.tile(scaled, (w, 1)).T



fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Primary Directions
directions = ["Right", "Left", "Down", "UP"]

for i, ax in enumerate(axes.flat[:4]):
    mask = get_direction_mask(directions[i], 1.0, 2.0)
    im = ax.imshow(mask, cmap='viridis')
    ax.set_title(f"{directions[i]} Gradient")
    fig.colorbar(im, ax=ax, label='Brightness Multiplier')

plt.tight_layout()
plt.show()

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Create grid
h, w = 200, 300
X, Y = np.meshgrid(np.arange(w), np.arange(h))
Z = get_symmetric_concave_mask(h, w, "Horizontal", 2.0, 1.5)

# Plot surface
surf = ax.plot_surface(X, Y, Z, cmap='viridis', rstride=10, cstride=10)

# Axis labels
ax.set_xlabel("Image Width")
ax.set_ylabel("Image Height")
ax.set_zlabel("Brightness Multiplier")

# Make background panes and grid lines transparent
ax.xaxis.set_pane_color((1, 1, 1, 0))
ax.yaxis.set_pane_color((1, 1, 1, 0))
ax.zaxis.set_pane_color((1, 1, 1, 0))

ax.xaxis._axinfo['grid']['color'] = (1, 1, 1, 0)
ax.yaxis._axinfo['grid']['color'] = (1, 1, 1, 0)
ax.zaxis._axinfo['grid']['color'] = (1, 1, 1, 0)

# Colorbar and title
fig.colorbar(surf, label='Intensity')
plt.title("3D Gradient Profile (Horizontal Concave)")

plt.tight_layout()
plt.show()

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

def get_direction_mask(h, w, direction, curve, strength):
    if direction == "Right":
        return np.tile(np.linspace(1.0, strength, w) ** curve, (h, 1))
    elif direction == "Left":
        return np.tile(np.linspace(strength, 1.0, w) ** curve, (h, 1))
    elif direction == "Down":
        return np.tile(np.linspace(1.0, strength, h) ** curve, (w, 1)).T
    elif direction == "UP":
        return np.tile(np.linspace(strength, 1.0, h) ** curve, (w, 1)).T
    else:
        raise ValueError(f"Unknown direction: {direction}")

def get_symmetric_concave_mask(h, w, direction="Horizontal", strength=2.0, power=1.5):
    size = w if direction == "Horizontal" else h
    x = np.linspace(-1, 1, size)
    curve = np.abs(x) ** power
    scaled = 1 + (strength - 1) * curve
    if direction == "Horizontal":
        return np.tile(scaled, (h, 1))
    elif direction == "Vertical":
        return np.tile(scaled, (w, 1)).T

def plot_merged_gradient_3D(settings: SaturationSettings):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    h, w = 200, 300
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    gradient = np.ones((h, w), dtype=np.float32)

    # Apply primary gradient
    if settings.primary_direction:
        gradient *= get_direction_mask(h, w, settings.primary_direction,
                                       settings.brightness_curve, settings.brightness)

    # Apply secondary gradient
    if settings.secondary_direction:
        gradient *= get_symmetric_concave_mask(h, w, settings.secondary_direction,
                                               settings.secondary_BrightGain, settings.brightness_concave_power)

    # Plot the combined gradient
    surf = ax.plot_surface(X, Y, gradient, cmap='viridis', rstride=10, cstride=10)

    ax.set_xlabel("Image Width")
    ax.set_ylabel("Image Height")
    ax.set_zlabel("Merged Brightness Multiplier")

    # Transparent background
    ax.xaxis.set_pane_color((1, 1, 1, 0))
    ax.yaxis.set_pane_color((1, 1, 1, 0))
    ax.zaxis.set_pane_color((1, 1, 1, 0))
    ax.xaxis._axinfo['grid']['color'] = (1, 1, 1, 0)
    ax.yaxis._axinfo['grid']['color'] = (1, 1, 1, 0)
    ax.zaxis._axinfo['grid']['color'] = (1, 1, 1, 0)

    fig.colorbar(surf, shrink=0.6, aspect=20, label='Intensity')
    plt.title("3D Merged Gradient Profile")
    plt.tight_layout()
    plt.show()

# Example usage
settings = SaturationSettings(
    primary_direction="Right",
    brightness_curve=1.0,
    brightness=1.5,
    secondary_direction="Vertical",
    brightness_concave_power=1.5,
    secondary_BrightGain=2.0
)

plot_merged_gradient_3D(settings)
