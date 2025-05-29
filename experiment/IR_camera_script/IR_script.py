import time
import board
import busio
import numpy as np
import adafruit_mlx90640
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
import shutil

# Global variables for region selection and live mode
regions = []       # Each region is a dict: {'x':, 'y':, 'width':, 'height':, 'patch':}
current_click = None
selection_mode = True  # When True, allow region selection via clicks
live_mode = False      # When True, live sensor acquisition and ROI average calc runs
max_regions = 4
roi_texts = []         # To store ROI average text objects (so we can remove them each frame)

# --------------------------
# Set up sensor and logging
# --------------------------
i2c = busio.I2C(board.SCL, board.SDA)
mlx = adafruit_mlx90640.MLX90640(i2c)
mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_4_HZ  # 4 Hz refresh rate

# Create a unique log file name (including date and time down to seconds)
filename = "mlx90640_data_" + time.strftime("%Y%m%d_%H%M%S") + ".txt"
log_file = open(filename, "a")

# Estimate bytes per frame (timestamp + 24x32 comma-separated pixel values)
dummy_line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} " + ','.join(['{:.2f}'.format(0.0) for _ in range(24*32)])
estimated_bytes = len(dummy_line) + 1  # add newline char

# Get free disk space and estimate recording capacity at 4Hz
total, used, free = shutil.disk_usage(".")
frames_per_second = 4
estimated_frames = free // estimated_bytes
estimated_seconds = estimated_frames / frames_per_second
estimated_hours = estimated_seconds / 3600

print(f"Free disk space: {free} bytes")
print(f"Estimated bytes per frame: {estimated_bytes} bytes")
print(f"Estimated recording capacity: {estimated_seconds:.1f} seconds (~{estimated_hours:.2f} hours)")

# --------------------------
# Set up figure, axes, and buttons
# --------------------------
fig, (ax_img, ax_info) = plt.subplots(2, 1, figsize=(12, 9), gridspec_kw={'height_ratios': [4, 1]})
plt.subplots_adjust(bottom=0.25)  # leave space for buttons
ax_img.set_title("Region Selection Mode:\nClick twice to define a region (max 4). Then click 'Start Live'.")

# Display an initial (blank) image.
initial_image = np.zeros((24, 32))
# (We display the flipped image for proper orientation)
im_disp = ax_img.imshow(np.fliplr(initial_image), vmin=0, vmax=60)
cbar = fig.colorbar(im_disp, ax=ax_img)
cbar.set_label('Temperature [$^{\circ}$C]', fontsize=14)

# Info panel (text displayed below the image)
ax_info.axis('off')
info_text_obj = ax_info.text(0.5, 0.5,
                             "Live Mode OFF\nSelect regions then click 'Start Live'",
                             ha='center', va='center', fontsize=14)

# Define button areas (using matplotlib.widgets.Button)
ax_button_start = plt.axes([0.1, 0.1, 0.2, 0.075])
ax_button_reset = plt.axes([0.4, 0.1, 0.2, 0.075])
ax_button_edit  = plt.axes([0.7, 0.1, 0.2, 0.075])
button_start = Button(ax_button_start, "Start Live")
button_reset = Button(ax_button_reset, "Reset Regions")
button_edit  = Button(ax_button_edit, "Edit Regions")

# --------------------------
# Define callback functions
# --------------------------

# --------------------------
# Define callback functions
# --------------------------
def on_click(event):
    """Handle mouse clicks for region selection (when not in live mode)."""
    global current_click, regions, selection_mode, live_mode
    if not selection_mode or live_mode:
        return
    if event.inaxes != ax_img:
        return
    x = event.xdata
    y = event.ydata
    # Only process clicks inside the image bounds (0 to 31 in x, 0 to 23 in y)
    if x is None or y is None or x < 0 or x > 31 or y < 0 or y > 23:
        return
    if current_click is None:
        current_click = (x, y)
        print(f"First corner at ({x:.2f}, {y:.2f})")
    else:
        # Second click: define rectangle from first click to current click
        x0, y0 = current_click
        x1, y1 = x, y
        x_min = min(x0, x1)
        x_max = max(x0, x1)
        y_min = min(y0, y1)
        y_max = max(y0, y1)
        width = x_max - x_min
        height = y_max - y_min
        if width == 0 or height == 0:
            print("Invalid region (zero size). Try again.")
            current_click = None
            return
        if len(regions) >= max_regions:
            print("Maximum number of regions reached.")
            current_click = None
            return
        # Create a rectangle patch to show the selected region.
        # (No transformation is needed here because user clicks are on the displayed image.)
        rect = patches.Rectangle((x_min - 0.5, y_min - 0.5), width, height,
                                 edgecolor='red', facecolor='none', linewidth=2)
        ax_img.add_patch(rect)
        regions.append({'x': x_min, 'y': y_min, 'width': width, 'height': height, 'patch': rect})
        fig.canvas.draw()
        print(f"Region added: ({x_min:.2f}, {y_min:.2f}, {width:.2f}, {height:.2f})")
        current_click = None

def start_live_callback(event):
    """Toggle live acquisition on/off."""
    global live_mode, selection_mode
    live_mode = not live_mode
    if live_mode:
        # When live mode starts, disable further region selection.
        selection_mode = False
        button_start.label.set_text("Stop Live")
        ax_img.set_title("Live Mode ON")
        info_text_obj.set_text("Live Mode ON\nCalculating ROI averages...")
    else:
        selection_mode = True
        button_start.label.set_text("Start Live")
        ax_img.set_title("Region Selection Mode:\nClick to define regions (max 4)")
        info_text_obj.set_text("Live Mode OFF\nSelect regions then click 'Start Live'")
    fig.canvas.draw()

def reset_regions_callback(event):
    """Clear all region selections."""
    global regions, current_click
    for reg in regions:
           rg['patch'].remove()
    regions.clear()
    current_click = None
    fig.canvas.draw()
    print("All regions cleared.")

def edit_regions_callback(event):
    """Switch back to editing mode (stop live if active)."""
    global live_mode, selection_mode
    if live_mode:
        live_mode = False
        button_start.label.set_text("Start Live")
    selection_mode = True
    ax_img.set_title("Edit Mode: Click to add regions (max 4)")
    info_text_obj.set_text("Live Mode OFF\nSelect regions then click 'Start Live'")
    fig.canvas.draw()

# Connect mouse click event to on_click callback
cid = fig.canvas.mpl_connect('button_press_event', on_click)
# Connect buttons to their callbacks
button_start.on_clicked(start_live_callback)
button_reset.on_clicked(reset_regions_callback)
button_edit.on_clicked(edit_regions_callback)

# --------------------------
# Main loop: Live acquisition and ROI average calculation
# --------------------------
frame = np.zeros((24 * 32,))
t_array = []

while True:
    if live_mode:
        t1 = time.monotonic()
        try:
            # Acquire a new frame from the sensor
            mlx.getFrame(frame)
            data_array = np.reshape(frame, (24, 32))

            # Log the frame with a full timestamp
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            pixel_values_str = ','.join(['{:.2f}'.format(val) for val in data_array.flatten()])
            log_file.write(f"{timestamp} {pixel_values_str}\n")
            log_file.flush()

            # Update the displayed image (flip horizontally for correct orientation)
            flipped_data = np.fliplr(data_array)
            im_disp.set_data(flipped_data)
            im_disp.set_clim(vmin=np.min(data_array), vmax=np.max(data_array))

            # Remove previous ROI average texts from the image (if any)
            for txt in roi_texts:
                txt.remove()
            roi_texts.clear()

            # For each region, convert display coordinates to raw sensor coordinates and calculate the average.

 (User clicks yield display coordinates; because we display np.fliplr(data_array), we must convert:
            #  raw_x = 31 - (x_disp + width) ... to ... raw_x_max = 31 - x_disp)
            info_str = "ROI Averages:\n"
            for i, reg in enumerate(regions):
                x_disp = reg['x']
                y_disp = reg['y']
                width = reg['width']
                height = reg['height']
                # Convert display (clicked) coordinates to raw data indices:
                raw_x_min = int(round(31 - (x_disp + width)))
                raw_x_max = int(round(31 - x_disp))
                raw_y_min = int(round(y_disp))
                raw_y_max = int(round(y_disp + height))
                # Clamp indices within sensor bounds
                raw_x_min = max(0, raw_x_min)
                raw_x_max = min(31, raw_x_max)
                raw_y_min = max(0, raw_y_min)
                raw_y_max = min(23, raw_y_max)
                if raw_x_max <= raw_x_min or raw_y_max <= raw_y_min:
                    roi_avg = float('nan')
                else:
                    roi = data_array[raw_y_min:raw_y_max, raw_x_min:raw_x_max]
                    roi_avg = np.mean(roi)
                info_str += f"Region {i+1}: {roi_avg:.2f} °C   "
                # Display the ROI average as text at the center of the region (using display coordinates)
                center_x = x_disp + width / 2
                center_y = y_disp + height / 2
                txt = ax_img.text(center_x, center_y, f"{roi_avg:.1f}",
                                  color="yellow", fontsize=12, ha="center", va="center")
                roi_texts.append(txt)

            info_str += f"\nRecording capacity: {estimated_seconds:.1f} s (~{estimated_hours:.2f} h)"
            info_text_obj.set_text(info_str)

            fig.canvas.draw()
            plt.pause(0.001)

            t_array.append(time.monotonic() - t1)
            if len(t_array) % 10 == 0:
                print('Sample Rate: {0:2.1f} fps'.format(len(t_array)/np.sum(t_array)))
        except Exception as e:
            print("Error in live loop:", e)
            time.sleep(0.1)
    else:
        plt.pause(0.1)