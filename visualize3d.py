import imageio
import os
from vedo import load, Plotter

# Define the path to your images and the output video file
video_path = "gt_144.mp4"
fps = 5  # Frames per second in the output video

# Prepare the video writer
writer = imageio.get_writer(video_path, fps=fps)

# Initialize Plotter with offscreen rendering
plotter = Plotter(offscreen=True)

for i in range(1, 16):  # Assuming 15 volumes, adjust as necessary
    file_path = f"eval_seg/seg_pred/scan_144_b{i:02d}.nii.gz"
    volume = load(file_path)
    volume.rotate_z(-90)

    plotter.clear()
    plotter.add(volume)
    
    # Render the scene
    plotter.show()  # In offscreen mode, this renders the scene without opening a window
    
    # Take a screenshot and add it to the video
    temp_screenshot_path = "temp_screenshot.png"
    plotter.screenshot(temp_screenshot_path)
    frame = imageio.imread(temp_screenshot_path)
    writer.append_data(frame)
    
    # Remove the temporary screenshot to clean up
    os.remove(temp_screenshot_path)

# Cleanup
writer.close()
plotter.close()

print(f"Video saved to {video_path}")
