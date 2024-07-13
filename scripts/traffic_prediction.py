import tkinter as tk
from tkinter import ttk
from datetime import datetime
from PIL import Image, ImageTk
import optimized_path_map
import utils
import time

# Global variables to store congestion index and arrival time
congestion_index = None
arrival_time = None

def predict_congestion(date_time):
    global congestion_index, arrival_time
    if not date_time:
        return 
    selected_segment_value = selected_road_segment.get()
    selected_datetime = datetime_entry.get()
    selected_mode_value = selected_mode.get()
    route_text.config(state=tk.NORMAL)
    route_text.delete(1.0, tk.END)
    route_text.insert(tk.END, f"Selected Road Segment: {selected_segment_value}\n")
    route_text.insert(tk.END, f"Selected DateTime: {selected_datetime}\n")
    route_text.insert(tk.END, f"Selected Mode: {selected_mode_value}\n")
    route_text.insert(tk.END, f"Predicting Traffic Metrics...\n")
    route_text.update_idletasks()
    time.sleep(1)
    if selected_segment_value == "All Segments":
        if selected_mode_value == "Congestion Index":
            prediction_result, congestion_index, _ = utils.predict_traffic(date_time, mode="Congestion Index")
            route_text.insert(tk.END, f"{prediction_result}\n")
        elif selected_mode_value == "Arrival Time":
            prediction_result, _, arrival_time = utils.predict_traffic(date_time, mode="Arrival Time")
            route_text.insert(tk.END, f"{prediction_result}\n")
        elif selected_mode_value == "Both":
            prediction_result, congestion_index, arrival_time = utils.predict_traffic(date_time, mode="Both")
            route_text.insert(tk.END, f"{prediction_result}\n")
    else:
        road_segment_map = {
            'Maitighar-Putalisadak': 0,
            'New Baneshwor-Maitighar': 1,
            'New Baneshwor-Old Baneshwor': 2,
            'Old Baneshwor-Seto Pool': 3,
            'Putalisadak-Hattisar': 4,
            'Rato Pool-Hattisar': 5,
            'Seto Pool-Putalisadak': 6,
            'Seto Pool-Rato Pool': 7
        }
        segment_id = road_segment_map.get(selected_segment_value)
        if selected_mode_value == "Congestion Index":
            prediction_result = utils.individual_traffic_metrics(date_time, segment_id, 'congestion_index')
            route_text.insert(tk.END, f"Congestion Index: {utils.congestion_mapping(prediction_result)}\n")
        elif selected_mode_value == "Arrival Time":
            prediction_result = utils.individual_traffic_metrics(date_time, segment_id, 'arrival_time')
            route_text.insert(tk.END, f"Arrival Time: {prediction_result}\n")
        elif selected_mode_value == "Both":
            congestion_result = utils.individual_traffic_metrics(date_time, segment_id, 'congestion_index')
            arrival_result = utils.individual_traffic_metrics(date_time, segment_id, 'arrival_time')
            route_text.insert(tk.END, f"Congestion Index: {utils.congestion_mapping(congestion_result)}\n")
            route_text.insert(tk.END, f"Arrival Time: {arrival_result}\n")
    route_text.config(state=tk.DISABLED)


def display_image(img_path):
    img = Image.open(img_path)
    img = img.resize((400, 300), Image.LANCZOS)
    img_tk = ImageTk.PhotoImage(img)
    route_text.image_create(tk.END, image=img_tk)
    route_text.insert(tk.END, "\n")
    route_text.image = img_tk  # Keep a reference to the image

def optimize_path():
    global congestion_index, arrival_time
    start = 'New Baneshwor'
    end = 'Hattisar'
    selected_segment_value = selected_road_segment.get()
    selected_mode_value = selected_mode.get()
    if congestion_index is None or arrival_time is None:
        route_text.config(state=tk.NORMAL)
        route_text.insert(tk.END, "Please predict traffic before optimizing the path.\n")
        route_text.config(state=tk.DISABLED)
        return
    if selected_mode_value != "Both" and selected_segment_value != "All Segments":
        route_text.config(state=tk.NORMAL)
        route_text.insert(tk.END, "Please select all road segment and both traffic metrics.\n")
        route_text.config(state=tk.DISABLED)
        return
    graph = utils.create_graph(utils.road_segments, congestion_index, arrival_time)
    optimal_path, best_cost, best_time = utils.ant_colony_optimization(graph, start, end)
    route_text.config(state=tk.NORMAL)
    route_text.insert(tk.END, f"Optimal path from {start} to {end}: {' -> '.join(optimal_path)}\n")
    route_text.insert(tk.END, f"Best cost: {best_cost}, Best time: {best_time}\n")
    route_text.config(state=tk.DISABLED)
    optimized_path_map.show_optimized_path(optimal_path)

def update_datetime():
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M")
    datetime_entry.config(state=tk.NORMAL)
    datetime_entry.delete(0, tk.END)
    datetime_entry.insert(0, current_datetime)

root = tk.Tk()
root.title("Traffic Congestion Prediction and Route Optimization")

# Center the window on the screen
window_width = 650
window_height = 750
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = (screen_width // 2) - (window_width // 2)
y = (screen_height // 2) - (window_height // 2)
root.geometry(f"{window_width}x{window_height}+{x}+{y}")

# Create a frame for datetime selection
datetime_frame = ttk.LabelFrame(root, text="Select Datetime:")
datetime_frame.pack(padx=10, pady=10)

# Create an entry for datetime
datetime_entry = ttk.Entry(datetime_frame)
datetime_entry.pack(side=tk.LEFT, pady=5)

# Load and resize the clock icon image
clock_icon = Image.open("assets/clock.png")
clock_icon = clock_icon.resize((15, 15), Image.LANCZOS)
clock_icon = ImageTk.PhotoImage(clock_icon)

# Create a clock button to update the datetime field
update_button = ttk.Button(datetime_frame, image=clock_icon, command=update_datetime)
update_button.pack(side=tk.LEFT, padx=5, pady=5)

# Create a frame for the road segment selection
segment_frame = ttk.LabelFrame(root, text="Select Road Segment:")
segment_frame.pack(padx=10, pady=10)
# Create a variable to store the selected road segment
selected_road_segment = tk.StringVar()

# Create the dropdown menu for road segment selection with increased width
segment_options = [
    'Maitighar-Putalisadak','New Baneshwor-Maitighar','New Baneshwor-Old Baneshwor',  'Old Baneshwor-Seto Pool', 
    'Rato Pool-Hattisar', 'Seto Pool-Putalisadak', 
    'Seto Pool-Rato Pool', 'Putalisadak-Hattisar', 'All Segments'
]
segment_dropdown = ttk.Combobox(segment_frame, textvariable=selected_road_segment, values=segment_options, state="readonly", width=30)
segment_dropdown.pack(pady=5)

# Create a frame for the prediction mode selection
mode_frame = ttk.LabelFrame(root, text="Select Prediction Mode:")
mode_frame.pack(padx=10, pady=10)
# Create a variable to store the selected mode
selected_mode = tk.StringVar()

mode_options = ['Congestion Index', 'Arrival Time', 'Both']
mode_dropdown = ttk.Combobox(mode_frame, textvariable=selected_mode, values=mode_options, state="readonly", width=20)
mode_dropdown.pack(pady=5)

# Create a button to generate the route
generate_button = ttk.Button(root, text="Predict Traffic", command=lambda: predict_congestion(datetime_entry.get()))
generate_button.pack(pady=10)

# Create a text area to display the generated route
route_text = tk.Text(root, width=75, height=25, state=tk.NORMAL)
route_text.config(state=tk.DISABLED)
route_text.pack(padx=10, pady=10)

# Create a frame for additional buttons
button_frame = tk.Frame(root)
button_frame.pack(pady=10)

optimized_path_button = ttk.Button(button_frame, text="View Optimized Path in Map", command=optimize_path)
optimized_path_button.pack(side=tk.LEFT, padx=5, pady=5)

root.mainloop()
