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

# def predict_congestion(model, date_time):
def predict_congestion(date_time):
    global congestion_index, arrival_time
    # if not model or not date_time:
    if not date_time:
        return 
    # selected_model_value = selected_model.get()
    selected_datetime = datetime_entry.get()
    route_text.config(state=tk.NORMAL)
    route_text.delete(1.0, tk.END)
    # route_text.insert(tk.END, f"Selected model: {selected_model_value}\n")
    route_text.insert(tk.END, f"Selected DateTime: {selected_datetime}\n")
    route_text.insert(tk.END, "Predicting congestion and arrival time for each segment...\n")
    route_text.update_idletasks()
    time.sleep(1)
    # Get the prediction result from utils.predict_traffic()
    prediction_result, congestion_index, arrival_time = utils.predict_traffic(date_time)
    # Display the prediction result in route_text
    route_text.insert(tk.END, f"{prediction_result}\n")
    route_text.config(state=tk.DISABLED)

def display_image(img_path):
    img = Image.open(img_path)
    img = img.resize((400, 300), Image.LANCZOS )
    img_tk = ImageTk.PhotoImage(img)
    route_text.image_create(tk.END, image=img_tk)
    route_text.insert(tk.END, "\n")
    route_text.image = img_tk  # Keep a reference to the image

def optimize_path():
    global congestion_index, arrival_time
    start = 'New Baneshwor'
    end = 'Hattisar'
    if congestion_index is None or arrival_time is None:
        route_text.config(state=tk.NORMAL)
        route_text.insert(tk.END, "Please predict traffic before optimizing the path.\n")
        route_text.config(state=tk.DISABLED)
        return
    graph = utils.create_graph(utils.road_segments, congestion_index, arrival_time)
    optimal_path = utils.ant_colony_optimization(graph, start, end)
    optimized_path_map.show_optimized_path(optimal_path)
    route_text.config(state=tk.NORMAL)
    route_text.insert(tk.END, f"Optimal path from {start} to {end}: {' -> '.join(optimal_path)}\n")
    route_text.config(state=tk.DISABLED)

def update_datetime():
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M")
    datetime_entry.config(state=tk.NORMAL)
    datetime_entry.delete(0, tk.END)
    datetime_entry.insert(0, current_datetime)

root = tk.Tk()
root.title("Traffic Congestion Control")

# Center the window on the screen
window_width = 650
window_height = 750
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = (screen_width // 2) - (window_width // 2)
y = (screen_height // 2) - (window_height // 2)
root.geometry(f"{window_width}x{window_height}+{x}+{y}")

# Create a frame for the model selection
model_frame = ttk.LabelFrame(root, text="Model:")
model_frame.pack(padx=10, pady=10)
# Create a variable to store the selected model
selected_model = tk.StringVar()

# Create the dropdown menu for model selection
# model_options = ["LSTM", "GRU", "RNN"]
# model_dropdown = ttk.Combobox(model_frame, textvariable=selected_model, values=model_options, state="readonly")
# model_dropdown.pack(pady=5)

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

# Create a button to generate the route
generate_button = ttk.Button(root, text="Predict Traffic", command=lambda: predict_congestion(datetime_entry.get()))
generate_button.pack(pady=10)

# Create a text area to display the generated route
route_text = tk.Text(root, width=75, height=30, state=tk.NORMAL)
route_text.config(state=tk.DISABLED)
route_text.pack(padx=10, pady=10)

# Create a frame for additional buttons
button_frame = tk.Frame(root)
button_frame.pack(pady=10)

optimized_path_button = ttk.Button(button_frame, text="View Optimized Path in Map", command=optimize_path)
optimized_path_button.pack(side=tk.LEFT, padx=5, pady=5)

root.mainloop()
