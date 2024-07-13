from datetime import datetime, timedelta
import random 
from tabulate import tabulate
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import joblib
import os


travel_time_model_path = os.path.join(os.path.dirname(__file__), 'rnn_travel_time.keras')
average_speed_model_path = os.path.join(os.path.dirname(__file__), 'lstm_average_speed.keras')
travel_time_scaler_X_path = os.path.join(os.path.dirname(__file__), 'travel_time_scaler_X.pkl')
travel_time_scaler_y_path = os.path.join(os.path.dirname(__file__), 'travel_time_scaler_y.pkl')
average_speed_scaler_X_path = os.path.join(os.path.dirname(__file__), 'scaler_X.pkl')
average_speed_scaler_y_path = os.path.join(os.path.dirname(__file__), 'scaler_y.pkl')

tt_scaler_X = joblib.load(travel_time_scaler_X_path)
tt_scaler_y = joblib.load(travel_time_scaler_y_path)

as_scaler_X = joblib.load(average_speed_scaler_X_path)
as_scaler_y = joblib.load(average_speed_scaler_y_path)




def congestion_mapping(congestion):
    if congestion is None:
        return "Unknown"
    if congestion > 40:
        return "Free Flow"
    elif 30 < congestion <= 40:
        return "Light Traffic"
    elif 20 < congestion <= 30:
        return "Moderate Traffic"
    elif 15 < congestion <= 20:
        return "Heavy Traffic"
    else:
        return "Severe Congestion"

    
road_intersections = {
    1: "New Baneshwor",
    2: "Old Baneshwor",
    4: "Seto Pool",
    5: "Rato Pool",
    6: "Maitighar",
    7: "Putalisadak",
    8: "Hattisar"
}

# road_segments =['Maitighar-Putalisadak','New Baneshwor-Maitighar','New Baneshwor-Old Baneshwor','Old Baneshwor-Seto Pool', 'Putalisadak-Hattisar','Rato Pool-Hattisar','Seto Pool-Putalisadak','Seto Pool-Rato Pool']
road_segments = [
    'New Baneshwor-Old Baneshwor',
    'New Baneshwor-Maitighar',
    'Old Baneshwor-Seto Pool',
    'Maitighar-Putalisadak',
    'Seto Pool-Putalisadak',
    'Seto Pool-Rato Pool',
    'Putalisadak-Hattisar',
    'Rato Pool-Hattisar'
]

def get_congestion_index(n):
    return congestion_mapping.get(n)

def added_datetime(base, added_mins):
    if isinstance(base, str):
        try:
            date_object = datetime.strptime(base, "%Y-%m-%d %H:%M")
        except ValueError as e:
            print(f"Error parsing date: {e}")
            return None
    else:
        date_object = base
    new_datetime = date_object + timedelta(minutes=added_mins)
    return new_datetime
    

def individual_traffic_metrics(datetime_input, road_segment, mode):
    road_segment = road_segment
    datetime_obj = pd.to_datetime(datetime_input)
    hour = datetime_obj.hour
    minute = datetime_obj.minute
    is_peakhour = 1 if (8 <= hour <= 11 or 16 <= hour <= 18) else 0
    is_weekend = 1 if datetime_obj.dayofweek == 5 else 0
    features = np.array([[road_segment, hour, minute, is_peakhour, is_weekend]])
    
    if mode == 'congestion_index':
        model = load_model(average_speed_model_path)
        features_scaled = as_scaler_X.transform(features)
        features_scaled = features_scaled.reshape((features_scaled.shape[0], 1, features_scaled.shape[1]))
        predicted_scaled = model.predict(features_scaled)
        result = as_scaler_y.inverse_transform(predicted_scaled)
        congestion_index = round(result[0][0], 2)
        print(f"Congestion Index for segment {road_segment}: {congestion_index}")
        return congestion_index
    
    elif mode == 'arrival_time':
        features_scaled = tt_scaler_X.transform(features)
        features_scaled = features_scaled.reshape((features_scaled.shape[0], 1, features_scaled.shape[1]))
        model = load_model(travel_time_model_path)
        predicted_scaled = model.predict(features_scaled)
        result = tt_scaler_y.inverse_transform(predicted_scaled)
        arrival_time = round(result[0][0], 2)
        print(f"Arrival Time for segment {road_segment}: {arrival_time}")
        return arrival_time

    print(f"Unknown mode: {mode}")
    return None


def predict_traffic(datetime_input, mode):
    data = []
    headers = ["Road Segment", "Congestion Index", "Arrival Time(min)"]
    congestion_index = []
    arrival_times = []

    if mode == "Congestion Index":
        headers = ["Road Segment", "Congestion Index"]
        for road_segment in road_segments:
            road_segment_id = road_segments.index(road_segment)
            congestion = individual_traffic_metrics(datetime_input, road_segment_id, 'congestion_index')
            print(f"Predicted Congestion for {road_segment}: {congestion}")
            congestion_index.append(congestion)
            row = [road_segment, congestion_mapping(congestion)]
            data.append(row)
        predicted_val = tabulate(data, headers, tablefmt="grid")
        return predicted_val, congestion_index, []

    elif mode == "Arrival Time":
        headers = ["Road Segment", "Arrival Time(min)"]
        for road_segment in road_segments:
            road_segment_id = road_segments.index(road_segment)
            arrival_time = individual_traffic_metrics(datetime_input, road_segment_id, 'arrival_time')
            print(f"Predicted Arrival Time for {road_segment}: {arrival_time}")
            arrival_times.append(arrival_time)
            row = [road_segment, arrival_time]
            data.append(row)
        predicted_val = tabulate(data, headers, tablefmt="grid")
        return predicted_val, [], arrival_times

    elif mode == "Both":
        for road_segment in road_segments:
            road_segment_id = road_segments.index(road_segment)
            congestion = individual_traffic_metrics(datetime_input, road_segment_id, 'congestion_index')
            arrival_time = individual_traffic_metrics(datetime_input, road_segment_id, 'arrival_time')
            print(f"Predicted Congestion for {road_segment}: {congestion}, Arrival Time: {arrival_time}")
            congestion_index.append(congestion)
            arrival_times.append(arrival_time)
            row = [road_segment, congestion_mapping(congestion), arrival_time]
            data.append(row)
        predicted_val = tabulate(data, headers, tablefmt="grid")
        return predicted_val, congestion_index, arrival_times


# def create_graph(road_segments, congestion_index, arrival_time):
#     graph = {}
#     for i, segment in enumerate(road_segments):
#         start, end = segment.split('-')
#         graph[start] = graph.get(start, {})
#         graph[end] = graph.get(end, {})
#         graph[start][end] = (congestion_index[i], arrival_time[i])
#         graph[end][start] = (congestion_index[i], arrival_time[i])
#     print(graph)
#     return graph
def create_graph(road_segments, congestion_index, arrival_time):
    graph = {}
    for i, segment in enumerate(road_segments):
        start, end = segment.split('-')
        if start not in graph:
            graph[start] = {}
        if end not in graph:
            graph[end] = {}
        graph[start][end] = (congestion_index[i], arrival_time[i])
        graph[end][start] = (congestion_index[i], arrival_time[i])  # Make sure it's bidirectional
    return graph


# def ant_colony_optimization(graph, start, end, num_ants=10, num_iterations=100, alpha=1, beta=2, rho=0.5):
#     best_path = None
#     best_cost = float('inf')
#     best_time = float('inf')

#     for _ in range(num_iterations):
#         pheromone_matrix = initialize_pheromone_matrix(graph)
#         for _ in range(num_ants):
#             path, cost, arrival_time = construct_solution(graph, start, end, pheromone_matrix, alpha, beta)
#             if cost < best_cost:
#                 best_path = path
#                 best_cost = cost
#                 best_time = arrival_time
#         update_pheromone(graph, pheromone_matrix, best_path, best_cost, rho)

#     return best_path, best_cost, best_time
def ant_colony_optimization(graph, start, end, num_ants=10, num_iterations=100, alpha=1, beta=2, rho=0.5):
    best_path = None
    best_cost = float('inf')
    best_time = float('inf')

    for iteration in range(num_iterations):
        pheromone_matrix = initialize_pheromone_matrix(graph)
        for ant in range(num_ants):
            path, cost, arrival_time = construct_solution(graph, start, end, pheromone_matrix, alpha, beta)
            if path and (cost < best_cost or (cost == best_cost and random.random() < 0.1)):
                best_path = path
                best_cost = cost
                best_time = arrival_time
                print(f"Iteration {iteration}, Ant {ant}: New best path found: {' -> '.join(best_path)}")
        update_pheromone(graph, pheromone_matrix, best_path, best_cost, rho)

    return best_path, best_cost, best_time


def initialize_pheromone_matrix(graph):
    pheromone_matrix = {}
    for node in graph:
        pheromone_matrix[node] = {}
        for neighbor in graph[node]:
            pheromone_matrix[node][neighbor] = 1.0
    return pheromone_matrix

import random

def construct_solution(graph, start, end, pheromone_matrix, alpha, beta):
    path = [start]
    current_node = start
    total_cost = 0
    total_arrival_time = 0

    while current_node != end:
        neighbors = list(graph[current_node].keys())
        probabilities = []
        for neighbor in neighbors:
            if neighbor not in path:
                congestion_index, arrival_time = calculate_dynamic_congestion(graph, path + [neighbor])
                pheromone = pheromone_matrix[current_node][neighbor]
                probability = (pheromone ** alpha) * ((1 / (congestion_index * arrival_time)) ** beta)
                probabilities.append((neighbor, probability))

        if not probabilities:
            break

        total_probability = sum(p[1] for p in probabilities)
        probabilities = [(neighbor, probability / total_probability) for neighbor, probability in probabilities]
        next_node = random.choices([neighbor for neighbor, _ in probabilities], weights=[probability for _, probability in probabilities])[0]
        path.append(next_node)
        current_node = next_node

    if current_node != end:
        return None, float('inf'), float('inf')

    total_cost, total_arrival_time = calculate_dynamic_congestion(graph, path)
    return path, total_cost, total_arrival_time

def calculate_dynamic_congestion(graph, path):
    total_cost = 0
    total_arrival_time = 0

    for i in range(len(path) - 1):
        start = path[i]
        end = path[i + 1]
        congestion_index, arrival_time = graph[start][end]
        total_cost += congestion_index
        total_arrival_time += arrival_time

    return total_cost, total_arrival_time


def update_pheromone(graph, pheromone_matrix, path, cost, rho):
    for node in pheromone_matrix:
        for neighbor in pheromone_matrix[node]:
            pheromone_matrix[node][neighbor] *= (1 - rho)

    for i in range(len(path) - 1):
        start = path[i]
        end = path[i + 1]
        pheromone_matrix[start][end] += 1 / cost
        pheromone_matrix[end][start] += 1 / cost


geojson_coordinate_mapping = {
    'New Baneshwor-Old Baneshwor': [
        [
            85.33548725201746,
            27.688290306054128
        ],
        [
            85.33601524185809,
            27.690958803679862
        ],
        [
            85.33620625635558,
            27.691485740050425
        ],
        [
            85.33621360320353,
            27.692487561011546
        ],
        [
            85.33691890313452,
            27.69350365753492
        ],
        [
            85.33713196175682,
            27.694212726448583
        ],
        [
            85.33736706197578,
            27.69535112950264
        ],
        [
            85.3376242016933,
            27.695552788359763
        ],
        [
            85.33752869320398,
            27.696131741185013
        ],
        [
            85.33775644552412,
            27.697009923978115
        ],
        [
            85.33838092814784,
            27.69935171134368
        ],
        [
            85.33882173911292,
            27.700121454500575
        ],
        [
            85.33929928430433,
            27.700492230098746
        ],
        [
            85.33965193306017,
            27.70059630723287
        ],
        [
            85.34010743770216,
            27.70151348268817
        ]
    ],
    'New Baneshwor-Maitighar': [
          [
            85.33546840102713,
            27.68825303293741
          ],
          [
            85.32994499359904,
            27.689531650640248
          ],
          [
            85.32820117097333,
            27.69039802916035
          ],
                    [
            85.32817605963487,
            27.690418698210223
          ],
          [
            85.32046667028033,
            27.694256327145084
          ]
    ],
    'Maitighar-Putalisadak': [
        [
            85.32058564098173,
            27.694263018318182
        ],
        [
            85.32220165144992,
            27.701061356807344
        ],
        [
            85.32286933517292,
            27.705670924023735
        ]
    ],
    'Old Baneshwor-Seto Pool': [
        [
            85.34009248389657,
            27.701525264076125
        ],
        [
            85.33608643385861,
            27.70285796086995
        ]
    ],
    'Rato Pool-Hattisar': [
        [
        85.33676549298349,
        27.708081830890393
        ],
        [
        85.3331676657429,
        27.708220232052042
        ],
        [
        85.32615441981619,
        27.709927004933476
        ],
        [
        85.3222986804522,
        27.71033294112374
        ]
    ],
    'Seto Pool-Putalisadak': [
        [
            85.33599447512097,
            27.702907176487585
        ],
        [
            85.32754831688123,
            27.70537983242373
        ],
        [
            85.32677875519704,
            27.70534061909808
        ],
        [
            85.32285896790512,
            27.70564942557712
        ]
    ],
    'Seto Pool-Rato Pool': [
        [
            85.33604553801223,
            27.702886807699088
        ],
        [
            85.33626689310626,
            27.70539014058886
        ],
        [
            85.33651525004149,
            27.70595555199948
        ],
        [
            85.33681091305823,
            27.708091524216343
        ]
    ],
    'Putalisadak-Hattisar': [
        [
            85.32284365785637,
            27.70565420050346
        ],
        [
            85.32281857698763,
            27.70678066140451
        ],
        [
            85.32180189798379,
            27.70800479757233
        ],
        [
            85.32216970073631,
            27.71038715925161
        ]
    ]
}



# import math
# from datetime import datetime,timedelta

# def update_traffic_conditions(graph, time):
#     for start in graph:
#         for end in graph[start]:
#             # Simulate traffic variation based on time (assuming 24-hour cycle)
#             hour = time.hour
#             factor = 1 + 0.5 * math.sin(hour * math.pi / 12)  # Traffic peaks at 6AM and 6PM
#             congestion, arrival = graph[start][end]
#             graph[start][end] = (
#                 congestion * factor,  # Adjust congestion index
#                 arrival * factor   # Adjust arrival time
#             )
#     return graph

def optimize_for_time(start_location, end_location, time):
    global congestion_index, arrival_time
    graph = create_graph(road_segments, congestion_index, arrival_time)
    graph = update_traffic_conditions(graph, time)
    optimal_path, best_cost, best_time = ant_colony_optimization(graph, start_location, end_location)
    return optimal_path, best_cost, best_time

import math
from datetime import datetime
def update_traffic_conditions(graph, time):
    biases = {
        ('New Baneshwor', 'Maitighar'): 1.5,
        ('New Baneshwor', 'Old Baneshwor'): 0.1,  # Slightly less congested
        ('Old Baneshwor', 'Seto Pool'): 0.2,
        ('Maitighar', 'Putalisadak'): 1.8,
        ('Seto Pool', 'Putalisadak'): 0.5,
        ('Seto Pool', 'Rato Pool'): 1.3,
        ('Putalisadak', 'Hattisar'): 1,
        ('Rato Pool', 'Hattisar'): 1.2
    }

    for start in graph:
        for end in graph[start]:
            hour = time.hour
            time_factor = 1 + 0.5 * math.sin(hour * math.pi / 12)
            
            bias = biases.get((start, end), biases.get((end, start), 1.0))
            
            congestion, arrival = graph[start][end]
            factor = time_factor * bias
            
            graph[start][end] = (
                congestion * factor,
                arrival * factor
            )
    return graph
