from datetime import datetime, timedelta
import random 
from tabulate import tabulate
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


road_intersections = {
    1: "New Baneshwor",
    2: "Old Baneshwor",
    3: "Bijulibazar",
    4: "Seto Pool",
    5: "Rato Pool",
    6: "Maitighar",
    7: "Putalisadak",
    8: "Hattisar"
}

road_segments =['New Baneshwor-Old Baneshwor', 'New Baneshwor-Bijulibazar','Bijulibazar-Maitighar', 'Maitighar-Putalisadak', 'Old Baneshwor-Seto Pool', 'Rato Pool-Hattisar','Bijulibazar-Seto Pool','Seto Pool-Putalisadak','Seto Pool-Rato Pool','Putalisadak-Hattisar']

congestion_mapping={
    1:"Free",
    2:"Basically Free",
    3:"Mild",
    4:"Moderate",
    5:"Heavy"
}
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

def predict_congestion(*args):
    return random.randrange(1,6)

def predict_arrival_time(*args):
    return random.randrange(5,21)


def predict_traffic():
    data = []
    headers = ["Road Segment", "Congestion Index", "Arrival Time(min)"]
    
    congestion_index = []
    arrival_times = []

    for road_segment in road_segments:
        congestion = predict_congestion(road_segment)
        arrival_time = predict_arrival_time(road_segment)
        
        congestion_index.append(congestion)
        arrival_times.append(arrival_time)

        row = [road_segment, congestion_mapping.get(congestion), arrival_time]
        data.append(row)

    predicted_val = tabulate(data, headers, tablefmt="grid")
    return predicted_val, congestion_index, arrival_times




def create_graph(road_segments, congestion_index, arrival_time):
    graph = {}
    for i, segment in enumerate(road_segments):
        start, end = segment.split('-')
        graph[start] = graph.get(start, {})
        graph[end] = graph.get(end, {})
        graph[start][end] = (congestion_index[i], arrival_time[i])
        graph[end][start] = (congestion_index[i], arrival_time[i])
    print(graph)
    return graph

def ant_colony_optimization(graph, start, end, num_ants=10, num_iterations=100, alpha=1, beta=2, rho=0.5):
    best_path = None
    best_cost = float('inf')

    for _ in range(num_iterations):
        pheromone_matrix = initialize_pheromone_matrix(graph)
        for _ in range(num_ants):
            path, cost = construct_solution(graph, start, end, pheromone_matrix, alpha, beta)
            if cost < best_cost:
                best_path = path
                best_cost = cost
        update_pheromone(graph, pheromone_matrix, best_path, best_cost, rho)

    return best_path

def initialize_pheromone_matrix(graph):
    pheromone_matrix = {}
    for node in graph:
        pheromone_matrix[node] = {}
        for neighbor in graph[node]:
            pheromone_matrix[node][neighbor] = 1.0
    return pheromone_matrix

def construct_solution(graph, start, end, pheromone_matrix, alpha, beta):
    path = [start]
    current_node = start
    cost = 0

    while current_node != end:
        neighbors = list(graph[current_node].keys())
        probabilities = []
        for neighbor in neighbors:
            if neighbor not in path:
                congestion_index, arrival_time = graph[current_node][neighbor]
                pheromone = pheromone_matrix[current_node][neighbor]
                probability = (pheromone ** alpha) * ((1 / congestion_index) ** beta)
                probabilities.append((neighbor, probability))

        if not probabilities:
            break

        total_probability = sum(p[1] for p in probabilities)
        probabilities = [(neighbor, probability / total_probability) for neighbor, probability in probabilities]
        next_node = random.choices([neighbor for neighbor, _ in probabilities], weights=[probability for _, probability in probabilities])[0]
        path.append(next_node)
        congestion_index, arrival_time = graph[current_node][next_node]
        cost += congestion_index * arrival_time
        current_node = next_node

    if current_node != end:
        return None, float('inf')

    return path, cost

def update_pheromone(graph, pheromone_matrix, best_path, best_cost, rho):
    for node in pheromone_matrix:
        for neighbor in pheromone_matrix[node]:
            pheromone_matrix[node][neighbor] *= (1 - rho)

    for i in range(len(best_path) - 1):
        start = best_path[i]
        end = best_path[i + 1]
        pheromone_matrix[start][end] += 1 / best_cost
        pheromone_matrix[end][start] += 1 / best_cost

def predict_traffic_metrics():
    congestion_index = [random.randint(1, 5) for _ in range(len(road_segments))]
    arrival_time = [random.randint(5, 20) for _ in range(len(road_segments))]
    return congestion_index, arrival_time



def plot_congestion_eda(root):
    fig = plt.Figure(figsize=(6, 4), dpi=100)
    ax = fig.add_subplot(111)
    congestion_index = [random.randint(1, 5) for _ in range(len(road_segments))]
    arrival_time = [random.randint(5, 20) for _ in range(len(road_segments))]

    ax.plot(road_segments, congestion_index, label='Congestion Index')
    ax.plot(road_segments, arrival_time, label='Arrival Time')
    ax.set_xlabel('Road Segment')
    ax.set_ylabel('Value')
    ax.set_title('Congestion Index and Arrival Time')
    ax.legend()

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack()

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
    'New Baneshwor-Bijulibazar': [
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
        ]
    ],
    'Bijulibazar-Maitighar': [
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
    'Bijulibazar-Seto Pool': [
        [
            85.3281532702743,
            27.69046225065216
        ],
        [
            85.3297219574963,
            27.69411901567564
        ],
        [
            85.32981799957054,
            27.694359961007436
        ],
        [
            85.32893761388624,
            27.69515365951534
        ],
        [
            85.3297539715204,
            27.695663891223077
        ],
        [
            85.33065036421868,
            27.695904833144866
        ],
        [
            85.33185089015183,
            27.69757723888472
        ],
        [
            85.33035651745769,
            27.6988578423139
        ],
        [
            85.33093276990519,
            27.699807406076175
        ],
        [
            85.33356484331324,
            27.699969436161894
        ],
        [
            85.33529360065967,
            27.701386678036187
        ],
        [
            85.33601391621954,
            27.702888934326182
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

