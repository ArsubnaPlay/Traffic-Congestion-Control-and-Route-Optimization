import utils
from datetime import datetime, timedelta
import pprint

def run_optimizations():
    start_location = 'New Baneshwor'
    end_location = 'Hattisar'
    
    base_date = datetime(2023, 7, 13)
    
    print(f"Optimal paths from {start_location} to {end_location} throughout the day (with biases):")
    print("=" * 80)
    
    for hour in range(24):
        current_time = base_date + timedelta(hours=hour)
        graph = utils.create_graph(utils.road_segments, utils.congestion_index, utils.arrival_time)
        graph = utils.update_traffic_conditions(graph, current_time)
        
        print(f"Time: {current_time.strftime('%H:%M')}")
        print("Current graph state:")
        pprint.pprint(graph)
        
        optimal_path, best_cost, best_time = utils.optimize_for_time(start_location, end_location, current_time)
        
        print(f"Optimal path: {' -> '.join(optimal_path)}")
        print(f"Cost: {best_cost:.2f}, Time: {best_time:.2f} minutes")
        print("-" * 80)

if __name__ == "__main__":
    _, utils.congestion_index, utils.arrival_time = utils.predict_traffic(datetime.now().strftime("%Y-%m-%d %H:%M"), mode="Both")
    run_optimizations()