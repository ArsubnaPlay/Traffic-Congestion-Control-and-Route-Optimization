import folium
import json
import webbrowser
import os
import utils

def show_optimized_path(optimized_path):
    updated_list = []
    for i in range(len(optimized_path) - 1):
        start = optimized_path[i]
        end = optimized_path[i+1]
        route_key = f"{start}-{end}"
        if route_key not in utils.geojson_coordinate_mapping:
            route_key = f"{end}-{start}"  # Try reverse order if not found
        if route_key in utils.geojson_coordinate_mapping:
            updated_list.append(route_key)
        else:
            print(f"Warning: No coordinates found for route {route_key}")

    list_cordinates_geojson = []

    for route in updated_list:
        if route in utils.geojson_coordinate_mapping:
            coordinates = utils.geojson_coordinate_mapping[route]
            for coord in coordinates:
                list_cordinates_geojson.append([coord[1], coord[0]])
        else:
            print(f"Warning: No coordinates found for route {route}")

    print(f"Optimized path: {optimized_path}")
    print(f"Updated list: {updated_list}")
    print(f"Coordinates: {list_cordinates_geojson}")

    # Ensure the map object is created outside the loop
    mymap = folium.Map(location=[27.7004, 85.3284], tiles="Cartodb Positron", zoom_start=15)
    with open('assets/baneshwor_hattisar.json') as f:
        geojson_data = json.load(f)

    def style_function(feature):
        geom_type = feature.get("geometry", {}).get("type")
        if geom_type == "LineString":
            return {
                'color': 'gray',
                'weight': 3,
                'opacity': 2
            }
        return {}

    folium.GeoJson(geojson_data, name="geojson", style_function=style_function).add_to(mymap)

    origin = [27.6883, 85.3355]
    destination = [27.7105, 85.3222]

    folium.Marker(origin, popup='Origin', icon=folium.Icon(color='green')).add_to(mymap)
    folium.Marker(destination, popup='Destination', icon=folium.Icon(color='red')).add_to(mymap)

    if list_cordinates_geojson:
        folium.PolyLine(
            locations=list_cordinates_geojson,
            color='blue',
            weight=5,
            opacity=0.8
        ).add_to(mymap)
    else:
        print("Warning: No coordinates to draw PolyLine")

    mymap.save('map.html')
    webbrowser.open('file://' + os.path.realpath('map.html'))