import folium
import json
import webbrowser
import os
import utils

def show_optimized_path(optimized_path):
    updated_list = [f"{optimized_path[i]}-{optimized_path[i+1]}" for i in range(len(optimized_path)-1)]
    list_cordinates_geojson = []

    for route in updated_list:
        coordinates = utils.geojson_coordinate_mapping[route]
        for coord in coordinates:
            list_cordinates_geojson.append([coord[1], coord[0]])

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
    folium.PolyLine(
        locations=list_cordinates_geojson,
        color='blue',
        weight=5,
        opacity=0.8
    ).add_to(mymap)

    mymap.save('map.html')
    webbrowser.open('file://' + os.path.realpath('map.html'))