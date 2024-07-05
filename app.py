import streamlit as st
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import requests
import json
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
from geopy.geocoders import Nominatim
import os
from dotenv import load_dotenv
import time
import re
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# Load environment variables
load_dotenv()

# Streamlit app
st.title("Gen AI Map Viewer")

# Input for OpenAI API key
api_key = st.text_input("Enter your OpenAI API key:", type="password")

if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

    # Initialize OpenAI LLM
    model_name = st.sidebar.selectbox(
        "Choose a model", ("gpt-3.5-turbo", "gpt-4", "gpt-4-32k")
    )
    llm = OpenAI(temperature=0, model_name=model_name)

    def geocode_area(area_name):
        geolocator = Nominatim(user_agent="my_agent")
        try:
            location = geolocator.geocode(area_name)
            if location:
                return location.latitude, location.longitude
            else:
                return None, None
        except Exception as e:
            return None, None

    def reverse_geocode(lat, lon):
        geolocator = Nominatim(user_agent="my_agent")
        try:
            location = geolocator.reverse(f"{lat}, {lon}")
            return location.address if location else "Unknown location"
        except Exception as e:
            return f"Error in reverse geocoding: {str(e)}"

    def generate_overpass_query(query):
        area_prompt = PromptTemplate(
            input_variables=["query"],
            template="What is the name of the area or city mentioned in this query: {query}? Respond with just the name of the area or city, nothing else.",
        )
        area_chain = LLMChain(llm=llm, prompt=area_prompt)
        area = area_chain.run(query=query).strip()

        # Geocode the area to get lat, lon
        lat, lon = geocode_area(area)
        if lat is None or lon is None:
            st.error(f"Could not geocode the area: {area}")
            return None

        # Extract the number if present in the query
        number_match = re.search(r"\b\d+\b", query)
        number = number_match.group(0) if number_match else ""

        overpass_prompt = PromptTemplate(
            input_variables=["query", "lat", "lon", "number"],
            template="""Generate an Overpass QL query to find {query} around the coordinates ({lat}, {lon}). Use the following format:
    [out:json];
    (
      node(around:15000,{lat},{lon})["key"="value"];
      way(around:15000,{lat},{lon})["key"="value"];
      relation(around:15000,{lat},{lon})["key"="value"];
    );
    out {number};

    Replace "key" and "value" with appropriate tags for the query.""",
        )
        overpass_chain = LLMChain(llm=llm, prompt=overpass_prompt)
        generated_query = overpass_chain.run(
            query=query, lat=lat, lon=lon, number=number
        )

        st.write("Generated Overpass query:", generated_query)
        return generated_query

    def run_overpass_query(query):
        overpass_url = "http://overpass-api.de/api/interpreter"
        try:
            response = requests.get(overpass_url, params={"data": query})
            response.raise_for_status()  # Raise an exception for bad status codes
            return response.json()
        except requests.exceptions.RequestException as e:
            return f"Error making request: {str(e)}"
        except json.JSONDecodeError:
            return f"Error decoding JSON. Raw response: {response.text}"

    # Define tools for the agent
    tools = [
        Tool(
            name="Generate Overpass Query",
            func=generate_overpass_query,
            description="Useful for generating Overpass QL queries based on the user's query",
        ),
        Tool(
            name="Run Overpass Query",
            func=run_overpass_query,
            description="Useful for running Overpass QL queries and getting results",
        ),
        Tool(
            name="Reverse Geocode",
            func=reverse_geocode,
            description="Useful for finding latitude and longitude from the location names",
        ),
    ]

    # Initialize the agent
    agent = initialize_agent(
        tools, llm, agent="zero-shot-react-description", verbose=True
    )

    query = st.text_input("Enter your search query:")

    if st.button("Search"):
        if query:
            generated_query = generate_overpass_query(query)
            if generated_query:
                result = run_overpass_query(generated_query)

                # Check if result is a string (error message)
                if isinstance(result, str):
                    st.error(f"An error occurred: {result}")
                    st.write(
                        "Please try refining your query or try a different search."
                    )
                elif isinstance(result, dict) and "elements" in result:
                    if result["elements"]:
                        st.success(f"Found {len(result['elements'])} results.")

                        # Normalize the result and handle different structures
                        df = pd.json_normalize(result["elements"])

                        # Ensure latitude and longitude are present in the DataFrame
                        df["lat"] = df.apply(
                            lambda row: (
                                row["lat"]
                                if "lat" in row
                                else (row["center"]["lat"] if "center" in row else None)
                            ),
                            axis=1,
                        )
                        df["lon"] = df.apply(
                            lambda row: (
                                row["lon"]
                                if "lon" in row
                                else (row["center"]["lon"] if "center" in row else None)
                            ),
                            axis=1,
                        )

                        # Filter out entries without lat/lon
                        df = df[df[["lat", "lon"]].notnull().all(axis=1)]

                        # Convert to GeoDataFrame
                        gdf = gpd.GeoDataFrame(
                            df, geometry=gpd.points_from_xy(df.lon, df.lat)
                        )

                        # Display the DataFrame without the geometry column
                        st.dataframe(gdf.drop(columns="geometry"))

                        # Create a map centered on the first result
                        first_element = gdf.iloc[0]
                        center_lat, center_lon = (
                            first_element["lat"],
                            first_element["lon"],
                        )

                        m = folium.Map(location=[center_lat, center_lon], zoom_start=13)

                        marker_cluster = MarkerCluster().add_to(m)

                        for _, row in gdf.iterrows():
                            lat, lon = row.geometry.y, row.geometry.x
                            popup_text = f"Type: {row.get('type', 'Unknown')}<br>"
                            popup_text += f"Tags: {row.get('tags', {})}<br>"
                            popup_text += f"Location: {reverse_geocode(lat, lon)}"

                            folium.Marker(location=[lat, lon], popup=popup_text).add_to(
                                marker_cluster
                            )

                        folium_static(m)
                    else:
                        st.warning(
                            "No results found. Try refining your search or try a different query."
                        )
                else:
                    st.warning(
                        "Unexpected result format. Please try a different query."
                    )
        else:
            st.warning("Please enter a search query.")

    # Add suggestions for specific tags
    st.markdown("### Suggested Tags for Queries")
    st.markdown("- Parks: `leisure=park`")
    st.markdown("- Restaurants: `amenity=restaurant`")
    st.markdown("- Cafes: `amenity=cafe`")
    st.markdown("- Hotels: `tourism=hotel`")

    # Add attribution
    st.markdown(
        "Data © OpenStreetMap contributors, ODbL 1.0. https://osm.org/copyright"
    )

    # Add a small delay to avoid rate limiting
    time.sleep(1)
