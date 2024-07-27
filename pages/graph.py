import ee
import pandas as pd
import streamlit as st
import geemap.foliumap as geemap
import geopandas as gpd
import matplotlib.pyplot as plt
import zipfile
import tempfile
import os
from app import Navbar, calc_index, datasets, indexes


def setup():
    st.set_page_config(layout="wide", page_title="Yearly index delta Graph", page_icon='📈')
    st.header("📈Graph")
    return "Initialization done."


def plot_index_over_time(satellite, index_name, start_year, end_year, region, graph_data):
    years = list(range(start_year, end_year + 1))
    index_values_dict = {data: [] for data in graph_data}

    for year in years:
        index_image, stats = calc_index(satellite, index_name, year, region, False)
        for data in graph_data:
            index_values_dict[data].append(stats[f"{index_name}_{data.lower()}"])

    df = pd.DataFrame({'Year': years})
    for data in graph_data:
        df[data] = index_values_dict[data]

    fig, ax = plt.subplots()
    for data in graph_data:
        ax.plot(df['Year'], df[data], marker='o', linestyle='-', label=f'{data} {index_name}')

    ax.set_title(f'{index_name} over Time ({start_year}-{end_year})')
    ax.set_xlabel('Year')
    ax.set_ylabel(f'{index_name} Value')
    ax.grid(True)
    ax.legend()

    return fig, df


def main():
    # Execute setup function
    setup()
    Navbar()
    row0_col1, row0_col2, row0_col3, row0_col4, row0_col5 = st.columns([1, 1, 1, 1, 1])
    row1_col1, row1_col2 = st.columns([1, 1])

    roi = None
    coordinates = None

    st.sidebar.markdown("""---""")
    st.sidebar.markdown("<h5 style='text-align: center; color: grey;'>Set point of interest</h5>",
                        unsafe_allow_html=True)
    sidebar_col1, sidebar_col2 = st.sidebar.columns([1, 1])
    with sidebar_col1:
        long = st.number_input('Longitude', value=0.0)
    with sidebar_col2:
        lat = st.number_input('Latitude', value=0.0)

    if long != 0 and lat != 0:
        coordinates = ee.Geometry.Point([long, lat])

    st.sidebar.markdown("<h3 style='text-align: center; color: grey;'>OR</h3>", unsafe_allow_html=True)

    # Upload a zipped shapefile
    uploaded_shp_file = st.sidebar.file_uploader("Upload a Zipped Shapefile", type=["zip"])

    if uploaded_shp_file is not None:
        # Extract the zip file
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(uploaded_shp_file, 'r') as zip_ref:
                zip_ref.extractall(tmpdir)

            # Find the shapefile within the extracted files
            shapefile_path = None
            for root, dirs, files in os.walk(tmpdir):
                for file in files:
                    if file.endswith(".shp"):
                        shapefile_path = os.path.join(root, file)
                        break

            if shapefile_path:
                # Read the shapefile into a GeoDataFrame
                gdf = gpd.read_file(shapefile_path)
            else:
                st.error("Shapefile (.shp) not found in the uploaded zip file.")

            if not gdf.empty:
                roi = geemap.geopandas_to_ee(gdf)

    with row0_col1:
        sat = st.selectbox("Select a satellite", list(datasets.keys()), index=0)

    with row0_col2:
        start_year = st.number_input("Start year", min_value=datasets[sat]['year_range'][0],
                                     max_value=datasets[sat]['year_range'][1],
                                     value=datasets[sat]['year_range'][0])
    with row0_col3:
        end_year = st.number_input("End year", min_value=datasets[sat]['year_range'][0],
                                   max_value=datasets[sat]['year_range'][1],
                                   value=datasets[sat]['year_range'][1])
    with row0_col4:
        index_name = st.selectbox('Select index', list(indexes.keys()), index=0)

    with row0_col5:
        graph_data = st.multiselect("Data", ["Max", "Mean", "Min"], default=("Max", "Mean", "Min"))

    if coordinates is not None and roi is None:
        region = coordinates
    elif roi is not None:
        region = roi
    else:
        region = None

    if start_year <= end_year:
        if graph_data is not None and region is not None:
            fig, df = plot_index_over_time(sat, index_name, start_year, end_year, region, graph_data)
            with row1_col1:
                st.pyplot(fig)
            with row1_col2:
                df['Year'] = df['Year'].astype(str)
                st.write(df)
    else:
        st.error("End year should be greater ar equal to start year.")


if __name__ == "__main__":
    main()
