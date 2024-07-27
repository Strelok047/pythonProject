import streamlit as st
from app import Navbar


def setup():
    st.set_page_config(layout="wide", page_title="Yearly index delta Graph", page_icon='📖')
    st.header("📖About")
    return "Initialization done."

def main():
    # Execute setup function
    setup()
    Navbar()

    markdown_text = """
    # Satellite Imagery Analysis and Yearly Index Delta Graph

    Welcome to our interactive tool designed for satellite imagery analysis, developed as part of a student internship project. This application integrates Google Earth Engine and Streamlit to enable dynamic exploration and visualization of satellite datasets such as Sentinel-2, Landsat series, and MODIS.

    ## Key Features:
    - **Satellite Imagery Analysis:**
      - Choose from various satellite datasets and visualize RGB composites.
      - Mask clouds for enhanced image quality using advanced processing techniques.
      - Calculate and visualize vegetation indices like NDVI, EVI, and others.

    - **Yearly Index Delta Graph:**
      - Explore temporal trends in vegetation indices over a defined region of interest.
      - Select start and end years to analyze changes in index values (Max, Mean, Min).
      - Upload zipped shapefiles to define custom regions or specify coordinates for analysis.

    - **Interactive Visualization and Data Presentation:**
      - Plot dynamic graphs and view detailed numerical data to understand index variations.
      - Customize brightness, gamma correction, and clipping options for RGB visualization.
      - Integration with Geopandas allows seamless interaction with geospatial data.

    This tool empowers users to harness satellite imagery for applications in environmental monitoring, agriculture, and urban planning, providing valuable insights into landscape dynamics over time.
    """

    st.markdown(markdown_text)
    st.header("Authors:")
    st.write("Baigabulov Adil")
    st.write("Bekenov Bekzhan")
    st.write("Ospanov Akhmet")
    st.markdown("<p style='text-align: left'>\
                Group 05-057-21-05 \
                <br>The Department of Computer Science \
                <br>Kazakh Agro-Technical Research University\
                <br>Astana\
                <br>Kazakhstan</p>",
                unsafe_allow_html=True)
if __name__ == "__main__":
    main()
