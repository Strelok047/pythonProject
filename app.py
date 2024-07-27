from io import BytesIO
import ee
import streamlit as st
import geemap.foliumap as geemap
import geopandas as gpd
import matplotlib.pyplot as plt
import zipfile
import tempfile
import os


# Настройка страницы Streamlit
def setup():
    st.set_page_config(layout="wide", page_title="Спутниковые Изображения", page_icon='🛰️')
    st.header("🛰️ Спутниковые Изображения")


# Панель навигации для ссылок на страницы
def Navbar():
    with st.sidebar:
        st.page_link('app.py', label='Спутниковые Изображения', icon='🛰️')
        st.page_link('pages/graph.py', label='График', icon='📈')
        st.page_link('pages/about.py', label='О проекте', icon='📖')


# Конфигурация наборов данных спутников
datasets = {
    'Sentinel-2': {
        'collection': 'COPERNICUS/S2_SR_HARMONIZED',
        'bands': ['B4', 'B3', 'B2', 'B8', 'B5'],  # bands: 0-Красный, 1-Синий, 2-Зеленый, 3-NIR, 4-Красная граница
        'year_range': [2019, 2023]
    },
    # Другие спутники...
}

# Формулы индексов растительности
indexes = {
    "NDVI": "(NIR - RED) / (NIR + RED)",
    "EVI": "2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))",
    # Другие индексы...
}


# Функция для маскировки облаков на изображениях
def mask_clouds(image, dataset):
    cloud_mask_band = datasets[dataset]['cloud_mask_band']
    cloud_mask_value = datasets[dataset]['cloud_mask_value']
    cloud_mask = image.select(cloud_mask_band).bitwiseAnd(cloud_mask_value).eq(0)
    return image.updateMask(cloud_mask)


# Функция для получения отфильтрованных изображений
def get_filtered_images(satellite, year, region):
    dataset = datasets[satellite]
    collection = ee.ImageCollection(dataset['collection'])
    filtered_images = collection.filterBounds(region).filterDate(f'{year}-01-01', f'{year}-12-31')

    if satellite == 'Sentinel-2':
        return filtered_images.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
            .filter(ee.Filter.lt('SNOW_ICE_PERCENTAGE', 20))
    else:
        return filtered_images.map(lambda image: mask_clouds(image, satellite))


# Функция для добавления RGB-слоя на карту
def add_rgb_layer_to_map(m, satellite, year, region, brightness, clip, gamma):
    filtered_images = get_filtered_images(satellite, year, region)
    median_image = filtered_images.median()

    if clip:
        median_image = median_image.clip(region)

    rgb_bands = [datasets[satellite]['bands'][i] for i in range(0, 3)]
    vis_params = {
        'bands': rgb_bands,
        'min': 0,
        'max': int(brightness) * 1000,
        'gamma': gamma
    }
    m.addLayer(median_image, vis_params, f'{satellite} {year} RGB')
    m.centerObject(region, 10)


# Функция для вычисления индекса
def calc_index(satellite, index_name, year, region, clip):
    filtered_images = get_filtered_images(satellite, year, region)
    image = filtered_images.median()

    red_band = datasets[satellite]['bands'][0]
    blue_band = datasets[satellite]['bands'][1]
    green_band = datasets[satellite]['bands'][2]
    nir_band = datasets[satellite]['bands'][3]
    red_edge_band = datasets[satellite]['bands'][4]

    if clip:
        image = image.clip(region)

    index = image.expression(indexes[index_name], {
        'RED': image.select(red_band),
        'BLUE': image.select(blue_band),
        'GREEN': image.select(green_band),
        'NIR': image.select(nir_band),
        'RED_EDGE': image.select(red_edge_band),
        'L': 0.5
    }).rename(index_name)

    stats = index.reduceRegion(
        reducer=ee.Reducer.mean().combine(
            reducer2=ee.Reducer.minMax(), sharedInputs=True
        ).combine(
            reducer2=ee.Reducer.stdDev(), sharedInputs=True
        ),
        geometry=region,
        scale=30,
        bestEffort=True
    ).getInfo()

    return index, stats


# Основная функция для управления приложением Streamlit
def main():
    setup()
    Navbar()

    row0_col1, row0_col2, row0_col3, row0_col4, row0_col5 = st.columns([1, 1, 1, 1, 1])
    row1_col1, row1_col2 = st.columns([5, 1])
    row2_col1, row2_col2, row2_col3 = st.columns([1, 1, 1])

    Map = geemap.Map()
    roi = None
    coordinates = None

    # Панель боковая для пользовательского ввода
    st.sidebar.markdown("""---""")
    st.sidebar.markdown("<h5 style='text-align: center; color: grey;'>Укажите точку интереса</h5>",
                        unsafe_allow_html=True)
    sidebar_col1, sidebar_col2 = st.sidebar.columns([1, 1])
    with sidebar_col1:
        long = st.number_input('Долгота', value=0.0)
    with sidebar_col2:
        lat = st.number_input('Широта', value=0.0)

    if long != 0 and lat != 0:
        coordinates = ee.Geometry.Point([long, lat])

    st.sidebar.markdown("<h3 style='text-align: center; color: grey;'>ИЛИ</h3>", unsafe_allow_html=True)

    # Загрузка и обработка шейп-файла
    uploaded_shp_file = st.sidebar.file_uploader("Загрузите архивированный шейп-файл", type=["zip"])

    if uploaded_shp_file is not None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(uploaded_shp_file, 'r') as zip_ref:
                zip_ref.extractall(tmpdir)

            shapefile_path = None
            for root, dirs, files in os.walk(tmpdir):
                for file in files:
                    if file.endswith(".shp"):
                        shapefile_path = os.path.join(root, file)
                        break

            if shapefile_path:
                gdf = gpd.read_file(shapefile_path)
                fig, ax = plt.subplots()
                gdf.plot(ax=ax)
                plt.xticks(rotation=90, fontsize=7)
                plt.yticks(fontsize=7)
                buf = BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                with row2_col1:
                    st.image(buf, caption='Plot Geopandas')
                if not gdf.empty:
                    roi = geemap.geopandas_to_ee(gdf)
            else:
                st.error("Шейп-файл (.shp) не найден в загруженном архиве.")

    # Ввод пользователя для выбора спутника, года и параметров визуализации
    with row0_col1:
        sat = st.selectbox("Выберите спутник", list(datasets.keys()), index=0)

    with row0_col2:
        selected_year = st.number_input("Выберите год", min_value=datasets[sat]['year_range'][0],
                                        max_value=datasets[sat]['year_range'][1],
                                        value=datasets[sat]['year_range'][0])

    with row0_col3:
        brightness = st.number_input("Установите яркость", value=3)

    with row0_col4:
        gamma = st.number_input("Установите гамму", value=1.4)

    with row0_col5:
        st.markdown("""""")
        st.markdown("""""")
        clip = st.toggle("Обрезать")

    with row1_col2:
        check_index = st.toggle("Добавить индекс")
        if check_index:
            index_name = st.selectbox("Выберите индекс", list(indexes.keys()), index=0)
            main_color = st.color_picker('Основной цвет', value='#00ff00')
            mid_color = st.color_picker('Средний цвет', value='#ffff00')
            secondary_color = st.color_picker("Вторичный цвет", value='#ff0000')

    # Визуализация и статистика
    if coordinates is not None and roi is None:
        Map.centerObject(coordinates, zoom=10)
        add_rgb_layer_to_map(Map, sat, selected_year, coordinates, brightness, None, gamma)

    if selected_year is not None and sat is not None and roi is not None:
        Map.centerObject(roi, zoom=10)
        add_rgb_layer_to_map(Map, sat, selected_year, roi, brightness, clip, gamma)

        if check_index:
            index_image, stats = calc_index(sat, index_name, selected_year, roi, clip)
            Map.addLayer(index_image, {'min': -1, 'max': 1, 'palette': [secondary_color, mid_color, main_color]},
                         f'{index_name}, {sat} {selected_year}')
            with row2_col2:
                fig, ax = plt.subplots()
                labels = ['Min', 'Mean', 'Max', 'Std Dev']
                values = [stats[f"{index_name}_min"], stats[f"{index_name}_mean"], stats[f"{index_name}_max"],
                          stats[f"{index_name}_stdDev"]]
                ax.bar(labels, values, color=[main_color, secondary_color, 'blue', 'green'])
                ax.set_title(f'{index_name} Статистика')
                ax.set_ylabel('Значение')
                ax.set_xlabel('Статистика')
                st.pyplot(fig)
                with row2_col3:
                    st.subheader(f"Статистика {index_name}")
                    st.write("Мин:", stats[f"{index_name}_min"])
                    st.write("Среднее:", stats[f"{index_name}_mean"])
                    st.write("Макс:", stats[f"{index_name}_max"])
                    st.write("Отклонение:", stats[f"{index_name}_stdDev"])

        if gdf is not None:
            Map.add_gdf(gdf, 'polygon')

    with row1_col1:
        Map.to_streamlit(height=600)


if __name__ == "__main__":
    main()
