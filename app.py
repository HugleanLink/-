import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import requests
from sklearn.cluster import KMeans
from math import radians, sin, cos, sqrt, atan2, asin, degrees
import io

# streamlit页面设置
st.set_page_config(page_title="选址", layout="wide")
st.title("起降站选址系统")
st.write("请输入城市名称和高德API Key，然后点击“开始选址分析”。")
SPECIAL_GA_CITIES = ["西宁市", "拉萨市", "昆明市"]
algo_choice = st.selectbox("选择选址算法（若不选择，自动决定）",["KMeans聚类算法", "遗传算法", "不选择", "景区建站算法"])
city = st.text_input("城市名称（例如：武汉市）")
api_key = st.text_input("输入高德API Key", type="password")
with st.expander("高级配置"):
    target_radius_km = st.text_input("指定中心繁华区半径", "8")
    num_clusters = st.text_input("中心繁华区个数", "1")
    num_primary_stations_per_circle = st.text_input("负责繁华区的一级站个数", "5")
    drone_range_km = st.text_input("无人机续航(千米)", "12")
    preset_filter_radius_km = st.text_input("超过城市中心坐标多少公里不纳入考虑", "30")
    outer_buffer_km = st.text_input("二级站的覆盖环带宽度(千米)", "20")
    secondary_radius_km = st.text_input("二级站的最远辐射距离(千米)", "4")
if st.button("开始选址分析"):
    if city.strip() == "":
        st.warning("请先输入城市名称。")
        st.stop()
    if algo_choice == "不选择":
        if any(c in city for c in SPECIAL_GA_CITIES):
            st.session_state["algo"] = "遗传算法"
            st.info(f"已为 {city} 自动选择遗传算法")
        else:
            st.session_state["algo"] = "KMeans聚类算法"
            st.info(f"已为 {city} 自动选择KMeans聚类算法")
    else:
        st.session_state["algo"] = algo_choice
    st.session_state["city"] = city
    st.session_state["api_key"] = api_key
    st.session_state["run_analysis"] = True
if "run_analysis" not in st.session_state or not st.session_state["run_analysis"]:
    st.stop()
if st.session_state["algo"] == "遗传算法":
    st.write("正在运行遗传算法…")
    import JonnyVan as ga
    ga_map, ga_info = ga.run_ga(st.session_state["city"], st.session_state["api_key"])
    st_folium(ga_map, width=900, height=600,returned_objects=[])
    with st.expander("算法信息"):
        st.json(ga_info)
    st.stop()
if st.session_state["algo"] == "景区建站算法":
    import ScenicPlanner as sp
    scenic_map, scenic_info = sp.run_scenic(city, api_key,)
    st_folium(scenic_map, width=900, height=600,returned_objects=[])
    with st.expander("景区选址信息"):
        st.json(scenic_info)
    st.stop()
if st.session_state["algo"] == "KMeans聚类算法":
    target_radius_km = float(target_radius_km)
    num_clusters = int(num_clusters)
    num_primary_stations_per_circle = int(num_primary_stations_per_circle)
    drone_range_km = float(drone_range_km)
    preset_filter_radius_km = float(preset_filter_radius_km)
    outer_buffer_km = float(outer_buffer_km)
    secondary_radius_km = float(secondary_radius_km)


    # 参数
    keywords = '商场,购物中心,餐饮服务,中餐厅,西餐厅,咖啡厅,甜品店,酒店,宾馆,酒吧,KTV,电影院,超市,便利店,写字楼,办公楼,地铁站,公交站'
    weights = {
        '餐饮服务': 1.0, '中餐厅': 1.0, '西餐厅': 1.0, '咖啡厅': 0.9, '甜品店': 0.9,
        '商场': 0.9, '购物中心': 0.9, '酒店': 0.7, '宾馆': 0.7,
        '酒吧': 0.8, 'KTV': 0.8, '电影院': 0.8,
        '超市': 0.8, '便利店': 0.7,
        '写字楼': 0.6, '办公楼': 0.6,
        '地铁站': 0.4, '公交站': 0.4
    }
    max_pages = 40
    num_secondary_stations = 6
    ring_buffer_km = 1.0


    # 工具函数
    def get_city_center(city, api_key):
        try:
            url = "https://restapi.amap.com/v3/geocode/geo"
            params = {'address': city, 'key': api_key}
            data = requests.get(url, params=params).json()
            if data['status'] == '1' and data['geocodes']:
                lng, lat = map(float, data['geocodes'][0]['location'].split(','))
                return lat, lng
            return None, None
        except:
            return None, None


    def haversine(lat1, lon1, lat2, lon2):
        R = 6371
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
        return 2 * R * atan2(sqrt(a), sqrt(1 - a))


    def get_destination_point(lat, lng, distance_km, bearing_deg):
        R = 6371
        lat1, lng1 = radians(lat), radians(lng)
        b = radians(bearing_deg)
        lat2 = asin(sin(lat1) * cos(distance_km / R) + cos(lat1) * sin(distance_km / R) * cos(b))
        lng2 = lng1 + atan2(sin(b) * sin(distance_km / R) * cos(lat1),
                            cos(distance_km / R) - sin(lat1) * sin(lat2))
        return degrees(lat2), degrees(lng2)


    def get_pois(city, keyword, api_key, page_size=20, max_pages=40):
        pois = []
        for page in range(1, max_pages + 1):
            try:
                url = "https://restapi.amap.com/v3/place/text"
                params = {
                    'keywords': keyword, 'city': city, 'output': 'json',
                    'offset': page_size, 'page': page, 'key': api_key
                }
                data = requests.get(url, params=params, timeout=10).json()
                if data['status'] != '1' or not data.get('pois'):
                    break
                for poi in data['pois']:
                    if poi.get("location"):
                        lng, lat = map(float, poi["location"].split(","))
                        pois.append({"lat": lat, "lng": lng, "name": poi["name"]})
            except:
                break
        return pd.DataFrame(pois)


    # 获取城市中心
    st.write("正在获取城市中心…")
    with st.spinner("请求高德 API 中…"):
        preset_center_lat, preset_center_lng = get_city_center(city, api_key)
    if preset_center_lat is None:
        st.error("无法获取城市中心，请检查城市名称或 API Key")
        st.stop()
    st.success(f"城市中心：({preset_center_lat:.5f}, {preset_center_lng:.5f})")
    # 获取 POI
    st.write("正在获取 POI 数据…")
    keyword_list = [k.strip() for k in keywords.split(",")]
    all_pois = pd.DataFrame()
    for kw in keyword_list:
        st.write(f"获取 `{kw}` 中…")
        df = get_pois(city, kw, api_key, max_pages=max_pages)
        if not df.empty:
            df["category"] = kw
            all_pois = pd.concat([all_pois, df])
    if all_pois.empty:
        st.error("没有获取到任何 POI，请检查 API Key。")
        st.stop()
    all_pois.drop_duplicates(subset=["lat", "lng", "name"], inplace=True)
    all_pois["weight"] = all_pois["category"].map(weights).fillna(0.5)
    # 过滤郊区
    d = [
        haversine(preset_center_lat, preset_center_lng, r["lat"], r["lng"])
        for _, r in all_pois.iterrows()
    ]
    all_pois = all_pois[np.array(d) <= preset_filter_radius_km]
    st.success(f"有效 POI 数量：{len(all_pois)}")
    # 聚类
    coords = all_pois[['lat', 'lng']].values
    weights_array = all_pois['weight'].values
    weighted_coords = np.repeat(coords, (weights_array * 10).astype(int) + 1, axis=0)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(weighted_coords)
    cluster_centers = kmeans.cluster_centers_
    circles = []
    primary_stations = []
    secondary_stations = []
    for i, (center_lat, center_lng) in enumerate(cluster_centers):
        distances = [haversine(center_lat, center_lng, r["lat"], r["lng"]) for _, r in
                     all_pois.iterrows()]
        valid = all_pois[np.array(distances) <= target_radius_km]
        if len(valid) == 0:
            continue
        radius = min(max([d for d in distances if d <= target_radius_km]), target_radius_km)
        circles.append({
            "center_lat": center_lat,
            "center_lng": center_lng,
            "radius_km": radius,
            "poi_count": len(valid)
        })
        ring_min = radius - ring_buffer_km
        ring_max = radius + ring_buffer_km
        ring_pois = all_pois[(np.array(distances) >= ring_min) & (np.array(distances) <= ring_max)]
        # 一级站
        if len(ring_pois) < num_primary_stations_per_circle:
            angle_step = 360 / num_primary_stations_per_circle
            for j in range(num_primary_stations_per_circle):
                angle = j * angle_step
                lat, lng = get_destination_point(center_lat, center_lng, radius, angle)
                primary_stations.append({
                    "id": f"P{i + 1}_{j + 1}",
                    "lat": lat,
                    "lng": lng,
                    "circle_id": i + 1
                })
        else:
            ring_coords = ring_pois[['lat', 'lng']].values
            ring_weights = ring_pois['weight'].values
            weighted_ring = np.repeat(ring_coords, (ring_weights * 10).astype(int) + 1, axis=0)
            km = KMeans(n_clusters=num_primary_stations_per_circle, random_state=42, n_init=10)
            centers = km.fit(weighted_ring).cluster_centers_
            for j, (lat, lng) in enumerate(centers):
                primary_stations.append({
                    "id": f"P{i + 1}_{j + 1}",
                    "lat": lat,
                    "lng": lng,
                    "circle_id": i + 1
                })
        # 二级站
        outer_min = radius
        outer_max = radius + outer_buffer_km
        outer_pois = all_pois[
            (np.array(distances) > outer_min) & (np.array(distances) <= outer_max)]
        need = num_primary_stations_per_circle * num_secondary_stations
        if len(outer_pois) < need:
            for pri in primary_stations[-num_primary_stations_per_circle:]:
                for k in range(num_secondary_stations):
                    angle = np.random.uniform(0, 360)
                    lat, lng = get_destination_point(pri["lat"], pri["lng"], secondary_radius_km,
                                                     angle)
                    if haversine(pri["lat"], pri["lng"], lat, lng) > drone_range_km:
                        continue
                    if haversine(center_lat, center_lng, lat, lng) <= radius:
                        continue
                    secondary_stations.append({
                        "id": f"S{i + 1}_{pri['id']}_{k + 1}",
                        "lat": lat,
                        "lng": lng,
                        "primary_id": pri["id"]
                    })
        else:
            coords2 = outer_pois[['lat', 'lng']].values
            w2 = outer_pois['weight'].values
            weighted_outer = np.repeat(coords2, (w2 * 10).astype(int) + 1, axis=0)
            km_outer = KMeans(n_clusters=need, random_state=42, n_init=10)
            centers2 = km_outer.fit(weighted_outer).cluster_centers_
            idx = 1
            for lat, lng in centers2:
                nearest = None
                md = 9999
                for pri in primary_stations[-num_primary_stations_per_circle:]:
                    d0 = haversine(pri["lat"], pri["lng"], lat, lng)
                    if d0 < md and d0 <= drone_range_km:
                        nearest = pri
                        md = d0
                if nearest:
                    secondary_stations.append({
                        "id": f"S{i + 1}_{idx}",
                        "lat": lat,
                        "lng": lng,
                        "primary_id": nearest["id"]
                    })
                    idx += 1


    # 绘制地图
    st.write("选址结果地图")
    map_center = [all_pois["lat"].mean(), all_pois["lng"].mean()]
    m = folium.Map(
        location=map_center,
        zoom_start=11,
        tiles="https://webrd01.is.autonavi.com/appmaptile?x={x}&y={y}&z={z}&lang=zh_cn&size=1&scale=1&style=8",
        attr="高德地图"
    )
    for _, r in all_pois.iterrows():
        folium.CircleMarker(
            [r["lat"], r["lng"]], radius=2,
            color="gray", fill=True, fill_opacity=0.6
        ).add_to(m)
    for idx, c in enumerate(circles):
        popup_html = (
            f"<b>繁华中心 {idx + 1}</b><br>"
            f"中心：({c['center_lat']:.6f}, {c['center_lng']:.6f})<br>"
            f"半径：{c['radius_km']:.2f} km<br>"
            f"内部 POI 数量：{c['poi_count']}<br>"
        )
        folium.Circle(
            [c["center_lat"], c["center_lng"]],
            radius=c["radius_km"] * 1000,
            color="red", weight=3, fill=True, fill_color="red", fill_opacity=0.2,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"繁华中心 {idx + 1}"
        ).add_to(m)
        folium.Marker(
            [c["center_lat"], c["center_lng"]],
            icon=folium.Icon(color="red", icon="star"),
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"繁华中心 {idx + 1}"
        ).add_to(m)
        folium.Marker(
            [c["center_lat"], c["center_lng"]],
            icon=folium.Icon(color="red", icon="star"),
            popup=f"繁华中心 {idx + 1}"
        ).add_to(m)
    for p in primary_stations:
        folium.Marker(
            [p["lat"], p["lng"]],
            icon=folium.Icon(color="orange", icon="star"),
            popup=f"一级站 {p['id']}"
        ).add_to(m)
    for s in secondary_stations:
        folium.Marker(
            [s["lat"], s["lng"]],
            icon=folium.Icon(color="blue", icon="info-sign"),
            popup=f"二级站 {s['id']}（服务 {s['primary_id']}）"
        ).add_to(m)
    for s in secondary_stations:
        for p in primary_stations:
            if s["primary_id"] == p["id"]:
                folium.PolyLine(
                    [[p["lat"], p["lng"]], [s["lat"], s["lng"]]],
                    color="yellow", weight=3, opacity=0.7
                ).add_to(m)
                break
    # 显示地图
    st_folium(m, width=900, height=600, returned_objects=[])


    # 导出 CSV
    csv_data = []
    for idx, c in enumerate(circles):
        csv_data.append({
            '类型': '圆圈',
            '区编号': idx + 1,
            '中心纬度': round(c['center_lat'], 6),
            '中心经度': round(c['center_lng'], 6),
            '半径_km': round(c['radius_km'], 2),
            'POI数量': c['poi_count']
        })
    for p in primary_stations:
        csv_data.append({
            '类型': '一级站',
            '区编号': p['id'],
            '中心纬度': round(p['lat'], 6),
            '中心经度': round(p['lng'], 6),
            '服务圈': p['circle_id'],
            '半径_km': '',
            'POI数量': ''
        })
    for s in secondary_stations:
        csv_data.append({
            '类型': '二级站',
            '区编号': s['id'],
            '中心纬度': round(s['lat'], 6),
            '中心经度': round(s['lng'], 6),
            '服务于': s['primary_id'],
            '半径_km': '',
            'POI数量': ''
        })
        
        
    csv_df = pd.DataFrame(csv_data)
    st.write("下载结果")
    # 下载 HTML
    html_str = m.get_root().render()
    html_bytes = html_str.encode("utf-8")
    st.download_button("下载 HTML 地图文件",data=html_bytes,file_name=f"{city}_选址地图.html",mime="text/html")
    # 下载 CSV
    csv_buf = io.BytesIO()
    csv_df.to_csv(csv_buf, index=False, encoding="utf-8-sig")
    csv_buf.seek(0)
    st.download_button("下载站点数据 CSV", data=csv_buf.getvalue(),file_name=f"{city}_选址结果.csv", mime="text/csv")
    # 下载原始 POI 数据
    poi_buf = io.BytesIO()
    all_pois.to_csv(poi_buf, index=False, encoding="utf-8-sig")
    poi_buf.seek(0)
    st.download_button("下载POI数据 CSV", data=poi_buf.getvalue(),file_name=f"{city}_POI数据.csv", mime="text/csv")

