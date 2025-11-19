# ScenicPlanner.py
import requests
import pandas as pd
import numpy as np
import folium
import time
import math
from shapely.geometry import Point, MultiPoint, Polygon, mapping
from shapely.ops import unary_union
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull

# Try to import alphashape; if not available, we'll fallback to convex hull
try:
    import alphashape  # type: ignore
    _HAS_ALPHASHAPE = True
except Exception:
    _HAS_ALPHASHAPE = False

# -----------------------------
# Configurable defaults
# -----------------------------
SCENIC_KEYWORDS = [
    "风景名胜", "世界遗产", "国家级景点", "省级景点",
    "海滩", "观景点", "公园", "森林公园", "湖泊", "景区入口", "码头"
]

DEFAULT_PAGE_SIZE = 25
DEFAULT_MAX_PAGES = 3
REQUEST_TIMEOUT = 20
REQUEST_RETRIES = 3
ALPHA_DEFAULT = 0.03  # alphashape alpha; may adjust depending on density
EDGE_DISTANCE_M = 150  # consider POI within 150 meters of boundary as "edge POI"
HOTSPOT_CLUSTERS = 5  # default KMeans cluster count on edge points
MIN_EDGE_POIS = 5  # if too few edge POI, reduce cluster count fallback

# AMap tile URL (vector). style can be changed (7=矢量,6=卫星,8=卫星+标注)
AMAP_TILE_URL = "https://webrd02.is.autonavi.com/appmaptile?style=7&x={x}&y={y}&z={z}"


# -----------------------------
# Utility functions
# -----------------------------
def _safe_get_json(url, params, timeout=REQUEST_TIMEOUT, retries=REQUEST_RETRIES):
    """GET request with retries, returns parsed json or {}."""
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception:
            time.sleep(1.2 * (attempt + 1))
    return {}


def geocode_city(city_name: str, api_key: str):
    """Use AMap geocode to get a city center (lat,lng) or (None,None)."""
    if not city_name or not api_key:
        return None, None
    url = "https://restapi.amap.com/v3/geocode/geo"
    params = {"address": city_name, "key": api_key}
    j = _safe_get_json(url, params)
    try:
        if j.get("status") == "1" and j.get("geocodes"):
            loc = j["geocodes"][0].get("location")
            if loc:
                lng, lat = map(float, loc.split(","))
                return lat, lng
    except Exception:
        pass
    return None, None


def fetch_pois_amap(city: str, api_key: str, keywords=SCENIC_KEYWORDS,
                    page_size=DEFAULT_PAGE_SIZE, max_pages=DEFAULT_MAX_PAGES):
    """
    Fetch POIs from AMap for a list of keywords.
    Returns pandas.DataFrame with columns ['lng','lat','name','category'].
    """
    all_pois = []
    if isinstance(keywords, str):
        keywords = [keywords]

    for kw in keywords:
        for page in range(1, max_pages + 1):
            url = "https://restapi.amap.com/v3/place/text"
            params = {
                "keywords": kw,
                "city": city,
                "output": "json",
                "offset": page_size,
                "page": page,
                "key": api_key
            }
            j = _safe_get_json(url, params)
            if not j or j.get("status") != "1" or not j.get("pois"):
                break
            for poi in j.get("pois", []):
                loc = poi.get("location")
                if not loc:
                    continue
                try:
                    lng, lat = map(float, loc.split(","))
                except Exception:
                    continue
                name = poi.get("name", "")
                all_pois.append({"lng": lng, "lat": lat, "name": name, "category": kw})
    df = pd.DataFrame(all_pois)
    if not df.empty:
        df = df.drop_duplicates(subset=["lng", "lat", "name"])
    return df


def _meters_to_deg_lat(meters):
    """Approx convert meters to degrees lat (approx)."""
    return meters / 111000.0


def _meters_to_deg_lng(meters, lat):
    """Approx convert meters to degrees lng at given latitude."""
    return meters / (111000.0 * math.cos(math.radians(lat)) + 1e-12)


def build_boundary_from_points(df_points: pd.DataFrame, alpha=ALPHA_DEFAULT):
    """
    Build a boundary polygon from point set.
    Prefer alphashape (gives concave), otherwise fallback to convex hull.
    Input df_points: must have 'lng','lat' columns.
    Returns shapely Polygon (or MultiPolygon); may return None if not possible.
    """
    if df_points is None or df_points.empty:
        return None

    points = [Point(xy) for xy in zip(df_points["lng"].values, df_points["lat"].values)]

    # if not enough points, return buffered point
    if len(points) == 1:
        return points[0].buffer(0.001)  # small buffer
    if len(points) == 2:
        return MultiPoint(points).convex_hull.buffer(0.001)

    try:
        if _HAS_ALPHASHAPE:
            shape = alphashape.alphashape(points, alpha)
            if shape is None or shape.is_empty:
                # fallback to convex hull
                coords = np.array([[p.x, p.y] for p in points])
                hull = ConvexHull(coords)
                poly = Polygon(coords[hull.vertices])
                return poly
            # ensure polygon
            if isinstance(shape, (Polygon, MultiPoint)):
                return shape
            else:
                return Polygon(shape.exterior) if hasattr(shape, "exterior") else shape
        else:
            # fallback: convex hull
            coords = np.array([[p.x, p.y] for p in points])
            hull = ConvexHull(coords)
            poly = Polygon(coords[hull.vertices])
            return poly
    except Exception:
        # last-resort fallback: unified buffer
        try:
            mp = MultiPoint(points)
            return mp.convex_hull
        except Exception:
            return None


def boundary_edge_pois(poi_df: pd.DataFrame, boundary, distance_m=EDGE_DISTANCE_M):
    """
    Return POIs that are within 'distance_m' meters from the boundary polygon.
    Uses approximate conversion meters -> degrees for filtering performance, then exact shapely distance.
    """
    if poi_df is None or poi_df.empty or boundary is None:
        return pd.DataFrame(columns=poi_df.columns if poi_df is not None else ["lng", "lat"])

    # approximate lat center for degree conversion
    center_lat = poi_df["lat"].mean()
    deg_lat = _meters_to_deg_lat(distance_m)
    deg_lng = _meters_to_deg_lng(distance_m, center_lat)

    # coarse bounding filter to speed up
    minx, miny, maxx, maxy = boundary.bounds
    buf_minx, buf_maxx = minx - deg_lng, maxx + deg_lng
    buf_miny, buf_maxy = miny - deg_lat, maxy + deg_lat

    subset = poi_df[
        (poi_df["lng"] >= buf_minx) & (poi_df["lng"] <= buf_maxx) &
        (poi_df["lat"] >= buf_miny) & (poi_df["lat"] <= buf_maxy)
    ].copy()

    if subset.empty:
        return subset

    # exact check: shapely distance in degrees (approx), we convert meters->degrees roughly
    # because shapely operates in the same coordinate system (deg). We compare using meter threshold converted to approx degrees.
    # Use center_lat conversion for approximate threshold
    thresh_deg_lat = deg_lat
    # compute exact distances using geodesic? approximate is OK here
    def close_to_boundary(row):
        p = Point(row["lng"], row["lat"])
        # shapely.distance is in degree units; convert to meters via approximate factor
        d_deg = p.distance(boundary)  # degrees
        # convert deg to meters roughly
        d_m = d_deg * 111000.0
        return d_m <= distance_m + 1e-6

    subset["near_boundary"] = subset.apply(close_to_boundary, axis=1)
    return subset[subset["near_boundary"]].drop(columns=["near_boundary"])


def detect_hotspots(edge_pois: pd.DataFrame, n_clusters=HOTSPOT_CLUSTERS):
    """
    Run KMeans on edge POIs to find hotspot centers.
    Returns DataFrame of clusters and list of cluster-center dicts.
    """
    if edge_pois is None or edge_pois.empty:
        return pd.DataFrame(), []

    coords = edge_pois[["lat", "lng"]].values
    # pick cluster count adaptively
    n = len(edge_pois)
    k = min(max(1, n_clusters), max(1, int(n / 5)))  # roughly one cluster per ~5 points, bounded
    k = max(1, k)
    try:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(coords)
        edge_pois = edge_pois.copy()
        edge_pois["cluster"] = labels
        centers = []
        for i in range(k):
            group = edge_pois[edge_pois["cluster"] == i]
            if group.empty:
                continue
            lat_mean = float(group["lat"].mean())
            lng_mean = float(group["lng"].mean())
            centers.append({"cluster": i, "lat": lat_mean, "lng": lng_mean, "count": len(group)})
        return edge_pois, centers
    except Exception:
        # fallback: treat midpoints
        lat_mean = float(edge_pois["lat"].mean())
        lng_mean = float(edge_pois["lng"].mean())
        return edge_pois.assign(cluster=0), [{"cluster": 0, "lat": lat_mean, "lng": lng_mean, "count": len(edge_pois)}]


def plan_edge_stations(centers, primary_radius_m=200):
    """
    Given centers (list of dicts with lat/lng), create a list of primary stations.
    primary_radius_m can be used downstream if needed.
    """
    stations = []
    for idx, c in enumerate(centers, start=1):
        stations.append({
            "id": f"S_{idx}",
            "lat": c["lat"],
            "lng": c["lng"],
            "count": c.get("count", 0),
            "primary_radius_m": primary_radius_m
        })
    return stations


# -----------------------------
# Main runner
# -----------------------------
def run_scenic(city: str, api_key: str,
               scenic_keywords=SCENIC_KEYWORDS,
               page_size=DEFAULT_PAGE_SIZE,
               max_pages=DEFAULT_MAX_PAGES,
               edge_distance_m=EDGE_DISTANCE_M,
               hotspot_clusters=HOTSPOT_CLUSTERS):
    """
    Main entry: given city name and api_key, returns (folium.Map, info_dict).

    info_dict contains:
        - poi_count
        - boundary_summary (bounds / area)
        - edge_poi_count
        - stations (list)
    """
    # basic validation
    if not city or not api_key:
        raise RuntimeError("city and api_key are required for run_scenic")

    # 1) fetch scenic POI
    poi_df = fetch_pois_amap(city, api_key, keywords=scenic_keywords,
                             page_size=page_size, max_pages=max_pages)

    # fallback: if no POI, try geocode city and create a single-point POI
    if poi_df.empty:
        lat_c, lng_c = geocode_city(city, api_key)
        if lat_c is None:
            raise RuntimeError("未获取到任何景区 POI，且无法通过地理编码回退。请检查城市名称或 API Key。")
        poi_df = pd.DataFrame([{"lng": lng_c, "lat": lat_c, "name": city, "category": "geocode_fallback"}])

    # 2) build boundary
    boundary = build_boundary_from_points(poi_df, alpha=ALPHA_DEFAULT)
    if boundary is None:
        raise RuntimeError("无法构建景区边界（点集异常）")

    # 3) extract edge POIs (near boundary)
    edge_pois = boundary_edge_pois(poi_df, boundary, distance_m=edge_distance_m)

    # if edge_pois too few, relax threshold once
    if edge_pois.shape[0] < MIN_EDGE_POIS and edge_distance_m < 1000:
        edge_pois = boundary_edge_pois(poi_df, boundary, distance_m=max(edge_distance_m, 300))

    # 4) detect hotspots on edge
    edge_pois_clustered, centers = detect_hotspots(edge_pois, n_clusters=hotspot_clusters)

    # 5) plan stations
    stations = plan_edge_stations(centers, primary_radius_m=200)

    # 6) produce folium map with AMap tiles
    map_center = [float(poi_df["lat"].mean()), float(poi_df["lng"].mean())]
    m = folium.Map(location=map_center, zoom_start=12, tiles=None)
    folium.TileLayer(tiles=AMAP_TILE_URL, attr="高德地图", name="高德矢量图", control=False).add_to(m)

    # draw boundary
    try:
        folium.GeoJson(mapping(boundary), name="景区边界",
                       tooltip=f"{city} 景区边界").add_to(m)
    except Exception:
        # fallback: if mapping fails, try convex hull points
        try:
            hull = MultiPoint([Point(xy) for xy in zip(poi_df["lng"], poi_df["lat"])]).convex_hull
            folium.GeoJson(mapping(hull), name="景区凸包").add_to(m)
        except Exception:
            pass

    # draw POIs (sample if too many)
    for _, r in poi_df.iterrows():
        folium.CircleMarker([r["lat"], r["lng"]], radius=3, color="#777777", fill=True, fill_opacity=0.6,
                            popup=r.get("name", "")).add_to(m)

    # draw edge pois in distinct color
    for _, r in edge_pois_clustered.iterrows() if not edge_pois_clustered.empty else []:
        folium.CircleMarker([r["lat"], r["lng"]], radius=4, color="#ff7800", fill=True, fill_opacity=0.9).add_to(m)

    # draw station markers
    for s in stations:
        folium.Marker([s["lat"], s["lng"]],
                      icon=folium.Icon(color="red", icon="plane"),
                      popup=f"一级站 {s['id']} (edge_count={s['count']})").add_to(m)
        # draw coverage circle
        folium.Circle([s["lat"], s["lng"]], radius=s["primary_radius_m"], color="red", weight=1, fill=False).add_to(m)

    info = {
        "city": city,
        "poi_count": int(poi_df.shape[0]),
        "boundary_bounds": boundary.bounds if boundary is not None else None,
        "edge_poi_count": int(edge_pois.shape[0]),
        "stations": stations
    }

    return m, info


# If run as script for quick test (not executed when imported)
if __name__ == "__main__":
    # simple local test (you must set API_KEY and city)
    API_KEY = "<YOUR_AMAP_KEY>"
    CITY = "西宁市"
    try:
        m, info = run_scenic(CITY, API_KEY)
        print("info:", info)
        # save map to file for local inspection
        m.save(f"{CITY}_scenic_map.html")
        print("map saved")
    except Exception as e:
        print("run_scenic failed:", e)
