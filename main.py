import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import shapely.wkt
from skimage.filters import threshold_otsu
from sentinelhub import CRS, BBox, DataCollection
from eolearn.core import EOTask, EOWorkflow, FeatureType, OutputTask, linearly_connect_tasks
from eolearn.features import NormalizedDifferenceIndexTask, SimpleFilterTask
from eolearn.geometry import VectorToRasterTask
from eolearn.io import SentinelHubInputTask
import osmnx as ox

# -----------------------------------
# STEP 1: Fetch Lake Polygon
# -----------------------------------

def fetch_lake_polygon_wkt(lake_name, country="Poland"):
    place_name = f"{lake_name}, {country}"
    tags = {"natural": "water"}
    gdf = ox.features_from_place(place_name, tags)
    gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]
    if gdf.empty:
        raise ValueError(f"Lake '{lake_name}' not found in {country}")
    geom = gdf.geometry.iloc[0]
    return geom.wkt, geom

lake_name = "Isąg"
dam_wkt, dam_nominal = fetch_lake_polygon_wkt(lake_name)

# -----------------------------------
# STEP 2: Create Bounding Box
# -----------------------------------

inflate_bbox = 0.1
minx, miny, maxx, maxy = dam_nominal.bounds
delx, dely = maxx - minx, maxy - miny
minx, maxx = minx - delx * inflate_bbox, maxx + delx * inflate_bbox
miny, maxy = miny - dely * inflate_bbox, maxy + dely * inflate_bbox
dam_bbox = BBox([minx, miny, maxx, maxy], crs=CRS.WGS84)

# -----------------------------------
# STEP 3: EO-Learn Tasks
# -----------------------------------

# Download Sentinel-2 data
download_task = SentinelHubInputTask(
    data_collection=DataCollection.SENTINEL2_L1C,
    bands_feature=(FeatureType.DATA, "BANDS"),
    resolution=30,
    maxcc=0.5,
    bands=["B02", "B03", "B04", "B08"],
    additional_data=[(FeatureType.MASK, "dataMask", "IS_DATA"), (FeatureType.MASK, "CLM")],
    cache_folder="cached_data"
)

# Calculate NDWI
calculate_ndwi = NormalizedDifferenceIndexTask(
    (FeatureType.DATA, "BANDS"), (FeatureType.DATA, "NDWI"), (1, 3)
)

# Add nominal water mask
dam_gdf = gpd.GeoDataFrame(crs=CRS.WGS84.pyproj_crs(), geometry=[dam_nominal])
add_nominal_water = VectorToRasterTask(
    dam_gdf,
    (FeatureType.MASK_TIMELESS, "NOMINAL_WATER"),
    values=1,
    raster_shape=(FeatureType.MASK, "IS_DATA"),
    raster_dtype=np.uint8,
)

# Add valid data mask (non-cloudy areas)
class AddValidDataMaskTask(EOTask):
    def execute(self, eopatch):
        is_data_mask = eopatch[FeatureType.MASK, "IS_DATA"].astype(bool)
        cloud_mask = ~eopatch[FeatureType.MASK, "CLM"].astype(bool)
        eopatch[FeatureType.MASK, "VALID_DATA"] = np.logical_and(is_data_mask, cloud_mask)
        return eopatch

add_valid_mask = AddValidDataMaskTask()

# Calculate coverage
def calculate_coverage(array):
    return 1.0 - np.count_nonzero(array) / np.size(array)

class AddValidDataCoverageTask(EOTask):
    def execute(self, eopatch):
        valid_data = eopatch[FeatureType.MASK, "VALID_DATA"]
        time, height, width, channels = valid_data.shape
        coverage = np.apply_along_axis(calculate_coverage, 1, valid_data.reshape((time, height * width * channels)))
        eopatch[FeatureType.SCALAR, "COVERAGE"] = coverage[:, np.newaxis]
        return eopatch

add_coverage = AddValidDataCoverageTask()

# Remove cloudy scenes
cloud_coverage_threshold = 0.05

class ValidDataCoveragePredicate:
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, array):
        return calculate_coverage(array) < self.threshold

remove_cloudy_scenes = SimpleFilterTask(
    (FeatureType.MASK, "VALID_DATA"), ValidDataCoveragePredicate(cloud_coverage_threshold)
)

# Detect water levels with minimum valid coverage check
class WaterDetectionTask(EOTask):
    def __init__(self, ndwi_feature=(FeatureType.DATA, "NDWI")):
        self.ndwi_feature = ndwi_feature

    def detect_water(self, ndwi):
        ndwi = np.where(np.isnan(ndwi), -1, ndwi)
        threshold = threshold_otsu(ndwi[ndwi > -1])
        return ndwi > threshold

    def execute(self, eopatch):
        min_valid_coverage = 0.5  # Require at least 50% valid data in nominal water area
        water_masks = np.asarray([self.detect_water(ndwi[..., 0]) for ndwi in eopatch.data["NDWI"]])
        water_masks = water_masks[..., np.newaxis] * eopatch.mask_timeless["NOMINAL_WATER"]
        valid_data = eopatch.mask["VALID_DATA"]
        nominal_water = eopatch.mask_timeless["NOMINAL_WATER"][..., 0]
        water_levels = []
        for t in range(len(water_masks)):
            valid_nominal = nominal_water & valid_data[t, ..., 0]
            valid_coverage_ratio = np.sum(valid_nominal) / np.sum(nominal_water)
            if valid_coverage_ratio < min_valid_coverage:
                water_level = np.nan  # Exclude if insufficient valid data
            else:
                water_level = np.sum(water_masks[t, ..., 0] & valid_data[t, ..., 0]) / np.sum(valid_nominal)
            water_levels.append(water_level)
        eopatch[FeatureType.MASK, "WATER_MASK"] = water_masks
        eopatch[FeatureType.SCALAR, "WATER_LEVEL"] = np.array(water_levels)[..., np.newaxis]
        return eopatch

water_detection = WaterDetectionTask()

# Output task
output_task = OutputTask("final_eopatch")

# Execute workflow
workflow_nodes = linearly_connect_tasks(
    download_task,
    calculate_ndwi,
    add_nominal_water,
    add_valid_mask,
    add_coverage,
    remove_cloudy_scenes,
    water_detection,
    output_task
)
workflow = EOWorkflow(workflow_nodes)
download_node = workflow_nodes[0]
time_interval = ["2021-01-01", "2024-12-31"]

result = workflow.execute({
    download_node: {"bbox": dam_bbox, "time_interval": time_interval}
})
patch = result.outputs["final_eopatch"]

# -----------------------------------
# STEP 4: Remove Anomalies with Rolling Median
# -----------------------------------

def remove_anomalies(water_levels, window=5, threshold=0.1):
    """
    Remove anomalies in water levels using a rolling median approach.

    Args:
        water_levels (np.array): Array of water level values.
        window (int): Size of the rolling window (default: 5).
        threshold (float): Difference threshold to identify anomalies (default: 0.1).

    Returns:
        np.array: Water levels with anomalies set to NaN.
    """
    water_levels = water_levels.copy()
    n = len(water_levels)
    for i in range(n):
        start = max(0, i - window // 2)
        end = min(n, i + window // 2 + 1)
        window_values = water_levels[start:end]
        median = np.nanmedian(window_values)
        if not np.isnan(median) and abs(water_levels[i] - median) > threshold:
            water_levels[i] = np.nan
    return water_levels

# Apply anomaly removal
water_levels = patch.scalar["WATER_LEVEL"][..., 0]
water_levels_clean = remove_anomalies(water_levels, window=5, threshold=0.1)
patch.scalar["WATER_LEVEL"][..., 0] = water_levels_clean

# -----------------------------------
# STEP 5: Plotting
# -----------------------------------

from skimage.filters import sobel
from skimage.morphology import closing, disk

def plot_rgb_w_water(eopatch, idx, filename=None):
    ratio = np.abs(eopatch.bbox.max_x - eopatch.bbox.min_x) / np.abs(eopatch.bbox.max_y - eopatch.bbox.min_y)
    fig, ax = plt.subplots(figsize=(ratio * 10, 10))
    ax.imshow(np.clip(2.5 * eopatch.data["BANDS"][..., [2, 1, 0]][idx], 0, 1))
    observed = closing(eopatch.mask["WATER_MASK"][idx, ..., 0], disk(1))
    nominal = sobel(eopatch.mask_timeless["NOMINAL_WATER"][..., 0])
    observed = sobel(observed)
    nominal = np.ma.masked_where(~nominal.astype(bool), nominal)
    observed = np.ma.masked_where(~observed.astype(bool), observed)
    ax.imshow(nominal, cmap=plt.cm.Reds)
    ax.imshow(observed, cmap=plt.cm.Blues)
    ax.axis("off")
    if filename:
        plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close(fig)

def plot_water_levels(eopatch, max_coverage=1.0, filename=None):
    fig, ax = plt.subplots(figsize=(20, 7))
    dates = np.asarray(eopatch.timestamps)
    valid_indices = ~np.isnan(eopatch.scalar["WATER_LEVEL"][..., 0])
    valid_dates = dates[valid_indices]
    valid_water_levels = eopatch.scalar["WATER_LEVEL"][valid_indices]
    valid_coverage = eopatch.scalar["COVERAGE"][valid_indices]
    ax.plot(valid_dates, valid_water_levels, "bo-", alpha=0.7, label="Water Level")
    ax.plot(valid_dates, valid_coverage, "--", color="gray", alpha=0.7, label="Valid Data Coverage")
    ax.set_ylim(0.0, 1.1)
    ax.set_xlabel("Date")
    ax.set_ylabel("Water Level / Coverage")
    ax.set_title(f"{lake_name} Water Levels")
    ax.grid(axis="y")
    ax.legend()
    if filename:
        plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close(fig)

# -----------------------------------
# STEP 6: Save Results
# -----------------------------------

plot_rgb_w_water(patch, 0, "water_overlay_first.png")
plot_rgb_w_water(patch, -1, "water_overlay_last.png")
plot_water_levels(patch, 1.0, "water_levels.png")
plt.show()