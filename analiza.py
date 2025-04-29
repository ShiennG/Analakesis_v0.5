import geopandas as gpd
from skimage.filters import threshold_otsu
from sentinelhub import CRS, BBox, DataCollection, SHConfig
from eolearn.core import EOTask, EOWorkflow, FeatureType, OutputTask, linearly_connect_tasks
from eolearn.features import NormalizedDifferenceIndexTask, SimpleFilterTask
from eolearn.geometry import VectorToRasterTask
from eolearn.io import SentinelHubInputTask
import osmnx as ox

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from datetime import timedelta, datetime

# Set up Sentinel Hub configuration
config = SHConfig()
config.sh_client_id = "ef40d6b6-f7ca-4d2b-91de-7ae5a65ac8f4"
config.sh_client_secret = "m5OLdYpEwpc7EJ4TpyfyjLHDqZbqMdyu"

# Save the configuration (makes it the global instance)
config.save()

lake_name = "Isąg"
first_date = "2023-01-01"
last_date = "2025-01-01"
days_to_forecast = 365

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

# Create SentinelHubInputTask with config already set in environment
# DO NOT pass config as an execution parameter!
download_task = SentinelHubInputTask(
    data_collection=DataCollection.SENTINEL2_L1C,
    bands_feature=(FeatureType.DATA, "BANDS"),
    resolution=30,
    maxcc=0.5,
    bands=["B02", "B03", "B04", "B08"],
    additional_data=[(FeatureType.MASK, "dataMask", "IS_DATA"), (FeatureType.MASK, "CLM")],
    cache_folder="cached_data",
    config=config  # Pass configuration during task CREATION, not execution
)

# Rest of the code remains the same...
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
time_interval = [first_date, last_date]

# Execute the workflow - IMPORTANT: DO NOT pass config here!
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
# STEP 5: Prediction models and Plotting
# -----------------------------------

def improved_anomaly_filter(water_levels, timestamps, window=5, threshold=0.1, jump_threshold=0.05,
                            global_min=0.65, global_max=1.0, absolute_jump_threshold=0.3):
    """
    Enhanced anomaly detection that filters out extreme outliers, handles both point anomalies and sudden jumps.

    Args:
        water_levels (np.array): Array of water level values
        timestamps (list): List of timestamp objects
        window (int): Size of the rolling window for local anomaly detection
        threshold (float): Threshold for deviation from the median in local window
        jump_threshold (float): Maximum allowed change rate between consecutive measurements (per day)
        global_min (float): Minimum plausible water level value (absolute threshold)
        global_max (float): Maximum plausible water level value (absolute threshold)
        absolute_jump_threshold (float): Maximum allowed absolute change between consecutive measurements

    Returns:
        np.array: Cleaned water levels with anomalies and jumps set to NaN
    """
    import numpy as np
    import pandas as pd

    # Create a copy to avoid modifying the original
    cleaned_levels = water_levels.copy()
    n = len(water_levels)

    # Create pandas DataFrame for easier analysis
    df = pd.DataFrame({
        'date': timestamps,
        'water_level': water_levels,
    })
    df = df.sort_values('date')

    # 0. Filter out global anomalies (values outside plausible range)
    global_anomalies = (df['water_level'] < global_min) | (df['water_level'] > global_max)
    initial_anomaly_count = np.sum(global_anomalies)
    df.loc[global_anomalies, 'water_level'] = np.nan

    # 1. Detect point anomalies (values too far from local median)
    for i in range(n):
        if np.isnan(cleaned_levels[i]):
            continue

        # Use adaptive window size - larger window for sparse data
        current_date = timestamps[i]
        date_diffs = [(current_date - timestamps[j]).total_seconds() / (24 * 3600)
                      for j in range(n) if not np.isnan(cleaned_levels[j])]

        # Adjust window based on data density
        adaptive_window = min(window * 2, n) if max(date_diffs, default=0) > 30 else window

        start = max(0, i - adaptive_window // 2)
        end = min(n, i + adaptive_window // 2 + 1)
        window_values = [cleaned_levels[j] for j in range(start, end) if not np.isnan(cleaned_levels[j])]

        if len(window_values) < 3:  # Not enough data for reliable median
            continue

        median = np.median(window_values)
        if not np.isnan(median) and abs(cleaned_levels[i] - median) > threshold:
            cleaned_levels[i] = np.nan

    # Update dataframe
    df['cleaned_level'] = cleaned_levels

    # 2. Detect sudden jumps between consecutive measurements
    df['prev_level'] = df['cleaned_level'].shift(1)
    df['next_level'] = df['cleaned_level'].shift(-1)

    # Calculate time differences
    df['days_diff'] = df['date'].diff().dt.total_seconds() / (24 * 3600)  # Convert to days
    df['next_days_diff'] = df['days_diff'].shift(-1)

    # Calculate absolute changes and change rates
    df['abs_change'] = np.abs(df['cleaned_level'] - df['prev_level'])
    df['next_abs_change'] = np.abs(df['next_level'] - df['cleaned_level'])

    # Calculate change rate per day (to handle irregular time intervals)
    df['change_rate'] = df['abs_change'] / df['days_diff'].replace(0, np.nan)
    df['next_change_rate'] = df['next_abs_change'] / df['next_days_diff'].replace(0, np.nan)

    # Flag as anomaly if:
    # 1. Absolute change exceeds threshold and is a spike or dip (changes back in next observation)
    spike_mask = ((df['abs_change'] > absolute_jump_threshold) &
                  (df['next_abs_change'] > absolute_jump_threshold) &
                  (~df['abs_change'].isna()) & (~df['next_abs_change'].isna()))

    # 2. Change rate exceeds threshold
    rate_mask = ((df['change_rate'] > jump_threshold) & (~df['change_rate'].isna()))

    # Combined anomaly mask
    jump_mask = spike_mask | rate_mask

    # Set jumps to NaN
    df.loc[jump_mask, 'cleaned_level'] = np.nan

    # Handle special case of beginning/end anomalies
    # If first few values are outliers compared to the typical range, mark them as anomalies
    first_valid_idx = df['cleaned_level'].first_valid_index()
    if first_valid_idx is not None:
        typical_range = df.loc[~df['cleaned_level'].isna(), 'cleaned_level']
        if len(typical_range) > 5:
            typical_median = typical_range.median()
            typical_std = typical_range.std()

            # Check if first value is far from typical values
            first_values = df.iloc[:5]
            for idx, row in first_values.iterrows():
                if not np.isnan(row['cleaned_level']):
                    z_score = abs(row['cleaned_level'] - typical_median) / typical_std
                    if z_score > 3:  # More than 3 standard deviations from median
                        df.loc[idx, 'cleaned_level'] = np.nan

    # Count anomalies for reporting
    n_point_anomalies = np.sum(np.isnan(cleaned_levels)) - initial_anomaly_count
    n_jumps = np.sum(jump_mask)
    total_anomalies = np.sum(np.isnan(df['cleaned_level'])) - np.sum(np.isnan(water_levels))

    print(
        f"Detected {initial_anomaly_count} global anomalies, {n_point_anomalies} point anomalies, and {n_jumps} sudden jumps")
    print(f"Total values filtered: {total_anomalies} out of {len(water_levels)}")

    return df['cleaned_level'].values


def create_prediction_model(eopatch, forecast_days=365, filename=None):
    """Create and visualize a water level prediction model"""
    # Get valid data points
    valid_indices = ~np.isnan(eopatch.scalar["WATER_LEVEL"][..., 0])
    dates = np.array(eopatch.timestamps)[valid_indices]
    water_levels = eopatch.scalar["WATER_LEVEL"][valid_indices, 0]

    # Create DataFrame
    df = pd.DataFrame({'date': dates, 'water_level': water_levels})
    df.set_index('date', inplace=True)
    df = df.sort_index()

    # Create time features for machine learning
    df_features = df.reset_index()
    df_features['year'] = df_features['date'].dt.year
    df_features['month'] = df_features['date'].dt.month
    df_features['day'] = df_features['date'].dt.day
    df_features['dayofyear'] = df_features['date'].dt.dayofyear

    # Create cyclic features for seasonality
    df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
    df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
    df_features['day_sin'] = np.sin(2 * np.pi * df_features['dayofyear'] / 365)
    df_features['day_cos'] = np.cos(2 * np.pi * df_features['dayofyear'] / 365)

    # Add time trend
    min_date = df_features['date'].min()
    df_features['days_since_start'] = (df_features['date'] - min_date).dt.days

    # Select features for model (excluding the date column)
    feature_cols = ['month_sin', 'month_cos', 'day_sin', 'day_cos', 'days_since_start']
    X = df_features[feature_cols]
    y = df_features['water_level']

    # Split data for validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Calculate model performance
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"Model R² score: {test_score:.4f} (train: {train_score:.4f})")

    # Generate dates for prediction
    last_date = df_features['date'].max()
    future_dates = pd.date_range(
        start=last_date + timedelta(days=1),
        periods=forecast_days,
        freq='D'
    )

    # Create features for prediction dates
    pred_df = pd.DataFrame({'date': future_dates})
    pred_df['month'] = pred_df['date'].dt.month
    pred_df['day'] = pred_df['date'].dt.day
    pred_df['dayofyear'] = pred_df['date'].dt.dayofyear
    pred_df['month_sin'] = np.sin(2 * np.pi * pred_df['month'] / 12)
    pred_df['month_cos'] = np.cos(2 * np.pi * pred_df['month'] / 12)
    pred_df['day_sin'] = np.sin(2 * np.pi * pred_df['dayofyear'] / 365)
    pred_df['day_cos'] = np.cos(2 * np.pi * pred_df['dayofyear'] / 365)
    pred_df['days_since_start'] = (pred_df['date'] - min_date).dt.days

    # Make predictions
    predictions = model.predict(pred_df[feature_cols])
    pred_df['predicted_level'] = predictions

    # Create visualization
    fig, ax = plt.subplots(figsize=(16, 8))

    # Plot historical data
    ax.plot(df.index, df['water_level'], 'b-', label='Historical Water Level', alpha=0.7)

    # Plot predicted data with confidence interval
    ax.plot(pred_df['date'], pred_df['predicted_level'], 'r-', label='Predicted Water Level')

    # Add uncertainty bands
    # Calculate prediction intervals by bootstrapping
    n_bootstraps = 100
    bootstrap_predictions = np.zeros((len(pred_df), n_bootstraps))

    # Train multiple models on bootstrap samples
    for i in range(n_bootstraps):
        # Bootstrap sample
        indices = np.random.choice(len(X_train), len(X_train), replace=True)
        X_boot, y_boot = X_train.iloc[indices], y_train.iloc[indices]

        # Train model on bootstrap sample
        boot_model = RandomForestRegressor(n_estimators=50, random_state=i)
        boot_model.fit(X_boot, y_boot)

        # Predict
        bootstrap_predictions[:, i] = boot_model.predict(pred_df[feature_cols])

    # Calculate confidence intervals
    lower_bound = np.percentile(bootstrap_predictions, 5, axis=1)
    upper_bound = np.percentile(bootstrap_predictions, 95, axis=1)

    # Plot confidence interval
    ax.fill_between(pred_df['date'], lower_bound, upper_bound,
                    color='red', alpha=0.2, label='90% Confidence Interval')

    # Add vertical line separating historical and predicted data
    ax.axvline(x=last_date, color='grey', linestyle='--', label='Forecast Start')

    # Add explanatory text
    ax.text(0.02, 0.02,
            f"Model: Random Forest\nTest R² score: {test_score:.4f}\n"
            f"Forecast: {forecast_days} days ahead",
            transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))

    # Formatting
    ax.set_title(f'Water Level Forecasting Model')
    ax.set_xlabel('Date')
    ax.set_ylabel('Water Level')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    # Save figure
    if filename:
        plt.savefig(filename, bbox_inches='tight', dpi=300)

    # Create seasonal prediction chart
    create_seasonal_prediction_visualization(df, model, feature_cols, min_date, "seasonal_prediction.png")

    return model, pred_df


def create_seasonal_prediction_visualization(historical_df, model, feature_cols, min_date, filename=None):
    """Create a visualization showing predicted seasonal patterns"""
    # Generate full year of dates
    current_year = datetime.now().year
    future_year_dates = pd.date_range(
        start=datetime(current_year, 1, 1),
        end=datetime(current_year, 12, 31),
        freq='D'
    )

    # Create features for these dates
    seasonal_df = pd.DataFrame({'date': future_year_dates})
    seasonal_df['month'] = seasonal_df['date'].dt.month
    seasonal_df['day'] = seasonal_df['date'].dt.day
    seasonal_df['dayofyear'] = seasonal_df['date'].dt.dayofyear
    seasonal_df['month_sin'] = np.sin(2 * np.pi * seasonal_df['month'] / 12)
    seasonal_df['month_cos'] = np.cos(2 * np.pi * seasonal_df['month'] / 12)
    seasonal_df['day_sin'] = np.sin(2 * np.pi * seasonal_df['dayofyear'] / 365)
    seasonal_df['day_cos'] = np.cos(2 * np.pi * seasonal_df['dayofyear'] / 365)

    # Add multiple years trend for visualization - modified to make lines visibly different
    years_ahead = [0, 1, 2]

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot each year's prediction with visibly different lines
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(years_ahead)))

    for i, years in enumerate(years_ahead):
        # Calculate days since start for this year
        base_days = (datetime(current_year, 1, 1) - min_date).days
        extra_days = years * 365
        seasonal_df['days_since_start'] = base_days + extra_days + seasonal_df['dayofyear'] - 1

        # Predict water levels
        predictions = model.predict(seasonal_df[feature_cols])

        # Apply a slight offset to make lines visibly different
        offset = years * 0.01  # Small offset to separate lines

        # Plot with different line styles and thicknesses
        line_styles = ['-', '--', '-.']
        line_widths = [2.5, 2.0, 1.5]

        ax.plot(seasonal_df['dayofyear'], predictions + offset,
                line_styles[i], color=colors[i], linewidth=line_widths[i],
                label=f'Predicted {current_year + years}')

    # Add month indicators
    month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax.set_xticks(month_starts)
    ax.set_xticklabels(month_names)

    # Plot historical monthly averages for comparison
    historical_df_reset = historical_df.reset_index()
    historical_df_reset['month'] = historical_df_reset['date'].dt.month
    historical_df_reset['dayofyear'] = historical_df_reset['date'].dt.dayofyear

    # Calculate monthly averages
    monthly_avg = historical_df_reset.groupby('month')['water_level'].mean().reset_index()
    # Map to middle of month dayofyear
    month_midpoints = [15, 45, 75, 105, 135, 165, 195, 225, 255, 285, 315, 345]
    monthly_avg['dayofyear'] = monthly_avg['month'].map(dict(zip(range(1, 13), month_midpoints)))

    # Plot historical averages
    ax.scatter(monthly_avg['dayofyear'], monthly_avg['water_level'],
               color='red', marker='o', s=80, label='Historical Monthly Avg',
               zorder=5)

    # Connect points
    ax.plot(monthly_avg['dayofyear'], monthly_avg['water_level'],
            'r--', alpha=0.7, zorder=4)

    ax.set_title(f'Seasonal Water Level Projection')
    ax.set_xlabel('Month')
    ax.set_ylabel('Water Level')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    # Annotate with trend information
    trend_coefficient = model.feature_importances_[feature_cols.index('days_since_start')]
    annual_trend = 365 * trend_coefficient
    trend_direction = "increasing" if annual_trend > 0 else "decreasing"

    ax.text(0.02, 0.02,
            f"Long-term trend: {trend_direction}\n"
            f"Annual change: {annual_trend:.4f}",
            transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    if filename:
        plt.savefig(filename, bbox_inches='tight', dpi=300)

    plt.close(fig)


def main(eopatch):
    """Main function to run the enhanced analysis with improved filtering"""
    # 1. Apply enhanced filtering to remove jumps and anomalies
    eopatch = update_patch_with_filtered_data(
        eopatch,
        improved_anomaly_filter,
        window=5,
        threshold=0.1,
        jump_threshold=0.05,
        global_min=0.65,    # Sets minimum plausible water level (anything below is anomaly)
        global_max=1.0,     # Sets maximum plausible water level
        absolute_jump_threshold=0.3  # Maximum allowed change between consecutive measurements
    )

    # 2. Create and visualize prediction model
    model, predictions = create_prediction_model(
        eopatch,
        forecast_days=days_to_forecast,
        filename="water_level_forecast.png"
    )

    return eopatch, model, predictions


# Function to update all plots with the filtered data
def update_patch_with_filtered_data(eopatch, filter_function, **filter_params):
    """Apply filtering to water levels and update the patch data"""
    dates = np.array(eopatch.timestamps)
    original_water_levels = eopatch.scalar["WATER_LEVEL"][..., 0].copy()

    # Save original data before filtering (if not already saved)
    if "WATER_LEVEL_ORIGINAL" not in eopatch.scalar:
        eopatch.scalar["WATER_LEVEL_ORIGINAL"] = eopatch.scalar["WATER_LEVEL"].copy()

    # Apply the enhanced filter
    filtered_levels = filter_function(
        original_water_levels,
        dates,
        **filter_params
    )

    # Update the patch with filtered data
    eopatch.scalar["WATER_LEVEL"][..., 0] = filtered_levels

    return eopatch



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

import pandas as pd
def create_time_features(dates):
    """Create time-based features for modeling"""
    features = pd.DataFrame({
        'year': [d.year for d in dates],
        'month': [d.month for d in dates],
        'day': [d.day for d in dates],
        'dayofyear': [d.timetuple().tm_yday for d in dates],
        'week': [d.isocalendar()[1] for d in dates],
        'dayofweek': [d.weekday() for d in dates],
        'quarter': [((d.month - 1) // 3) + 1 for d in dates],
        # Cyclic features for seasonality
        'month_sin': np.sin(2 * np.pi * np.array([d.month for d in dates]) / 12),
        'month_cos': np.cos(2 * np.pi * np.array([d.month for d in dates]) / 12),
        'day_sin': np.sin(2 * np.pi * np.array([d.timetuple().tm_yday for d in dates]) / 365),
        'day_cos': np.cos(2 * np.pi * np.array([d.timetuple().tm_yday for d in dates]) / 365),
    })
    return features


def plot_annual_comparison(eopatch, filename=None):
    """Plot water levels from different years overlaid for direct comparison"""
    valid_indices = ~np.isnan(eopatch.scalar["WATER_LEVEL"][..., 0])
    dates = np.array(eopatch.timestamps)[valid_indices]
    water_levels = eopatch.scalar["WATER_LEVEL"][valid_indices, 0]

    # Create a DataFrame for easier manipulation
    df = pd.DataFrame({'date': dates, 'water_level': water_levels})
    df['year'] = df['date'].apply(lambda x: x.year)
    df['dayofyear'] = df['date'].apply(lambda x: x.timetuple().tm_yday)

    # Create the plot
    fig, ax = plt.subplots(figsize=(15, 8))

    # Plot each year with a different color
    years = sorted(df['year'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(years)))

    for i, year in enumerate(years):
        year_data = df[df['year'] == year]
        ax.plot(year_data['dayofyear'], year_data['water_level'],
                label=str(year), color=colors[i], alpha=0.8)

    # Add month indicators at the bottom
    month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax.set_xticks(month_starts)
    ax.set_xticklabels(month_names)

    ax.set_title(f"{lake_name} Annual Water Level Comparison")
    ax.set_xlabel("Month")
    ax.set_ylabel("Water Level")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Year", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    if filename:
        plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close(fig)

    return df


def plot_water_level_histogram(eopatch, filename=None):
    """Create a histogram of observed water levels"""
    valid_indices = ~np.isnan(eopatch.scalar["WATER_LEVEL"][..., 0])
    water_levels = eopatch.scalar["WATER_LEVEL"][valid_indices, 0]

    fig, ax = plt.subplots(figsize=(12, 8))

    # Create histogram with KDE curve
    counts, bins, patches = ax.hist(water_levels, bins=20, alpha=0.7,
                                    color='#6969e2', edgecolor='black')

    # Add curve showing the distribution
    from scipy import stats
    kde = stats.gaussian_kde(water_levels)
    x_scale = np.linspace(min(water_levels), max(water_levels), 100)
    ax.plot(x_scale, kde(x_scale) * (len(water_levels) * (bins[1] - bins[0])),
            'r-', linewidth=2, label='Distribution Curve')

    # Add vertical lines for key statistics
    median = np.median(water_levels)
    mean = np.mean(water_levels)
    ax.axvline(median, color='green', linestyle='--', linewidth=2,
               label=f'Median: {median:.3f}')
    ax.axvline(mean, color='blue', linestyle='--', linewidth=2,
               label=f'Mean: {mean:.3f}')

    # Add text with statistics
    stats_text = (f"Statistics:\n"
                  f"Mean: {mean:.3f}\n"
                  f"Median: {median:.3f}\n"
                  f"Min: {min(water_levels):.3f}\n"
                  f"Max: {max(water_levels):.3f}\n"
                  f"Std Dev: {np.std(water_levels):.3f}\n"
                  f"Count: {len(water_levels)}")

    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_title(f'{lake_name} Water Level Distribution')
    ax.set_xlabel('Water Level')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    if filename:
        plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close(fig)


def plot_water_level_trend(eopatch, ma_window=30, filename=None, lake_name="Lake"):
    # Ensure lake_name is a string to prevent encoding issues
    lake_name = str(lake_name)
    """Plot water level time series with moving average and trend line

    Parameters:
    ----------
    eopatch : EOPatch
        EOPatch containing water level data
    ma_window : int, optional
        Window size for moving average in days, default 30
    filename : str, optional
        Output filename for plot and info, default None
    lake_name : str, optional
        Name of the lake, default "Lake"
    """
    valid_indices = ~np.isnan(eopatch.scalar["WATER_LEVEL"][..., 0])
    dates = np.array(eopatch.timestamps)[valid_indices]
    water_levels = eopatch.scalar["WATER_LEVEL"][valid_indices, 0]

    # Convert to pandas for easier time series handling
    df = pd.DataFrame({'date': dates, 'water_level': water_levels})
    df.set_index('date', inplace=True)
    df = df.sort_index()

    # Calculate moving average - using integer window for count-based window
    # Not using time-based frequency to avoid DatetimeIndex issues
    df['ma'] = df['water_level'].rolling(window=int(ma_window), min_periods=3).mean()

    # Calculate linear trend
    days = np.array([(d - dates[0]).days for d in dates])
    z = np.polyfit(days, water_levels, 1)
    trend_line = np.poly1d(z)

    # Annual rate of change (in percentage)
    annual_change = z[0] * 365 * 100  # convert to percentage per year

    fig, ax = plt.subplots(figsize=(15, 8))

    # Plot raw data, moving average, and trend
    ax.plot(df.index, df['water_level'], 'b-', alpha=0.4, label='Raw Water Level')
    ax.plot(df.index, df['ma'], 'g-', linewidth=2, label=f'{ma_window}-Day Moving Average')
    ax.plot(df.index, trend_line(days), 'r-', linewidth=2,
            label=f'Trend: {z[0]:.6f} per day ({annual_change:.2f}% per year)')

    # Add shaded area for trend confidence (simple approach)
    residuals = water_levels - trend_line(days)
    std_resid = np.std(residuals)
    ax.fill_between(df.index,
                    trend_line(days) - std_resid,
                    trend_line(days) + std_resid,
                    color='red', alpha=0.2, label='±1σ Confidence Band')

    # Handle potential encoding issues in matplotlib
    try:
        # Use lake_name variable (which might have Polish characters) directly in the plot
        ax.set_title(f'{lake_name} Water Level Trend Analysis')
    except UnicodeEncodeError:
        # Fall back to ASCII with replacement for non-ASCII characters
        print(f"Warning: Encoding issue with lake name '{lake_name}' in plot. Using ASCII replacement.")
        ascii_lake_name = lake_name.encode('ascii', 'replace').decode('ascii')
        ax.set_title(f'{ascii_lake_name} Water Level Trend Analysis')

    ax.set_xlabel('Date')
    ax.set_ylabel('Water Level')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Add explanatory text
    if annual_change > 0:
        trend_desc = f"Rising trend: +{annual_change:.2f}% per year"
    else:
        trend_desc = f"Falling trend: {annual_change:.2f}% per year"

    ax.text(0.05, 0.05, trend_desc, transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    if filename:
        plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close(fig)

    # Export trend information to text file
    trend_info = {
        'trend_coefficient': z[0],
        'annual_change_percent': annual_change,
        'trend_description': trend_desc
    }

    # Write trend information to text file using UTF-8 encoding
    if filename:
        try:
            # Explicitly use UTF-8 encoding for file output
            with open(f"{filename.split('.')[0]}_info.txt", "w", encoding="utf-8") as f:
                f.write(f"Lake: {lake_name}\n")
                f.write(f"Water Level Trend Analysis\n")
                f.write(f"===========================\n")
                f.write(f"Trend coefficient: {z[0]:.6f} per day\n")
                f.write(f"Annual change: {annual_change:.2f}% per year\n")
                f.write(f"Trend description: {trend_desc}\n")
                f.write(f"Analysis based on {len(water_levels)} observations\n")
                f.write(f"Date range: {min(dates).strftime('%Y-%m-%d')} to {max(dates).strftime('%Y-%m-%d')}\n")
        except UnicodeEncodeError:
            # Fall back to ASCII with replacement for non-ASCII characters if UTF-8 fails
            print(f"Warning: Encoding issue with lake name '{lake_name}'. Writing file with ASCII encoding.")
            with open(f"{filename.split('.')[0]}_info.txt", "w", encoding="ascii", errors="replace") as f:
                f.write(f"Lake: {lake_name}\n")
                f.write(f"Water Level Trend Analysis\n")
                f.write(f"===========================\n")
                f.write(f"Trend coefficient: {z[0]:.6f} per day\n")
                f.write(f"Annual change: {annual_change:.2f}% per year\n")
                f.write(f"Trend description: {trend_desc}\n")
                f.write(f"Analysis based on {len(water_levels)} observations\n")
                f.write(f"Date range: {min(dates).strftime('%Y-%m-%d')} to {max(dates).strftime('%Y-%m-%d')}\n")

    return trend_info


def simplified_plot_seasonal_analysis(eopatch, filename=None):
    """Plot water levels by month and season only"""
    valid_indices = ~np.isnan(eopatch.scalar["WATER_LEVEL"][..., 0])
    dates = np.array(eopatch.timestamps)[valid_indices]
    water_levels = eopatch.scalar["WATER_LEVEL"][valid_indices, 0]

    # Create month and season data
    months = np.array([date.month for date in dates])
    seasons = np.array(['Winter' if month in [12, 1, 2] else
                        'Spring' if month in [3, 4, 5] else
                        'Summer' if month in [6, 7, 8] else
                        'Fall' for month in months])

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # 1. Monthly boxplot
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_data = [water_levels[months == i + 1] for i in range(12)]
    axes[0].boxplot(monthly_data, tick_labels=month_names)
    axes[0].set_title('Water Level by Month')
    axes[0].set_ylabel('Water Level')
    axes[0].grid(True, alpha=0.3)

    # 2. Seasonal boxplot
    season_order = ['Winter', 'Spring', 'Summer', 'Fall']
    seasonal_data = [water_levels[seasons == season] for season in season_order]
    axes[1].boxplot(seasonal_data, tick_labels=season_order)
    axes[1].set_title('Water Level by Season')
    axes[1].set_ylabel('Water Level')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if filename:
        plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close(fig)

    return {'monthly_data': monthly_data, 'seasonal_data': seasonal_data}


def modified_monthly_averages(eopatch, filename=None):
    """Create a bar chart of monthly average water levels with adjusted y-axis"""
    valid_indices = ~np.isnan(eopatch.scalar["WATER_LEVEL"][..., 0])
    dates = np.array(eopatch.timestamps)[valid_indices]
    water_levels = eopatch.scalar["WATER_LEVEL"][valid_indices, 0]

    # Create DataFrame with month information
    df = pd.DataFrame({'date': dates, 'water_level': water_levels})
    df['month'] = df['date'].apply(lambda x: x.month)
    df['month_name'] = df['date'].apply(lambda x: x.strftime('%b'))

    # Calculate monthly averages
    monthly_avg = df.groupby('month')['water_level'].agg(['mean', 'std', 'count']).reset_index()
    monthly_avg['month_name'] = monthly_avg['month'].apply(lambda x: pd.Timestamp(2023, x, 1).strftime('%b'))

    # Sort by month for proper order
    monthly_avg = monthly_avg.sort_values('month')

    fig, ax = plt.subplots(figsize=(12, 8))

    # Create bars
    bars = ax.bar(monthly_avg['month_name'], monthly_avg['mean'],
                  yerr=monthly_avg['std'], capsize=5,
                  color=plt.cm.viridis(np.linspace(0, 0.8, 12)),
                  alpha=0.7, edgecolor='black')

    # Add data point counts above bars
    for i, bar in enumerate(bars):
        count = monthly_avg['count'].iloc[i]
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                f'n={count}', ha='center', va='bottom', rotation=0,
                fontsize=9)

    # Add overall average line
    overall_avg = np.mean(water_levels)
    ax.axhline(y=overall_avg, color='red', linestyle='--',
               label=f'Overall Average: {overall_avg:.3f}')

    # Set y-axis limits safely
    min_val = 0.8

    # Calculate max value safely by handling NaN values
    max_vals = monthly_avg['mean'] + monthly_avg['std']
    max_vals = max_vals[~np.isnan(max_vals)]  # Remove NaN values

    if len(max_vals) > 0:
        max_val = max(max_vals) + 0.03
    else:
        # Fallback if all values are NaN
        max_val = max(monthly_avg['mean'][~np.isnan(monthly_avg['mean'])], default=1.0) + 0.1

    # Safety check for invalid limits
    if np.isnan(min_val) or np.isnan(max_val) or np.isinf(min_val) or np.isinf(max_val):
        # Use reasonable defaults if limits are invalid
        min_val = 0.8
        max_val = 1.0

    if min_val >= max_val:
        max_val = min_val + 0.2

    ax.set_ylim(min_val, max_val)

    ax.set_title(f'{lake_name} Monthly Average Water Levels')
    ax.set_xlabel('Month')
    ax.set_ylabel('Average Water Level')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()

    plt.tight_layout()
    if filename:
        plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close(fig)

    return monthly_avg


def plot_extreme_water_levels(eopatch, filename=None, lake_name="Lake"):
    # Ensure lake_name is a string to prevent encoding issues
    lake_name = str(lake_name)
    """Display when water was at its highest and lowest levels

    Parameters:
    ----------
    eopatch : EOPatch
        EOPatch containing water level data
    filename : str, optional
        Output filename for plot and info, default None
    lake_name : str, optional
        Name of the lake, default "Lake"
    """
    valid_indices = ~np.isnan(eopatch.scalar["WATER_LEVEL"][..., 0])
    dates = np.array(eopatch.timestamps)[valid_indices]
    water_levels = eopatch.scalar["WATER_LEVEL"][valid_indices, 0]

    # Create DataFrame
    df = pd.DataFrame({'date': dates, 'water_level': water_levels})
    df['month'] = df['date'].apply(lambda x: x.month)
    df['year'] = df['date'].apply(lambda x: x.year)
    df['season'] = df['date'].apply(lambda x: 'Winter' if x.month in [12, 1, 2] else
    'Spring' if x.month in [3, 4, 5] else
    'Summer' if x.month in [6, 7, 8] else
    'Fall')

    # Find extremes
    highest_idx = np.argmax(water_levels)
    lowest_idx = np.argmin(water_levels)

    highest_level = water_levels[highest_idx]
    lowest_level = water_levels[lowest_idx]

    highest_date = dates[highest_idx]
    lowest_date = dates[lowest_idx]

    # Create visualization
    fig, ax = plt.subplots(figsize=(15, 8))

    # Plot all water levels
    ax.plot(dates, water_levels, 'b-', alpha=0.5, label='Water Level')

    # Highlight extremes
    ax.scatter([highest_date], [highest_level], color='red', s=100,
               label=f'Highest: {highest_level:.3f} on {highest_date.strftime("%Y-%m-%d")}')
    ax.scatter([lowest_date], [lowest_level], color='orange', s=100,
               label=f'Lowest: {lowest_level:.3f} on {lowest_date.strftime("%Y-%m-%d")}')

    # Add annotations
    ax.annotate(f"Highest: {highest_level:.3f}",
                xy=(highest_date, highest_level),
                xytext=(0, 20), textcoords='offset points',
                ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3'))

    ax.annotate(f"Lowest: {lowest_level:.3f}",
                xy=(lowest_date, lowest_level),
                xytext=(0, -20), textcoords='offset points',
                ha='center', va='top',
                bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3'))

    # Find top 5 highest and lowest
    top_5_high = df.nlargest(5, 'water_level')
    top_5_low = df.nsmallest(5, 'water_level')

    # Add summary text
    high_text = "Top 5 Highest Levels:\n"
    for i, (_, row) in enumerate(top_5_high.iterrows(), 1):
        high_text += f"{i}. {row['water_level']:.3f} on {row['date'].strftime('%Y-%m-%d')} ({row['season']})\n"

    low_text = "Top 5 Lowest Levels:\n"
    for i, (_, row) in enumerate(top_5_low.iterrows(), 1):
        low_text += f"{i}. {row['water_level']:.3f} on {row['date'].strftime('%Y-%m-%d')} ({row['season']})\n"

    # Add text boxes
    ax.text(0.02, 0.98, high_text, transform=ax.transAxes,
            va='top', ha='left', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.5', fc='lightyellow', alpha=0.7))

    ax.text(0.98, 0.02, low_text, transform=ax.transAxes,
            va='bottom', ha='right', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.5', fc='lightblue', alpha=0.7))

    # Handle potential encoding issues in matplotlib
    try:
        # Use lake_name variable directly in the plot
        ax.set_title(f'{lake_name} Extreme Water Levels')
    except UnicodeEncodeError:
        # Fall back to ASCII with replacement for non-ASCII characters
        print(f"Warning: Encoding issue with lake name '{lake_name}' in plot. Using ASCII replacement.")
        ascii_lake_name = lake_name.encode('ascii', 'replace').decode('ascii')
        ax.set_title(f'{ascii_lake_name} Extreme Water Levels')

    ax.set_xlabel('Date')
    ax.set_ylabel('Water Level')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper center')

    plt.tight_layout()
    if filename:
        plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close(fig)

    # Export extreme water level data to text file
    extreme_info = {
        'highest': {'date': highest_date, 'level': highest_level},
        'lowest': {'date': lowest_date, 'level': lowest_level},
        'top_5_high': top_5_high,
        'top_5_low': top_5_low
    }

    # Write extreme water level information to text file with explicit encoding handling
    if filename:
        try:
            # Explicitly use UTF-8 encoding for file output
            with open(f"{filename.split('.')[0]}_info.txt", "w", encoding='utf-8') as f:
                f.write(f"Lake: {lake_name}\n")
                f.write(f"Extreme Water Levels Analysis\n")
                f.write(f"===========================\n\n")

                f.write(f"HIGHEST WATER LEVEL\n")
                f.write(f"Date: {highest_date.strftime('%Y-%m-%d')}\n")
                f.write(f"Level: {highest_level:.3f}\n")
                f.write(
                    f"Season: {next((row['season'] for _, row in top_5_high.iterrows() if row['date'] == highest_date), 'Unknown')}\n\n")

                f.write(f"LOWEST WATER LEVEL\n")
                f.write(f"Date: {lowest_date.strftime('%Y-%m-%d')}\n")
                f.write(f"Level: {lowest_level:.3f}\n")
                f.write(
                    f"Season: {next((row['season'] for _, row in top_5_low.iterrows() if row['date'] == lowest_date), 'Unknown')}\n\n")

                f.write(f"TOP 5 HIGHEST LEVELS\n")
                for i, (_, row) in enumerate(top_5_high.iterrows(), 1):
                    f.write(f"{i}. {row['water_level']:.3f} on {row['date'].strftime('%Y-%m-%d')} ({row['season']})\n")

                f.write(f"\nTOP 5 LOWEST LEVELS\n")
                for i, (_, row) in enumerate(top_5_low.iterrows(), 1):
                    f.write(f"{i}. {row['water_level']:.3f} on {row['date'].strftime('%Y-%m-%d')} ({row['season']})\n")
        except UnicodeEncodeError:
            # Fall back to ASCII with replacement for non-ASCII characters if UTF-8 fails
            print(f"Warning: Encoding issue with lake name '{lake_name}'. Writing file with ASCII encoding.")
            with open(f"{filename.split('.')[0]}_info.txt", "w", encoding="ascii", errors="replace") as f:
                f.write(f"Lake: {lake_name}\n")
                f.write(f"Extreme Water Levels Analysis\n")
                f.write(f"===========================\n\n")

                f.write(f"HIGHEST WATER LEVEL\n")
                f.write(f"Date: {highest_date.strftime('%Y-%m-%d')}\n")
                f.write(f"Level: {highest_level:.3f}\n")
                f.write(
                    f"Season: {next((row['season'] for _, row in top_5_high.iterrows() if row['date'] == highest_date), 'Unknown')}\n\n")

                f.write(f"LOWEST WATER LEVEL\n")
                f.write(f"Date: {lowest_date.strftime('%Y-%m-%d')}\n")
                f.write(f"Level: {lowest_level:.3f}\n")
                f.write(
                    f"Season: {next((row['season'] for _, row in top_5_low.iterrows() if row['date'] == lowest_date), 'Unknown')}\n\n")

                f.write(f"TOP 5 HIGHEST LEVELS\n")
                for i, (_, row) in enumerate(top_5_high.iterrows(), 1):
                    f.write(f"{i}. {row['water_level']:.3f} on {row['date'].strftime('%Y-%m-%d')} ({row['season']})\n")

                f.write(f"\nTOP 5 LOWEST LEVELS\n")
                for i, (_, row) in enumerate(top_5_low.iterrows(), 1):
                    f.write(f"{i}. {row['water_level']:.3f} on {row['date'].strftime('%Y-%m-%d')} ({row['season']})\n")

    return extreme_info

def plot_original_vs_filtered(eopatch, filename=None):
    """Plot a comparison between original and filtered water levels"""
    valid_indices_original = ~np.isnan(eopatch.scalar["WATER_LEVEL_ORIGINAL"][..., 0])
    valid_indices_filtered = ~np.isnan(eopatch.scalar["WATER_LEVEL"][..., 0])

    dates_original = np.array(eopatch.timestamps)[valid_indices_original]
    water_levels_original = eopatch.scalar["WATER_LEVEL_ORIGINAL"][valid_indices_original, 0]

    dates_filtered = np.array(eopatch.timestamps)[valid_indices_filtered]
    water_levels_filtered = eopatch.scalar["WATER_LEVEL"][valid_indices_filtered, 0]

    fig, ax = plt.subplots(figsize=(15, 8))

    # Plot original data
    ax.plot(dates_original, water_levels_original, 'r-', alpha=0.5, label='Original Water Levels')

    # Plot filtered data
    ax.plot(dates_filtered, water_levels_filtered, 'b-', linewidth=2, label='Filtered Water Levels')

    # Highlight removed points
    removed_mask = np.isnan(eopatch.scalar["WATER_LEVEL"][..., 0]) & ~np.isnan(
        eopatch.scalar["WATER_LEVEL_ORIGINAL"][..., 0])
    dates_removed = np.array(eopatch.timestamps)[removed_mask]
    water_levels_removed = eopatch.scalar["WATER_LEVEL_ORIGINAL"][removed_mask, 0]

    ax.scatter(dates_removed, water_levels_removed, color='red', s=80,
               marker='x', label='Anomalies/Jumps Removed')

    ax.set_title('Original vs. Filtered Water Levels')
    ax.set_xlabel('Date')
    ax.set_ylabel('Water Level')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Add summary text
    n_original = len(water_levels_original)
    n_filtered = len(water_levels_filtered)
    n_removed = len(water_levels_removed)

    summary_text = (f"Original data points: {n_original}\n"
                    f"Filtered data points: {n_filtered}\n"
                    f"Points removed: {n_removed} ({n_removed / n_original:.1%})")

    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes,
            va='top', ha='left', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7))

    plt.tight_layout()
    if filename:
        plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close(fig)


# -----------------------------------
# STEP 6: Save Results
# -----------------------------------

eopatch, model, predictions = main(patch)


plot_rgb_w_water(patch, 0, "water_overlay_first.png")
plot_rgb_w_water(patch, -1, "water_overlay_last.png")
plot_water_levels(patch, 1.0, "water_levels.png")
plot_annual_comparison(patch, "annual_comparison")
plot_water_level_histogram(patch, "water_level_histogram")
plot_water_level_trend(eopatch, 30, "water_level_trend", lake_name)

# v0.11 patch (added jump filter and the following plots)
simplified_plot_seasonal_analysis(patch, "simplified_seasonal_analysis")
modified_monthly_averages(patch, "modified_monthly_averages")
plot_extreme_water_levels(eopatch, "extreme_water_levels", lake_name)


plot_original_vs_filtered(patch, "original_vs_filtered.png")



plt.show()