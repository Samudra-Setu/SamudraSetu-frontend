import streamlit as st
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Page config
st.set_page_config(
    page_title="SAMUDRA SETU - Oil Spill Detection & Vessel Attribution",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1e3a8a;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .warning-box {
        background-color: #fef3c7;
        border: 1px solid #f59e0b;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d1fae5;
        border: 1px solid #10b981;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #dbeafe;
        border: 1px solid #3b82f6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 0.5rem;
        padding: 1rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1e293b;
        border-bottom: 2px solid #3b82f6;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .warning-text {
        color: #7A3E32;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'ais_data' not in st.session_state:
    st.session_state.ais_data = None
if 'ais_stats' not in st.session_state:
    st.session_state.ais_stats = None
if 'classification_result' not in st.session_state:
    st.session_state.classification_result = None
if 'segmentation_result' not in st.session_state:
    st.session_state.segmentation_result = None
if 'geospatial_result' not in st.session_state:
    st.session_state.geospatial_result = None
if 'drift_result' not in st.session_state:
    st.session_state.drift_result = None
if 'attribution_result' not in st.session_state:
    st.session_state.attribution_result = None

# Import modules
try:
    import torch
    import torch.nn.functional as F
    import numpy as np
    import pandas as pd
    from PIL import Image
    from datetime import datetime, timezone, timedelta
    import folium
    from streamlit_folium import st_folium
    import plotly.express as px
    import plotly.graph_objects as go
    
    from src.prediction.predict import predict_image
    from src.models.classifier import load_model, create_model
    from src.data.dataset import create_inference_loader, get_transforms
    from src.segmentation.base import create_segmenter, PrototypeSegmenter
    from src.segmentation.pipeline import SegmentationGeospatialPipeline, create_pipeline
    from src.geospatial.pipeline import analyze_spills, format_spill_report, segmentation_to_geospatial
    from src.ais.loader import load_ais_data, load_ais_data_auto, get_ais_summary, log_ais_debug_info
    from src.ais.proximity import find_vessels_near_spill, format_proximity_report, get_vessel_trajectory
    from src.drift.prototype import create_prototype_model, PrototypeConstantCurrentModel
    from src.drift.ais_integration import run_drift_ais_analysis, format_drift_ais_report
    from src.attribution.attribution import run_attribution_analysis, format_attribution_report
    from src.attribution.models import AttributionWeights
    from src.data.geotiff_loader import load_geotiff_for_pipeline, GeoTIFFLoadError, check_geotiff_georeferencing
    
    MODULES_LOADED = True
except Exception as e:
    MODULES_LOADED = False
    st.error(f"Failed to import modules: {e}")

# Helper functions
@st.cache_resource
def load_classification_model(model_path, model_name="mobilenet_v3_small", image_size=128):
    """Load the classification model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, num_classes=2, model_name=model_name, device=device)
    return model, device, image_size

@st.cache_resource
def load_segmentation_pipeline(classifier_path, classifier_model="mobilenet_v3_small", image_size=128, pixel_size_m=10.0, bounds=None, prototype_method="sar_threshold", require_georeferencing=True):
    """Load the segmentation pipeline."""
    return create_pipeline(
        classifier_path=classifier_path,
        classifier_model=classifier_model,
        image_size=image_size,
        pixel_size_m=pixel_size_m,
        bounds=bounds,
        prototype_method=prototype_method,
        require_georeferencing=require_georeferencing
    )

@st.cache_data
def load_ais_data_cached(filepath):
    """Load AIS data with caching - supports file or directory."""
    return load_ais_data_auto(filepath, validate=True)


@st.cache_data
def load_ais_data_with_debug(filepath):
    """Load AIS data with debug info."""
    df, stats = load_ais_data_auto(filepath, validate=True)
    debug_info = log_ais_debug_info(df, label=f"AIS Data from {filepath}")
    return df, stats, debug_info

# Main app
def main():
    # Header
    st.image("assets/samudra_setu_logo.jpg", width=400)
    st.markdown('<div class="sub-header">AI-Based Oil Spill Detection & Vessel Attribution System</div>', unsafe_allow_html=True)
    
    # Scientific disclaimer
    st.markdown("""
    <div class="warning-box">
        <span class="warning-text">⚠️ SCIENTIFIC DISCLAIMER:</span> The vessel attribution output is a model-based candidate ranking 
        intended for research and demonstration. It is <strong>not proof of legal responsibility or causation</strong>. 
        Prototype ocean-drift and vessel-type assumptions have not been scientifically validated.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model paths
        st.subheader("Model Paths")
        model_path = st.text_input("Classification Model", value="models/best_model.pth")
        model_name = st.selectbox("Model Architecture", 
            ["mobilenet_v3_small", "resnet18", "resnet34", "efficientnet_b0"],
            index=0)
        image_size = st.number_input("Image Size", value=128, min_value=64, max_value=512, step=32)
        
        st.subheader("AIS Data")
        ais_path = st.text_input("AIS CSV File or Directory", value="data/raw/ais", 
                                 help="Path to a single CSV file or directory containing multiple CSV files")
        ais_load_mode = st.radio("Load Mode", ["Auto-detect (file or directory)", "Single file", "Directory"], index=0)
        
        st.subheader("Drift Parameters")
        current_u = st.number_input("Current U (Eastward m/s)", value=0.1, step=0.01, format="%.2f")
        current_v = st.number_input("Current V (Northward m/s)", value=0.05, step=0.01, format="%.2f")
        drift_duration = st.number_input("Drift Duration (hours)", value=6, min_value=1, max_value=72)
        drift_step = st.number_input("Drift Time Step (hours)", value=1.0, step=0.5, min_value=0.5)
        
        st.subheader("Search Parameters")
        search_radius = st.number_input("Search Radius (km)", value=50, min_value=1, max_value=500)
        time_window = st.number_input("Time Window (hours)", value=6.0, step=1.0, min_value=0.5)
        
        st.subheader("Attribution Weights")
        w_distance = st.slider("Distance Weight", 0.0, 1.0, 0.30, 0.05)
        w_temporal = st.slider("Temporal Weight", 0.0, 1.0, 0.25, 0.05)
        w_drift = st.slider("Drift/Trajectory Weight", 0.0, 1.0, 0.30, 0.05)
        w_vessel_type = st.slider("Vessel Type Weight", 0.0, 1.0, 0.10, 0.05)
        w_data_quality = st.slider("Data Quality Weight", 0.0, 1.0, 0.05, 0.05)
        
        # Normalize weights
        total_weight = w_distance + w_temporal + w_drift + w_vessel_type + w_data_quality
        if total_weight > 0:
            weights = {
                'distance': w_distance / total_weight,
                'temporal': w_temporal / total_weight,
                'drift': w_drift / total_weight,
                'vessel_type': w_vessel_type / total_weight,
                'data_quality': w_data_quality / total_weight,
            }
        else:
            weights = {'distance': 0.3, 'temporal': 0.25, 'drift': 0.3, 'vessel_type': 0.1, 'data_quality': 0.05}
        
        st.markdown("---")
        st.subheader("Spill Information (Optional)")
        spill_lat = st.number_input("Spill Latitude", value=12.5, format="%.6f")
        spill_lon = st.number_input("Spill Longitude", value=72.4, format="%.6f")
        spill_time_str = st.text_input("Spill Time (ISO format)", value="2022-01-15T06:30:00Z")
        
        try:
            spill_time = datetime.fromisoformat(spill_time_str.replace('Z', '+00:00'))
            if spill_time.tzinfo is None:
                spill_time = spill_time.replace(tzinfo=timezone.utc)
        except:
            spill_time = datetime(2022, 1, 15, 6, 30, tzinfo=timezone.utc)
        
        st.subheader("Geospatial Bounds (Optional)")
        bounds_min_lon = st.number_input("Min Lon", value=70.0, format="%.2f")
        bounds_min_lat = st.number_input("Min Lat", value=10.0, format="%.2f")
        bounds_max_lon = st.number_input("Max Lon", value=71.0, format="%.2f")
        bounds_max_lat = st.number_input("Max Lat", value=11.0, format="%.2f")
        bounds = (bounds_min_lon, bounds_min_lat, bounds_max_lon, bounds_max_lat) if st.checkbox("Use Bounds") else None
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📥 Input", 
        "🔍 Detection", 
        "🗺️ Localization", 
        "🌊 Drift Backtracking", 
        "🚢 AIS Search", 
        "🎯 Attribution", 
        "🗺️ Map"
    ])
    
    # Tab 1: Input
    with tab1:
        st.markdown('<div class="section-header">📥 Input</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Upload SAR Image")
            uploaded_file = st.file_uploader(
                "Choose a SAR image (JPG/PNG/TIFF)",
                type=["jpg", "jpeg", "png", "tif", "tiff"]
            )
            
            if uploaded_file is not None:
                # Save uploaded file to temp
                temp_path = f"temp_upload{Path(uploaded_file.name).suffix}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Check georeferencing for GeoTIFF
                georef_info = None
                is_geotiff = Path(uploaded_file.name).suffix.lower() in ('.tif', '.tiff')
                
                if is_geotiff:
                    georef_info = check_geotiff_georeferencing(temp_path)
                    if georef_info.get('is_georeferenced'):
                        st.success(f"✅ GeoTIFF is georeferenced: CRS={georef_info['crs']}, Transform valid")
                    else:
                        st.warning(f"⚠️ GeoTIFF is NOT georeferenced: CRS={georef_info.get('crs', 'None')}, Identity transform={georef_info.get('is_identity_transform', 'Unknown')}")
                        st.info("For drift tracking, you need a properly georeferenced Sentinel-1 GeoTIFF from the original .SAFE product.")
                
                # Load image for display (PIL)
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption="Uploaded SAR Image", use_container_width=True)
                
                st.session_state.temp_image_path = temp_path
                st.session_state.image = image
                st.session_state.georef_info = georef_info
                st.session_state.is_geotiff = is_geotiff
                
                st.success(f"Image loaded: {image.size[0]}x{image.size[1]} pixels")
                
                # Show image info
                st.markdown("**Image Information:**")
                st.write(f"- Dimensions: {image.size[0]} x {image.size[1]}")
                st.write(f"- Mode: {image.mode}")
                st.write(f"- Format: {uploaded_file.type}")
                if georef_info:
                    st.write(f"- Georeferenced: {'✅ Yes' if georef_info.get('is_georeferenced') else '❌ No'}")
                    st.write(f"- CRS: {georef_info.get('crs', 'None')}")
                    st.write(f"- Bounds (WGS84): {georef_info.get('bounds', 'N/A')}")
        
        with col2:
            st.markdown("### Sample Images")
            
            # Load sample images
            sample_class1 = "data/raw/archive/kaggle/data/Class_1/class_1_00001.jpg"
            sample_class0 = "data/raw/archive/kaggle/data/Class_0/class_0_00001.jpg"
            
            col_a, col_b = st.columns(2)
            with col_a:
                if os.path.exists(sample_class1):
                    sample_img = Image.open(sample_class1).convert("RGB")
                    st.image(sample_img, caption="Oil Spill Sample (Class 1)", use_container_width=True)
                    if st.button("Use Oil Spill Sample"):
                        st.session_state.temp_image_path = sample_class1
                        st.session_state.image = Image.open(sample_class1).convert("RGB")
                        st.rerun()
            
            with col_b:
                if os.path.exists(sample_class0):
                    sample_img = Image.open(sample_class0).convert("RGB")
                    st.image(sample_img, caption="Clean Sea Sample (Class 0)", use_container_width=True)
                    if st.button("Use Clean Sea Sample"):
                        st.session_state.temp_image_path = sample_class0
                        st.session_state.image = Image.open(sample_class0).convert("RGB")
                        st.rerun()
            
            # Spill info display
            st.markdown("### Spill Information")
            if 'spill_time' in locals():
                st.info(f"""
                **Spill Location:** ({spill_lat:.6f}, {spill_lon:.6f})  
                **Detection Time:** {spill_time.isoformat()}  
                **Geospatial Bounds:** {bounds if bounds else 'Not set'}
                """)
    
    # Tab 2: Detection
    with tab2:
        st.markdown('<div class="section-header">🔍 Oil Spill Detection</div>', unsafe_allow_html=True)
        
        if 'temp_image_path' not in st.session_state:
            st.warning("Please upload or select an image in the Input tab first.")
        else:
            if st.button("🔍 Run Oil Spill Detection", type="primary", use_container_width=True):
                with st.spinner("Running oil spill classification..."):
                    try:
                        # Load model
                        model, device, img_size = load_classification_model(model_path, model_name, image_size)
                        
                        # Run prediction
                        result = predict_image(
                            model_path=model_path,
                            image_path=st.session_state.temp_image_path,
                            model_name=model_name,
                            image_size=image_size
                        )
                        
                        st.session_state.classification_result = result
                        st.session_state.model = model
                        st.session_state.device = device
                        st.session_state.image_size = image_size
                        
                        st.success("Detection completed!")
                    except Exception as e:
                        st.error(f"Detection failed: {e}")
            
            # Display results
            if st.session_state.classification_result:
                result = st.session_state.classification_result
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Prediction", result['class_name'])
                with col2:
                    st.metric("Confidence", f"{result['confidence']:.1%}")
                with col3:
                    st.metric("Class Index", result['class_index'])
                
                # Probability bar chart
                probs = result['all_probabilities']
                prob_df = pd.DataFrame(list(probs.items()), columns=['Class', 'Probability'])
                fig = px.bar(prob_df, x='Class', y='Probability', 
                            title="Class Probabilities",
                            color='Class',
                            color_discrete_map={'OIL_SPILL': '#ef4444', 'NO_OIL_SPILL': '#10b981'})
                fig.update_layout(yaxis_range=[0, 1])
                st.plotly_chart(fig, use_container_width=True)
                
                # Raw probabilities
                with st.expander("Raw Probabilities"):
                    for cls, prob in result['all_probabilities'].items():
                        st.write(f"**{cls}**: {prob:.4f}")
    
    # Tab 3: Localization
    with tab3:
        st.markdown('<div class="section-header">🗺️ Spill Localization & Geospatial Analysis</div>', unsafe_allow_html=True)
        
        if 'temp_image_path' not in st.session_state:
            st.warning("Please upload or select an image in the Input tab first.")
        elif not MODULES_LOADED:
            st.error("Modules not loaded. Please check installation.")
        else:
            # Segmentation options
            col1, col2 = st.columns([1, 1])
            with col1:
                prototype_method = st.selectbox("Segmentation Method", 
                    ["sar_threshold", "classifier_sliding"],
                    format_func=lambda x: "SAR Threshold (Fast)" if x == "sar_threshold" else "Classifier Sliding Window (Slow)")
            with col2:
                pixel_size_m = st.number_input("Pixel Size (meters/pixel)", value=10.0, step=1.0)
                min_area = st.number_input("Min Area (pixels)", value=10, min_value=1)
            
            # Georeferencing option
            require_geo = st.checkbox("Require valid georeferencing (fail if missing)", value=True, 
                                        help="If checked, processing will fail with clear error if input lacks CRS/geotransform")
            
            if st.button("🗺️ Run Segmentation & Geospatial Analysis", type="primary", use_container_width=True):
                with st.spinner("Running segmentation and geospatial analysis..."):
                    try:
                        # Create pipeline
                        pipeline = create_pipeline(
                            classifier_path=model_path,
                            classifier_model=model_name,
                            image_size=image_size,
                            pixel_size_m=pixel_size_m,
                            bounds=bounds,
                            prototype_method=prototype_method,
                            require_georeferencing=require_geo
                        )
                        
                        # Process image using file path (handles GeoTIFF georeferencing automatically)
                        result = pipeline.process_from_file(
                            st.session_state.temp_image_path,
                            require_georeferencing=require_geo
                        )
                        
                        st.session_state.segmentation_result = result.segmentation
                        st.session_state.geospatial_result = result.geospatial
                        
                        # Show georeferencing status
                        if result.geospatial.spills:
                            first_spill = result.geospatial.spills[0]
                            if first_spill.centroid_lat is not None:
                                st.success(f"✅ Segmentation completed! Spill centroid: ({first_spill.centroid_lat:.6f}, {first_spill.centroid_lon:.6f})")
                            else:
                                st.warning("⚠️ Segmentation completed but NO GEOGRAPHIC COORDINATES - image not georeferenced. Drift tracking will not work.")
                        else:
                            st.success("Segmentation completed! No spills detected.")
                        
                    except GeoTIFFLoadError as e:
                        st.error(f"Georeferencing error: {e}")
                        st.info("""
                        **To fix this:**
                        1. Use the original Sentinel-1 .SAFE product (not a derived TIFF)
                        2. Or georeference using GDAL: `gdalwarp -t_srs EPSG:4326 input.tif output.tif`
                        3. Or provide manual bounds in the sidebar (Min/Max Lon/Lat)
                        4. Or uncheck 'Require valid georeferencing' for prototype testing
                        """)
                    except Exception as e:
                        st.error(f"Segmentation failed: {e}")
            
            # Display segmentation results
            if st.session_state.segmentation_result:
                seg_result = st.session_state.segmentation_result
                geo_result = st.session_state.geospatial_result
                
                # Prototype warning
                if "PrototypeSegmenter" in seg_result.metadata.get("model", ""):
                    st.markdown("""
                    <div class="warning-box">
                        <span class="warning-text">⚠️ PROTOTYPE LOCALIZATION:</span> 
                        Pixel-level annotated segmentation data not yet available. 
                        This mask is generated using a heuristic prototype method and is 
                        <strong>not a scientifically validated oil-spill boundary</strong>.
                    </div>
                    """, unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Segmentation Mask")
                    mask_img = Image.fromarray((seg_result.mask * 255).astype(np.uint8))
                    st.image(mask_img, caption="Binary Mask (White = Oil Spill)", use_container_width=True)
                    
                    if seg_result.confidence_map is not None:
                        conf_img = Image.fromarray((seg_result.confidence_map * 255).astype(np.uint8))
                        st.image(conf_img, caption="Confidence Map", use_container_width=True)
                
                with col2:
                    st.subheader("Overlay")
                    overlay = np.array(st.session_state.image).copy()
                    overlay[seg_result.mask == 1] = [255, 0, 0]  # Red overlay
                    overlay = (overlay * 0.7 + np.array(st.session_state.image) * 0.3).astype(np.uint8)
                    st.image(overlay, caption="Overlay (Red = Oil Spill)", use_container_width=True)
                    
                    # Statistics
                    st.metric("Spill Pixels", f"{seg_result.mask.sum():,}")
                    st.metric("Spill Coverage", f"{seg_result.mask.mean()*100:.1f}%")
                    if seg_result.confidence_map is not None:
                        st.metric("Mean Confidence", f"{seg_result.confidence_map[seg_result.mask > 0].mean():.3f}")
                
                # Geospatial results
                if st.session_state.geospatial_result:
                    geo_result = st.session_state.geospatial_result
                    st.markdown("### Geospatial Analysis")
                    
                    # Report
                    report = format_spill_report(geo_result)
                    st.code(report)
                    
                    # Debug info: show coordinate conversion details
                    if geo_result.spills:
                        spill = geo_result.spills[0]
                        st.markdown("### 🔍 Debug: Coordinate Conversion")
                        st.write(f"**Detected spill centroid (pixel):** ({spill.centroid_px[0]:.1f}, {spill.centroid_px[1]:.1f})")
                        
                        # Check what bounds were used
                        pipeline_bounds = None
                        if 'pipeline' in locals() and hasattr(pipeline, 'bounds') and pipeline.bounds:
                            pipeline_bounds = pipeline.bounds
                        elif bounds:
                            pipeline_bounds = bounds
                        
                        if pipeline_bounds:
                            st.write(f"**Manual bounds used:** Lon[{pipeline_bounds[0]:.4f}, {pipeline_bounds[2]:.4f}], Lat[{pipeline_bounds[1]:.4f}, {pipeline_bounds[3]:.4f}]")
                            if spill.centroid_lat is not None:
                                st.write(f"**Calculated spill lat/lon:** ({spill.centroid_lat:.6f}, {spill.centroid_lon:.6f})")
                                st.success("✅ Geographic coordinates computed from manual bounds!")
                            else:
                                st.error("❌ Failed to compute geographic coordinates from bounds")
                        else:
                            st.write("**Manual bounds:** Not provided")
                            if spill.centroid_lat is not None:
                                st.write(f"**Calculated spill lat/lon:** ({spill.centroid_lat:.6f}, {spill.centroid_lon:.6f})")
                                st.info("ℹ️ Coordinates from GeoTIFF georeferencing")
                            else:
                                st.error("❌ No geographic coordinates - need either georeferenced GeoTIFF or manual bounds")
                    
                    # Summary table
                    if geo_result.spills:
                        spill_data = []
                        for i, spill in enumerate(geo_result.spills):
                            spill_data.append({
                                "Spill #": i+1,
                                "Centroid (px)": f"({spill.centroid_px[0]:.1f}, {spill.centroid_px[1]:.1f})",
                                "Area (px)": spill.area_px,
                                "Area (km²)": f"{spill.area_sqkm:.4f}" if spill.area_sqkm else "N/A",
                                "Centroid Lat/Lon": f"({spill.centroid_lat:.6f}, {spill.centroid_lon:.6f})" if spill.centroid_lat else "N/A",
                                "Eccentricity": f"{spill.eccentricity:.3f}",
                                "Solidity": f"{spill.solidity:.3f}",
                                "Orientation (°)": f"{np.degrees(spill.orientation):.1f}"
                            })
                        st.dataframe(pd.DataFrame(spill_data), use_container_width=True)
    
    # Tab 4: Drift Backtracking
    with tab4:
        st.markdown('<div class="section-header">🌊 Ocean Drift Backtracking</div>', unsafe_allow_html=True)
        
        if not st.session_state.geospatial_result:
            st.warning("Please run segmentation/geospatial analysis first in the Localization tab.")
        else:
            geo_result = st.session_state.geospatial_result
            
            # Check georeferencing status
            has_geo_coords = geo_result.spills and geo_result.spills[0].centroid_lat is not None
            
            if not has_geo_coords:
                st.error("❌ Cannot run drift backtracking: No geographic coordinates available.")
                st.info("""
                **The spill centroid has no latitude/longitude.** This happens when:
                - The input image is not a georeferenced GeoTIFF
                - No bounds were provided in the sidebar
                - The GeoTIFF has identity transform (no georeferencing)
                
                **To enable drift tracking:**
                1. Use a properly georeferenced Sentinel-1 GeoTIFF from the original .SAFE product
                2. Or provide geographic bounds in the sidebar (Min/Max Lon/Lat)
                3. Or georeference your TIFF using GDAL/SNAP
                """)
            else:
                # Debug info for drift backtracking
                if geo_result.spills:
                    spill = geo_result.spills[0]
                    st.markdown("### 🔍 Debug: Drift Backtracking Input")
                    st.write(f"**Spill centroid (pixel):** ({spill.centroid_px[0]:.1f}, {spill.centroid_px[1]:.1f})")
                    st.write(f"**Spill centroid (lat/lon):** ({spill.centroid_lat:.6f}, {spill.centroid_lon:.6f})")
                    st.write(f"**Spill time:** {spill_time.isoformat()}")
                    st.write(f"**Drift duration:** {drift_duration} hours")
                    st.write(f"**Current (U, V):** ({current_u}, {current_v}) m/s")
                    st.write(f"**Coordinates passed to drift model:** lat={spill.centroid_lat:.6f}, lon={spill.centroid_lon:.6f}")
                
                if st.button("🌊 Run Drift Backtracking", type="primary", use_container_width=True):
                    with st.spinner("Running drift backtracking..."):
                        try:
                            # Use the first spill's centroid as observation point
                            if geo_result.spills:
                                spill = geo_result.spills[0]
                                obs_lat = spill.centroid_lat
                                obs_lon = spill.centroid_lon
                                
                                # Run drift backtracking
                                drift_model = create_prototype_model(
                                    "constant",
                                    u_velocity=current_u,
                                    v_velocity=current_v
                                )
                                
                                from src.drift.base import OilSpillDriftQuery
                                drift_query = OilSpillDriftQuery(
                                    spill_latitude=obs_lat,
                                    spill_longitude=obs_lon,
                                    spill_time=spill_time,
                                    backtrack_hours=drift_duration,
                                    time_step_hours=drift_step,
                                    current_source=drift_model.current_source,
                                )
                                
                                drift_result = drift_model.simulate(drift_query)
                                st.session_state.drift_result = drift_result
                                
                                st.success("Drift backtracking completed!")
                            else:
                                st.warning("No spills detected for drift backtracking.")
                        except Exception as e:
                            st.error(f"Drift backtracking failed: {e}")
            
            # Display drift results
            if st.session_state.drift_result:
                drift_result = st.session_state.drift_result
                
                # Prototype warning
                st.markdown("""
                <div class="warning-box">
                    <span class="warning-text">⚠️ PROTOTYPE OCEAN DRIFT MODEL:</span> 
                    This uses constant current assumptions and is <strong>not a validated ocean prediction model</strong>.
                    Real ocean drift requires validated current data from HYCOM, Copernicus, NCEP, or OSCAR.
                </div>
                """, unsafe_allow_html=True)
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Distance", f"{drift_result.total_distance_km:.2f} km")
                with col2:
                    st.metric("Net Displacement", f"{drift_result.net_displacement_km:.2f} km")
                with col3:
                    st.metric("Avg Current Speed", f"{drift_result.avg_current_speed*100:.1f} cm/s")
                with col4:
                    st.metric("Steps", len(drift_result.trajectory))
                
                # Trajectory plot
                traj_df = pd.DataFrame([{
                    'step': s.step,
                    'lat': s.latitude,
                    'lon': s.longitude,
                    'time': s.timestamp,
                    'current_speed': s.current_speed * 100,  # cm/s
                    'current_dir': s.current_direction
                } for s in drift_result.trajectory])
                
                # Map trajectory
                fig = go.Figure()
                fig.add_trace(go.Scattermap(
                    lat=traj_df['lat'],
                    lon=traj_df['lon'],
                    mode='markers+lines',
                    marker=dict(size=8, color=traj_df['step'], colorscale='Viridis', showscale=True),
                    line=dict(width=2, color='blue'),
                    text=[f"Step {s.step}<br>Time: {s.timestamp}<br>Speed: {s.current_speed*100:.1f} cm/s" for s in drift_result.trajectory],
                    hoverinfo='text',
                    name='Drift Trajectory'
                ))
                
                # Add start and end markers
                fig.add_trace(go.Scattermap(
                    lat=[drift_result.start_latitude],
                    lon=[drift_result.start_longitude],
                    mode='markers',
                    marker=dict(size=15, color='green', symbol='circle'),
                    name='Spill Observation'
                ))
                fig.add_trace(go.Scattermap(
                    lat=[drift_result.end_latitude],
                    lon=[drift_result.end_longitude],
                    mode='markers',
                    marker=dict(size=15, color='red', symbol='x'),
                    name='Backtracked Origin'
                ))
                
                fig.update_layout(
                    map_style="open-street-map",
                    map=dict(center=dict(lat=traj_df['lat'].mean(), lon=traj_df['lon'].mean()), zoom=8),
                    height=500,
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Trajectory table
                with st.expander("Trajectory Details"):
                    traj_display = traj_df[['step', 'lat', 'lon', 'time', 'current_speed', 'current_dir']].copy()
                    traj_display.columns = ['Step', 'Latitude', 'Longitude', 'Time', 'Speed (cm/s)', 'Direction (°)']
                    st.dataframe(traj_display, use_container_width=True)
    
    # Tab 5: AIS Search
    with tab5:
        st.markdown('<div class="section-header">🚢 AIS Vessel Search</div>', unsafe_allow_html=True)
        
        if not st.session_state.drift_result:
            st.warning("Please run drift backtracking first in the Drift Backtracking tab.")
        else:
            drift_result = st.session_state.drift_result
            
            col1, col2 = st.columns([1, 1])
            with col1:
                ais_search_radius = st.number_input("AIS Search Radius (km)", value=search_radius, min_value=1, max_value=500)
            with col2:
                ais_time_window = st.number_input("AIS Time Window (hours)", value=time_window, step=1.0)
            
            vessel_type_filter = st.multiselect("Filter by Vessel Type", ["Tanker", "Cargo", "Fishing", "Passenger", "Tug", "Offshore"])
            min_sog = st.number_input("Minimum SOG (knots)", value=0.0, step=0.5)
            
            if st.button("🚢 Search AIS Vessels", type="primary", use_container_width=True):
                with st.spinner("Loading AIS data and searching..."):
                    try:
                        # Load AIS data with debug info
                        ais_data, ais_stats, debug_info = load_ais_data_with_debug(ais_path)
                        st.session_state.ais_data = ais_data
                        st.session_state.ais_stats = ais_stats
                        st.session_state.ais_debug = debug_info
                        
                        # Show debug info
                        with st.expander("🔍 AIS Data Debug Info", expanded=True):
                            st.json(debug_info)
                        
                        # Use backtracked origin for search
                        drift_result = st.session_state.drift_result
                        backtracked_lat = drift_result.end_latitude
                        backtracked_lon = drift_result.end_longitude
                        backtracked_time = drift_result.end_time
                        
                        st.info(f"Searching at backtracked origin: ({backtracked_lat:.6f}, {backtracked_lon:.6f}) at {backtracked_time.isoformat()}")
                        st.info(f"Search radius: {ais_search_radius} km, Time window: ±{ais_time_window} hours")
                        if vessel_type_filter:
                            st.info(f"Vessel type filter: {vessel_type_filter}")
                        if min_sog > 0:
                            st.info(f"Minimum SOG: {min_sog} knots")
                        
                        # Run AIS proximity search
                        proximity_result = find_vessels_near_spill(
                            ais_data=st.session_state.ais_data,
                            spill_lat=backtracked_lat,
                            spill_lon=backtracked_lon,
                            spill_time=backtracked_time,
                            radius_km=ais_search_radius,
                            time_window_hours=ais_time_window,
                            vessel_types=vessel_type_filter if vessel_type_filter else None,
                            min_sog=min_sog if min_sog > 0 else None
                        )
                        
                        st.session_state.proximity_result = proximity_result
                        st.success(f"Found {len(proximity_result.vessels_found)} vessels!")
                        
                        # Show search debug info
                        with st.expander("🔍 Search Debug Info", expanded=True):
                            st.write(f"Search location: ({backtracked_lat:.6f}, {backtracked_lon:.6f})")
                            st.write(f"Search time: {backtracked_time.isoformat()}")
                            st.write(f"Radius: {ais_search_radius} km")
                            st.write(f"Time window: ±{ais_time_window} hours")
                            st.write(f"Total vessels in dataset: {proximity_result.total_vessels_in_dataset}")
                            st.write(f"Vessels after spatial filter: {len(proximity_result.vessels_found)}")
                            if proximity_result.vessels_found:
                                st.write("Vessel details:")
                                for i, v in enumerate(proximity_result.vessels_found):
                                    st.write(f"  {i+1}. {v.mmsi} - {v.vessel_name or 'N/A'} ({v.vessel_type or 'N/A'}) - {v.distance_km:.2f} km - Δt={v.time_diff_hours:+.2f}h")
                    except Exception as e:
                        st.error(f"AIS search failed: {e}")
                        import traceback
                        st.code(traceback.format_exc())
            
            # Display AIS results
            if 'proximity_result' in st.session_state:
                prox_result = st.session_state.proximity_result
                
                # Summary
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Vessels Found", len(prox_result.vessels_found))
                with col2:
                    st.metric("Search Radius", f"{prox_result.search_radius_km} km")
                with col3:
                    st.metric("Time Window", f"±{prox_result.time_window_hours}h" if prox_result.time_window_hours else "None")
                with col4:
                    st.metric("Total Vessels in Dataset", prox_result.total_vessels_in_dataset)
                
                if prox_result.vessels_found:
                    # Vessel table
                    vessel_data = []
                    for i, v in enumerate(prox_result.vessels_found):
                        vessel_data.append({
                            "Rank": i+1,
                            "MMSI": v.mmsi,
                            "Name": v.vessel_name or "N/A",
                            "Type": v.vessel_type or "N/A",
                            "Distance (km)": f"{v.distance_km:.2f}",
                            "Bearing (°)": f"{v.bearing_deg:.1f}",
                            "Time Diff (h)": f"{v.time_diff_hours:+.2f}",
                            "SOG (kn)": f"{v.sog:.1f}" if v.sog else "N/A",
                            "COG (°)": f"{v.cog:.0f}" if v.cog else "N/A",
                            "Lat": f"{v.latitude:.6f}",
                            "Lon": f"{v.longitude:.6f}"
                        })
                    
                    df_vessels = pd.DataFrame(vessel_data)
                    st.dataframe(df_vessels, use_container_width=True)
                    
                    # Map
                    st.subheader("Vessel Locations")
                    map_fig = go.Figure()
                    
                    # Add spill/backtracked origin
                    map_fig.add_trace(go.Scattermap(
                        lat=[drift_result.end_latitude] if st.session_state.drift_result else [12.5],
                        lon=[drift_result.end_longitude] if st.session_state.drift_result else [72.4],
                        mode='markers',
                        marker=dict(size=15, color='red', symbol='x'),
                        name='Backtracked Origin'
                    ))
                    
                    # Add vessels
                    vessel_lats = [v.latitude for v in prox_result.vessels_found]
                    vessel_lons = [v.longitude for v in prox_result.vessels_found]
                    vessel_names = [f"{v.vessel_name or 'N/A'} ({v.mmsi})" for v in prox_result.vessels_found]
                    vessel_types = [v.vessel_type or 'Unknown' for v in prox_result.vessels_found]
                    
                    map_fig.add_trace(go.Scattermap(
                        lat=vessel_lats,
                        lon=vessel_lons,
                        mode='markers',
                        marker=dict(size=10, color='blue'),
                        text=[f"{n}<br>MMSI: {m}<br>Type: {t}" for n, m, t in zip(vessel_names, [v.mmsi for v in prox_result.vessels_found], vessel_types)],
                        hoverinfo='text',
                        name='AIS Vessels'
                    ))
                    
                    map_fig.update_layout(
                        map_style="open-street-map",
                        map=dict(center=dict(
                            lat=np.mean(vessel_lats) if vessel_lats else 12.5,
                            lon=np.mean(vessel_lons) if vessel_lons else 72.4
                        ), zoom=8),
                        height=500,
                        margin=dict(l=0, r=0, t=30, b=0)
                    )
                    st.plotly_chart(map_fig, use_container_width=True)
                    
                    # Proximity report
                    with st.expander("Full Proximity Report"):
                        st.code(format_proximity_report(prox_result))
                else:
                    st.info("No vessels found within search criteria. Try expanding the search radius or time window.")
    
    # Tab 6: Attribution
    with tab6:
        st.markdown('<div class="section-header">🎯 Vessel Attribution Scoring</div>', unsafe_allow_html=True)
        
        if 'proximity_result' not in st.session_state:
            st.warning("Please run AIS search first in the AIS Search tab.")
        else:
            st.markdown("### Attribution Configuration")
            
            # Show current weights
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Distance", f"{weights['distance']:.0%}")
            with col2:
                st.metric("Temporal", f"{weights['temporal']:.0%}")
            with col3:
                st.metric("Drift/Trajectory", f"{weights['drift']:.0%}")
            with col4:
                st.metric("Vessel Type", f"{weights['vessel_type']:.0%}")
            with col5:
                st.metric("Data Quality", f"{weights['data_quality']:.0%}")
            
            # Show AIS data debug info
            if 'ais_debug' in st.session_state:
                with st.expander("🔍 AIS Data Debug Info (from AIS Search tab)", expanded=False):
                    st.json(st.session_state.ais_debug)
            
            # Use same filters as AIS Search tab
            st.markdown("### Search Filters (from AIS Search tab)")
            col1, col2 = st.columns(2)
            with col1:
                vessel_type_filter = st.multiselect("Filter by Vessel Type", ["Tanker", "Cargo", "Fishing", "Passenger", "Tug", "Offshore"], key="attr_vessel_type_filter")
            with col2:
                min_sog = st.number_input("Minimum SOG (knots)", value=0.0, step=0.5, key="attr_min_sog")
            
            # Show search parameters being used
            st.info(f"""
            **Search Parameters:**
            - Spill location: ({spill_lat:.6f}, {spill_lon:.6f})
            - Spill time: {spill_time.isoformat()}
            - Drift duration: {drift_duration} hours
            - Search radius: {search_radius} km
            - Time window: ±{time_window} hours
            - Vessel type filter: {vessel_type_filter if vessel_type_filter else 'None'}
            - Minimum SOG: {min_sog if min_sog > 0 else 'None'}
            """)
            
            if st.button("🎯 Run Attribution Scoring", type="primary", use_container_width=True):
                with st.spinner("Running attribution analysis..."):
                    try:
                        # Create weights object
                        attr_weights = AttributionWeights(
                            distance=weights['distance'],
                            temporal=weights['temporal'],
                            drift=weights['drift'],
                            vessel_type=weights['vessel_type'],
                            data_quality=weights['data_quality']
                        )
                        
                        # Run attribution with same filters as AIS search
                        attribution_result = run_attribution_analysis(
                            ais_data=st.session_state.ais_data,
                            spill_latitude=spill_lat,
                            spill_longitude=spill_lon,
                            spill_time=spill_time,
                            drift_duration_hours=drift_duration,
                            drift_time_step_hours=drift_step,
                            search_radius_km=search_radius,
                            time_window_hours=time_window,
                            current_u=current_u,
                            current_v=current_v,
                            scoring_weights=attr_weights,
                            vessel_types=vessel_type_filter if vessel_type_filter else None,
                            min_sog=min_sog if min_sog > 0 else None,
                        )
                        
                        st.session_state.attribution_result = attribution_result
                        st.success("Attribution analysis completed!")
                    except Exception as e:
                        st.error(f"Attribution failed: {e}")
                        import traceback
                        st.code(traceback.format_exc())
            
            # Display attribution results
            if st.session_state.attribution_result:
                attr_result = st.session_state.attribution_result
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Candidates", attr_result.total_candidates)
                with col2:
                    st.metric("Top Score", f"{attr_result.candidates[0].final_score:.3f}" if attr_result.candidates else "N/A")
                with col3:
                    st.metric("Top Vessel", attr_result.candidates[0].vessel_name if attr_result.candidates else "N/A")
                with col4:
                    st.metric("Types Found", len(attr_result.candidates_by_type))
                
                # Attribution table
                st.subheader("Ranked Candidates")
                table_data = attr_result.to_table()
                if table_data:
                    df_attr = pd.DataFrame(table_data)
                    # Format for display
                    display_cols = ['rank', 'mmsi', 'vessel_name', 'vessel_type', 
                                   'distance_km', 'time_diff_hours',
                                   'distance_score', 'temporal_score', 'drift_score', 
                                   'vessel_type_score', 'data_quality_score', 'final_score']
                    df_display = df_attr[display_cols].copy()
                    df_display.columns = ['Rank', 'MMSI', 'Name', 'Type', 'Dist(km)', 'ΔT(h)',
                                         'Dist Score', 'Temp Score', 'Drift Score', 'Type Score', 'DQ Score', 'Final']
                    st.dataframe(df_display, use_container_width=True)
                
                # Top candidate details
                if attr_result.candidates:
                    top = attr_result.get_top_n(3)
                    st.subheader("Top 3 Candidates - Evidence")
                    for c in top:
                        with st.expander(f"Rank {c.rank}: {c.vessel_name} ({c.mmsi}) - Score: {c.final_score:.3f}"):
                            st.write(f"**Type:** {c.vessel_type}")
                            st.write(f"**Distance to Origin:** {c.distance_to_origin_km:.2f} km")
                            st.write(f"**Time Difference:** {c.time_diff_hours:+.2f} hours")
                            st.write(f"**Component Scores:**")
                            st.write(f"  - Distance: {c.component_scores.distance_score:.3f}")
                            st.write(f"  - Temporal: {c.component_scores.temporal_score:.3f}")
                            st.write(f"  - Drift/Trajectory: {c.component_scores.drift_score:.3f}")
                            st.write(f"  - Vessel Type: {c.component_scores.vessel_type_score:.3f}")
                            st.write(f"  - Data Quality: {c.component_scores.data_quality_score:.3f}")
                            st.write(f"**Evidence:** {c.evidence_summary}")
                            st.write(f"**Final Score:** {c.final_score:.3f}")
                
                # Full report
                with st.expander("Full Attribution Report"):
                    st.code(format_attribution_report(attr_result))
                
                # Disclaimer
                st.markdown("""
                <div class="warning-box">
                    <span class="warning-text">⚠️ IMPORTANT:</span> The vessel attribution output is a <strong>model-based candidate ranking</strong> 
                    intended for research and demonstration. It is <strong>not proof of legal responsibility or causation</strong>. 
                    Prototype ocean-drift and vessel-type assumptions have not been scientifically validated.
                </div>
                """, unsafe_allow_html=True)
    
    # Tab 7: Map
    with tab7:
        st.markdown('<div class="section-header">🗺️ Interactive Map</div>', unsafe_allow_html=True)
        
        # Create comprehensive map
        map_center_lat = spill_lat
        map_center_lon = spill_lon
        
        if st.session_state.drift_result:
            map_center_lat = st.session_state.drift_result.end_latitude
            map_center_lon = st.session_state.drift_result.end_longitude
        
        # Create folium map
        m = folium.Map(location=[map_center_lat, map_center_lon], zoom_start=9, tiles="OpenStreetMap")
        
        # Add tile layers
        folium.TileLayer('CartoDB positron', name='Light Map').add_to(m)
        folium.TileLayer('CartoDB dark_matter', name='Dark Map').add_to(m)
        folium.TileLayer('OpenStreetMap', name='OpenStreetMap').add_to(m)
        
        # Spill observation point
        folium.Marker(
            location=[spill_lat, spill_lon],
            popup=f"Spill Observation\nLat: {spill_lat:.6f}\nLon: {spill_lon:.6f}\nTime: {spill_time.isoformat()}",
            icon=folium.Icon(color='red', icon='exclamation-triangle', prefix='fa')
        ).add_to(m)
        
        # Drift trajectory
        if st.session_state.drift_result:
            drift_result = st.session_state.drift_result
            
            # Trajectory line
            traj_coords = [[s.latitude, s.longitude] for s in drift_result.trajectory]
            folium.PolyLine(
                traj_coords,
                color='blue',
                weight=3,
                opacity=0.8,
                tooltip='Drift Backtrack Trajectory'
            ).add_to(m)
            
            # Spill observation
            folium.Marker(
                location=[drift_result.start_latitude, drift_result.start_longitude],
                popup=f"Spill Observation\nLat: {drift_result.start_latitude:.6f}\nLon: {drift_result.start_longitude:.6f}",
                icon=folium.Icon(color='red', icon='exclamation-triangle', prefix='fa')
            ).add_to(m)
            
            # Backtracked origin
            folium.Marker(
                location=[drift_result.end_latitude, drift_result.end_longitude],
                popup=f"Backtracked Origin\nLat: {drift_result.end_latitude:.6f}\nLon: {drift_result.end_longitude:.6f}\nTime: {drift_result.end_time.isoformat()}",
                icon=folium.Icon(color='blue', icon='flag', prefix='fa')
            ).add_to(m)
        
        # AIS vessels
        if 'proximity_result' in st.session_state and st.session_state.proximity_result.vessels_found:
            for i, v in enumerate(st.session_state.proximity_result.vessels_found):
                color = 'green' if i == 0 else 'blue'
                folium.CircleMarker(
                    location=[v.latitude, v.longitude],
                    radius=8,
                    popup=f"{v.vessel_name or 'N/A'} ({v.mmsi})\nType: {v.vessel_type or 'N/A'}\nDistance: {v.distance_km:.2f} km\nTime Diff: {v.time_diff_hours:+.2f}h",
                    color=color,
                    fill=True,
                    fill_opacity=0.7
                ).add_to(m)
        
        # Segmentation spill centroids
        if st.session_state.geospatial_result and st.session_state.geospatial_result.spills:
            for i, spill in enumerate(st.session_state.geospatial_result.spills):
                if spill.centroid_lat and spill.centroid_lon:
                    folium.CircleMarker(
                        location=[spill.centroid_lat, spill.centroid_lon],
                        radius=10,
                        popup=f"Spill #{i+1}\nArea: {spill.area_sqkm:.4f} km²\nCentroid: ({spill.centroid_lat:.6f}, {spill.centroid_lon:.6f})",
                        color='orange',
                        fill=True,
                        fill_opacity=0.5
                    ).add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Add scale
        folium.plugins.MeasureControl(position='topright', primary_length_unit='kilometers').add_to(m)
        
        # Render map
        st_folium(m, width=None, height=600, returned_objects=[])
        
        # Legend
        st.markdown("""
        **Map Legend:**
        - 🔴 **Red Triangle**: Spill Observation Point
        - 🔵 **Blue Flag**: Backtracked Origin (Drift Model)
        - 🔵 **Blue Circles**: AIS Vessels (Green = Top Candidate)
        - 🟠 **Orange Circles**: Segmentation Spill Centroids
        - 🔵 **Blue Line**: Drift Backtrack Trajectory
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; padding: 1rem;">
    SAMUDRA SETU — Oil Spill Detection & Vessel Attribution System<br>
    <small>Research & Demonstration Platform | Not for Operational Use</small>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()