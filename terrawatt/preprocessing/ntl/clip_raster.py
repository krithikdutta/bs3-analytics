import rasterio
from rasterio.mask import mask
import geopandas as gpd

def clip_raster(raster_path, shapefile_path, output_path):
    """
    Clips a raster file using a shapefile boundary.
    
    Parameters:
    raster_path (str): Path to the input raster file
    shapefile_path (str): Path to the input shapefile
    output_path (str): Path where the clipped raster will be saved
    """
    try:
        # Read the shapefile and get its geometry
        print("Loading shapefile...")
        aoi_gdf = gpd.read_file(shapefile_path)
        geometries = aoi_gdf.geometry.values

        # Open the raster
        print("Opening raster file...")
        with rasterio.open(raster_path) as src:
            # Check if CRS matches and reproject if needed
            if aoi_gdf.crs != src.crs:
                print(f"Shapefile CRS ({aoi_gdf.crs}) does not match raster CRS ({src.crs}). Reprojecting...")
                aoi_gdf = aoi_gdf.to_crs(src.crs)
                geometries = aoi_gdf.geometry.values

            # Perform the clipping operation
            print("Clipping raster...")
            out_image, out_transform = mask(
                src,
                geometries,
                crop=True,
                all_touched=False
            )
            
            # Copy and update metadata
            out_meta = src.meta.copy()
            out_meta.update({
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "driver": "GTiff"
            })
            
            # Write the clipped raster
            print(f"Writing output to {output_path}...")
            with rasterio.open(output_path, "w", **out_meta) as dest:
                dest.write(out_image)

        print("Clipping complete!")
        return True
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return False

# Example usage:
if __name__ == "__main__":
    raster_path = "VNL_npp_2024_global_vcmslcfg_v2_c202502261200.average.dat.tif"
    shapefile_path = "Raipur Shapefile/raipur_shapefile.shp"
    output_path = "clipped_nightlights_rpr.tif"
    
    clip_raster(raster_path, shapefile_path, output_path)