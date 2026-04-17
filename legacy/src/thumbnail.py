import os
from zipfile import ZipFile

# Configuration
FILES_DIRECTORY = "files"
THUMBNAIL_DIRECTORY = "thumbnail"

def get_szi_zip(image_id):
    file_path = os.path.join(FILES_DIRECTORY, f"{image_id}.szi")
    return ZipFile(file_path, 'r')

def extract_thumbnail(image_id):
    try:
        with get_szi_zip(image_id) as z:
            tile_path = f"{image_id}_files/10/0_0.jpg"
            if tile_path not in z.namelist():
                raise FileNotFoundError(f"Tile {tile_path} not found in archive.")
            
            # Create thumbnail directory if it doesn't exist
            os.makedirs(THUMBNAIL_DIRECTORY, exist_ok=True)
            
            # Extract and save the thumbnail
            thumbnail_path = os.path.join(THUMBNAIL_DIRECTORY, f"{image_id}.jpg")
            with open(thumbnail_path, 'wb') as f:
                f.write(z.read(tile_path))
            print(f"Extracted thumbnail for {image_id}")
    except Exception as e:
        print(f"Failed to extract thumbnail for {image_id}: {str(e)}")

def main():
    # Get all .szi files
    szi_files = [f[:-4] for f in os.listdir(FILES_DIRECTORY) if f.endswith('.szi')]
    
    print(f"Found {len(szi_files)} .szi files")
    for image_id in szi_files:
        extract_thumbnail(image_id)

if __name__ == "__main__":
    main()
