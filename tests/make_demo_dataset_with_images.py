from datasets import Dataset, Features, Value, Image
from PIL import Image as PILImage
import os

os.makedirs("demo_images", exist_ok=True)

# Generate some sample images
paths = []
for i in range(4):
    p = f"demo_images/{i}.png"
    PILImage.new("RGB", (512, 512)).save(p)
    paths.append(p)

data = {
  "id": ["ok0","missing_meta_b","missing_coordinates_y1","coords_look_like_1000_space"],
  "image": paths,           
  "image_w": [512,512,512,512],
  "image_h": [512,512,512,512],
  "meta": [{"a":1,"b":2},{"a":1},{"a":1,"b":2},{"a":1,"b":2}],
  "coordinates": [{"x0":10,"y0":10,"x1":100,"y1":100},
           {"x0":10,"y0":10,"x1":100,"y1":100},
           {"x0":10,"y0":10,"x1":100,"y1":None},
           {"x0":50,"y0":80,"x1":900,"y1":950}],
}

features = Features({
  "id": Value("string"),
  "image": Image(),         
  "image_w": Value("int32"),
  "image_h": Value("int32"),
  "meta": {"a": Value("int32"), "b": Value("int32")},
  "coordinates": {"x0": Value("int32"), "y0": Value("int32"), "x1": Value("int32"), "y1": Value("int32")},
})

ds = Dataset.from_dict(data).cast(features)
ds.save_to_disk("demo_ds_img")
print("Saved to demo_ds_img")
print(ds.features)

