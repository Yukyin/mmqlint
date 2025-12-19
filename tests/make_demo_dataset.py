from datasets import Dataset, Features, Value
import os

OUT_DIR = "demo_ds"

def main():
    # Original data (note: some samples are missing meta.b; one field is also intentionally missing in coordinates)
    raw = [
        {
            "id": "ok1",
            "image_w": 512,
            "image_h": 512,
            "meta": {"a": 1, "b": 2},
            "coordinates": {"x0": 10, "y0": 20, "x1": 100, "y1": 120},
        },
        {
            "id": "missing_meta_b",
            "image_w": 512,
            "image_h": 512,
            "meta": {"a": 9},  # Missing b -> after cast it will be filled with None
            "coordinates": {"x0": 5, "y0": 5, "x1": 60, "y1": 60},
        },
        {
            "id": "missing_coordinates_y1",
            "image_w": 512,
            "image_h": 512,
            "meta": {"a": 3, "b": 4},
            "coordinates": {"x0": 10, "y0": 10, "x1": 20},  # Missing y1 -> after cast it will be filled with None
        },
        {
            "id": "coords_look_like_1000_space",
            "image_w": 512,
            "image_h": 512,
            "meta": {"a": 7, "b": 8},
            # coordinates clearly exceed 512, but it looks like a 0..1000 coordinate system
            "coordinates": {"x0": 50, "y0": 80, "x1": 900, "y1": 950},
        },
    ]

    ds = Dataset.from_list(raw)

    # all nested keys are declared and are int32
    # Note: for the datasets struct or dict field, using a dict in Features is sufficient  

    feats = Features(
        {
            "id": Value("string"),
            "image_w": Value("int32"),
            "image_h": Value("int32"),
            "meta": {
                "a": Value("int32"),
                "b": Value("int32"),
            },
            "coordinates": {
                "x0": Value("int32"),
                "y0": Value("int32"),
                "x1": Value("int32"),
                "y1": Value("int32"),
            },
        }
    )

    # Key point: cast will fill missing nested keys with None
    ds = ds.cast(feats)

    if os.path.exists(OUT_DIR):
        # Clean the old directory
        import shutil
        shutil.rmtree(OUT_DIR)

    ds.save_to_disk(OUT_DIR)
    print(f"Saved demo dataset to: {OUT_DIR}")
    print(ds)
    print("features:", ds.features)

if __name__ == "__main__":
    main()
