from datasets import Dataset, Features, Value

def main():
    feats = Features({
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
    })

    rows = [
        {"id":"ok1","image_w":512,"image_h":512,"meta":{"a":1,"b":2},"coordinates":{"x0":10,"y0":20,"x1":100,"y1":200}},
        {"id":"ok2","image_w":512,"image_h":512,"meta":{"a":3,"b":4},"coordinates":{"x0":0,"y0":0,"x1":511,"y1":511}},
        {"id":"ok3","image_w":1000,"image_h":1000,"meta":{"a":5,"b":6},"coordinates":{"x0":50,"y0":80,"x1":900,"y1":950}},
        {"id":"ok4","image_w":640,"image_h":480,"meta":{"a":7,"b":8},"coordinates":{"x0":12,"y0":34,"x1":123,"y1":234}},
    ]

    ds = Dataset.from_list(rows, features=feats)
    out = "demo_ds_all_ok"
    ds.save_to_disk(out)
    print(f"Saved demo dataset to: {out}")
    print(ds)

if __name__ == "__main__":
    main()

