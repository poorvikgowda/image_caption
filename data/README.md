# Dataset Setup Guide

This project supports **Flickr8k** and **Flickr30k** datasets.

## Expected Directory Layout

### Flickr8k
Place the dataset under `data/flickr8k/`:
```
data/
└── flickr8k/
    ├── Images/
    │   ├── 1000268201_693b08cb0e.jpg
    │   ├── 1001773457_577c3a7d70.jpg
    │   └── ... (8,091 JPEG images)
    └── captions.txt
```

### Flickr30k
Place the dataset under `data/flickr30k/`:
```
data/
└── flickr30k/
    ├── Images/
    │   ├── 1000092795.jpg
    │   └── ... (31,783 JPEG images)
    └── captions.txt
```

## Format of `captions.txt`
The system supports both common formatting styles:
1. Standard Kaggle CSV style:
   ```csv
   image,caption
   1000268201_693b08cb0e.jpg,A child in a pink dress is climbing up a set of stairs in an entry way.
   ```
2. Standard Flickr text style:
   ```text
   1000268201_693b08cb0e.jpg#0 A child in a pink dress is climbing up a set of stairs in an entry way.
   ```

## Dataset Configuration
In `configs/config.yaml`, set:
```yaml
dataset:
  name: "flickr8k" # or flickr30k
  image_dir: "data/flickr8k/Images"
  captions_file: "data/flickr8k/captions.txt"
```

## Automatic Synthetic Dataset Generation
If the real Flickr dataset is not found when running data preparation or training scripts, the system will automatically create a synthetic dummy dataset (`data/flickr8k/Images` and `data/flickr8k/captions.txt`) containing sample colorful images and captions to allow immediate pipeline verification and testing without downloading 1GB+ files.
