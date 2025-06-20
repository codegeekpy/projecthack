import kagglehub

# Download latest version
path = kagglehub.dataset_download("melissamonfared/sephora-skincare-reviews")

print("Path to dataset files:", path)