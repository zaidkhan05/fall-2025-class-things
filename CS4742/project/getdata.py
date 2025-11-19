import kagglehub

# Download latest version
path = kagglehub.dataset_download("timilsinabimal/newsarticlecategories")

print("Path to dataset files:", path)