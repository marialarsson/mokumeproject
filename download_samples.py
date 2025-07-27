import os

import gdown
import zipfile

if __name__ == "__main__":
    if not os.path.exists("Samples"):
        os.makedirs("Samples/H11")
        os.makedirs("Samples/CN03")
        
    if not os.path.exists("Samples/H11/nca_cube.npz"):
        url = (
            "https://drive.google.com/uc?id=1itzh5y7TdOk96RRDkXAxJj8_pl3AV-ks"
        )
        gdown.download(url, "data.zip", quiet=False)
        print("Unzipping the Generated 3D Cube data...")
        with zipfile.ZipFile("data.zip", "r") as zip_ref:
            zip_ref.extractall("./")
        os.remove("data.zip")
    else:
        print("The 3D Cube data already exists...")
        print("Remove the existing files at Samples/ to download again.\n")