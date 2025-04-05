import zipfile
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Extract a zip file.")
    parser.add_argument("zip_path", help="Path to the zip file")
    parser.add_argument("target_dir", help="Directory to extract files to")
    args = parser.parse_args()

    if not os.path.exists(args.zip_path):
        print(f"Error: Zip file '{args.zip_path}' not found.")
        return

    with zipfile.ZipFile(args.zip_path, "r") as zip_ref:
        zip_ref.extractall(args.target_dir)
        print(f"Extracted '{args.zip_path}' to '{args.target_dir}'")

if __name__ == "__main__":
    main()