import argparse
import os
import sys
from utility import duplicate_images, duplicate_image_delete

def main():
    parser = argparse.ArgumentParser(description="Find and delete duplicate images")
    parser.add_argument("--folder_path", type=str, required=True, help="Folder path (required)")
    parser.add_argument("--txt_file_path", type=str, help="Text file path (optional)")
    
    args = parser.parse_args()

    try:
        if not os.path.isdir(args.folder_path):
            raise FileNotFoundError("Folder path does not exist or is not a directory.")

        if args.txt_file_path is None:
            print("-- Find Duplicate Files --")
            duplicate_images.find_duplicates(args.folder_path)

        else:
            if not os.path.isfile(args.txt_file_path):
                raise FileNotFoundError("Text file path does not exist or is not a file.")

            print("-- Delete Duplicate Images --")
            duplicate_image_delete.delete_duplicates(args.txt_file_path, args.folder_path)

    except FileNotFoundError as e:
        print(f"Path Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()