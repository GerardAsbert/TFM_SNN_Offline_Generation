import shutil
from pathlib import Path


def extract_symbol_name(filename):
    """
    Extract handwritten symbol name from filename.

    Example:
        sample_12-8-Time.txt  -> Time
        abc_xyz_sharp.txt     -> sharp

    Takes everything after the LAST "_" and before ".txt"
    """

    stem = Path(filename).stem

    if "_" not in stem:
        return None

    return stem.split("_")[-1]


def organize_author_directory(author_dir):
    """
    Create symbol folders inside one author directory
    and move matching txt files into them.
    """

    author_dir = Path(author_dir)

    for item in author_dir.iterdir():

        # Ignore directories
        if item.is_dir():
            continue

        # Only process txt files
        if item.suffix.lower() != ".txt":
            continue

        symbol_name = extract_symbol_name(item.name)

        if symbol_name is None:
            print(f"Skipping (no symbol found): {item.name}")
            continue

        # Create symbol directory
        symbol_dir = author_dir / symbol_name
        symbol_dir.mkdir(exist_ok=True)

        # Move file
        destination = symbol_dir / item.name

        shutil.move(str(item), str(destination))

        print(f"Moved: {item.name} -> {symbol_name}/")


def process_root_directory(root_dir):
    """
    Root directory contains author-id subdirectories.
    """

    root_dir = Path(root_dir)

    for author_dir in root_dir.iterdir():

        if not author_dir.is_dir():
            continue

        print(f"\nProcessing author: {author_dir.name}")

        organize_author_directory(author_dir)

    print("\nDone.")


if __name__ == "__main__":

    ROOT_DIRECTORY = "/data/113-2/users/gasbert/HOMUS_PROCESSED"

    process_root_directory(ROOT_DIRECTORY)