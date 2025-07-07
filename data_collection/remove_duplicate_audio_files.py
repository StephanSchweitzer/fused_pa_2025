import os
from pathlib import Path

DIRECTORY_PATH = r"C:\path\to\your\directory"

DRY_RUN = True


def find_duplicate_files(directory_path, dry_run=True):
    directory_path = Path(directory_path)
    
    if not directory_path.exists():
        raise FileNotFoundError(f"Directory path does not exist: {directory_path}")
    
    if not directory_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {directory_path}")
    
    duplicate_files = []
    deleted_files = []
    errors = []
    
    print(f"Scanning directory: {directory_path}")
    print(f"Mode: {'DRY RUN' if dry_run else 'EXECUTE'}")
    print("-" * 50)
    
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if "_dup" in file:
                file_path = Path(root) / file
                duplicate_files.append(file_path)
                
                if not dry_run:
                    try:
                        file_path.unlink()
                        deleted_files.append(file_path)
                        print(f"Deleted: {file_path}")
                    except Exception as e:
                        error_msg = f"Error deleting {file_path}: {e}"
                        errors.append(error_msg)
                        print(error_msg)
                else:
                    print(f"Would delete: {file_path}")
    
    return {
        'found': duplicate_files,
        'deleted': deleted_files,
        'errors': errors
    }


def main():
    try:
        result = find_duplicate_files("tts_data\processed\cremad", dry_run=False)
        
        print(f"\nSummary:")
        print(f"Files found with '_dup': {len(result['found'])}")
        
        if not DRY_RUN:
            print(f"Files successfully deleted: {len(result['deleted'])}")
            print(f"Errors encountered: {len(result['errors'])}")
            
            if result['errors']:
                print("\nErrors:")
                for error in result['errors']:
                    print(f"  {error}")
        else:
            print("\nTo actually delete these files, set DRY_RUN = False")
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    main()