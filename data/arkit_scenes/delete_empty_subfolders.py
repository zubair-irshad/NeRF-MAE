import os

def delete_empty_subfolders(directory):
    for root, dirs, files in os.walk(directory, topdown=False):
        for folder in dirs:
            folder_path = os.path.join(root, folder)
            if not os.listdir(folder_path):
                print("Deleting empty subfolder:", folder_path)
                os.rmdir(folder_path)

# Provide the path to the directory containing the subfolders
directory_path = "/arkit_data/ngp_data"

# Call the function to delete empty subfolders
delete_empty_subfolders(directory_path)

def count_non_empty_folders_and_files(directory):
    non_empty_folders = 0
    file_count = 0

    for root, dirs, files in os.walk(directory):
        non_empty_folders += len(dirs)
        # for folder in dirs:
        #     # folder_path = os.path.join(root, folder)
        #     # if os.listdir(folder_path):
        #     #     non_empty_folders += 1
        
        if root.endswith("images"):
            file_count += len(files)

    return non_empty_folders, file_count

# Call the function to count non-empty folders and files under "images"
non_empty_folders_count, file_count = count_non_empty_folders_and_files(directory_path)

print("Non-empty folders count:", non_empty_folders_count)
print("File count:", file_count)