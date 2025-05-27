# wsl_permission_manager.py
import os
import stat

def set_path_permissions(path_to_set, permission_mode_octal=0o666):
    """
    Sets the permissions for a given file or directory path.

    Args:
        path_to_set (str): The file or directory path (can be a WSL path).
        permission_mode_octal (int): The permission mode in octal notation.
                                     Default is 0o666 (rw-rw-rw-).
                                     For directories, you might need 0o777 (rwxrwxrwx).
    Returns:
        bool: True if permissions were set successfully, False otherwise.
    """
    print(f"Attempting to set permissions for: '{path_to_set}' to {oct(permission_mode_octal)}")

    if not os.path.exists(path_to_set):
        print(f"Error: Path '{path_to_set}' does not exist.")
        return False

    try:
        os.chmod(path_to_set, permission_mode_octal)
        print(f"Successfully set permissions for '{path_to_set}' to {oct(permission_mode_octal)}.")
        # Optionally, you can verify by reading the stats, but this is more complex for WSL from Windows
        # current_mode = stat.S_IMODE(os.stat(path_to_set).st_mode)
        # print(f"Verified current mode: {oct(current_mode)}")
        return True
    except PermissionError:
        print(f"Error: Permission denied. You might not have the necessary rights to change permissions for '{path_to_set}'.")
        print("       Try running your Jupyter Notebook or script with elevated privileges,")
        print("       or ensure your user has rights to modify permissions on this WSL path.")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False

def grant_universal_read_write_access(file_path):
    """
    Grants read and write access (0o666) to everyone for a specific file.
    WARNING: This is broadly permissive. Use with caution.
    """
    print("WARNING: Setting permissions to read/write for Owner, Group, and Others (0o666).")
    return set_path_permissions(file_path, 0o666)

def grant_universal_directory_access(dir_path):
    """
    Grants read, write, and execute access (0o777) to everyone for a specific directory.
    This allows listing, creating, and deleting files within the directory.
    WARNING: This is broadly permissive. Use with caution.
    """
    print("WARNING: Setting permissions to read/write/execute for Owner, Group, and Others (0o777).")
    return set_path_permissions(dir_path, 0o777)

if __name__ == "__main__":
    print("This script provides functions to manage file/directory permissions.")
    print("Import these functions into your Jupyter Notebook or another Python script.")
    print("\nExample usage (if you were to run this directly, which is not the primary intent):")

    # Create dummy files/dirs for testing if run directly (won't work for WSL paths without WSL)
    # This part is more for illustrating direct execution and might need adjustment for your environment.
    if os.name == 'posix': # Simple check if on a Linux-like system for direct test
        TEST_FILE = "./test_file.txt"
        TEST_DIR = "./test_dir"

        with open(TEST_FILE, "w") as f:
            f.write("test")
        if not os.path.exists(TEST_DIR):
            os.makedirs(TEST_DIR)

        print(f"\nTesting on a local file: {TEST_FILE}")
        grant_universal_read_write_access(TEST_FILE)

        print(f"\nTesting on a local directory: {TEST_DIR}")
        grant_universal_directory_access(TEST_DIR)

        # Clean up
        # os.remove(TEST_FILE)
        # os.rmdir(TEST_DIR)
    else:
        print("\nDirect execution example is simplified. For WSL paths, call from your notebook.")
        print("Example WSL path: '\\\\wsl.localhost\\Ubuntu\\home\\user\\yourfile.txt'")