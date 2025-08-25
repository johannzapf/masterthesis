import os
from cryptography.fernet import Fernet


def generate_key(key_path="encryption_key.key"):
    """
    Generate a new encryption key and save it to a file.
    """
    key = Fernet.generate_key()
    with open(key_path, "wb") as key_file:
        key_file.write(key)
    print(f"Key saved to {key_path}")
    return key


def load_key(key_path="encryption_key.key"):
    """
    Load an existing encryption key from a file.
    """
    with open(key_path, "rb") as key_file:
        key = key_file.read()
    return key


def encrypt_files(directory, key):
    """
    Recursively encrypt all CSV/ZIP files containing 'sovanta' in the filename.
    Encrypted files will have '.enc' appended.
    """
    cipher = Fernet(key)
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            if "sovanta" in file.lower() and (file.endswith(".csv") or file.endswith(".zip")):
                file_path = os.path.join(subdir, file)
                with open(file_path, "rb") as f:
                    data = f.read()
                encrypted_data = cipher.encrypt(data)
                with open(file_path + ".enc", "wb") as f:
                    f.write(encrypted_data)
                print(f"Encrypted: {file_path}")


def decrypt_files(directory, key):
    """
    Recursively decrypt all files ending with '.enc' containing 'sovanta' in the filename.
    Decrypted files will overwrite the '.enc' files with original file names.
    """
    cipher = Fernet(key)
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            if "sovanta" in file.lower() and (file.endswith(".csv") or file.endswith(".zip")):
                file_path = os.path.join(subdir, file)
                with open(file_path, "rb") as f:
                    encrypted_data = f.read()
                decrypted_data = cipher.decrypt(encrypted_data)
                original_path = file_path[:-4]  # Remove '.enc'
                with open(original_path, "wb") as f:
                    f.write(decrypted_data)
                print(f"Decrypted: {original_path}")


if __name__ == "__main__":
    key = load_key()
    decrypt_files("../", key)
