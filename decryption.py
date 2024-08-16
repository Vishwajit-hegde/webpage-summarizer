# type: ignore
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
from cryptography.fernet import Fernet
import base64
import os

def decrypt_api(password: str, encrypted: str) -> str:
    # Decode the base64 encoded string
    encrypted_data = base64.urlsafe_b64decode(encrypted)
    
    # Extract the salt and ciphertext
    salt = encrypted_data[:16]
    ciphertext = encrypted_data[16:]
    
    # Derive the key again using the same method
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend()
    )
    key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
    
    # Decrypt the ciphertext
    fernet = Fernet(key)
    plaintext = fernet.decrypt(ciphertext).decode()
    
    return plaintext

