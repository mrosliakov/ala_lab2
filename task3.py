import numpy as np

def encrypt_message(message, key_matrix):
    message_vector = np.array([ord(char) for char in message])
    eigenvalues, eigenvectors = np.linalg.eig(key_matrix)
    
    diagonalized_key = np.dot(np.dot(eigenvectors, np.diag(eigenvalues)), np.linalg.inv(eigenvectors))
    encrypted_vector = np.dot(diagonalized_key, message_vector)
    return encrypted_vector

def decrypt_message(encrypted_vector, key_matrix):
    eig_val, eig_vec = np.linalg.eig(key_matrix)
    
    inv_eig_val = 1 / eig_val
    diag_inv = np.diag(inv_eig_val)
    inv_eig_vec = np.linalg.inv(eig_vec)
    
    key_matrix_inv = np.dot(np.dot(eig_vec, diag_inv), inv_eig_vec)

    decr_vector = np.real(np.dot(key_matrix_inv, encrypted_vector))
    decr = "".join(chr(int(round(x))) for x in decr_vector)
    return decr

if __name__ == "__main__":
    
    message = input("Enter a message to encrypt (Empty line for default): ")
    if not message:
        message = "Hello, World!"
    n = len(message)
    key_matrix = np.random.randint(0, 256, (n, n))
    
    print("Key Matrix:\n", key_matrix)
    
    print(f"Original: {message}")
    
    encr = encrypt_message(message, key_matrix)
    print(f"Encrypted vector: {encr}")
    
    decr = decrypt_message(encr, key_matrix)
    print(f"Decrypted: {decr}")