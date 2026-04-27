import time
import base64
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend


def load_private_key(path: str):
    with open(path, 'rb') as f:
        return serialization.load_pem_private_key(
            f.read(), password=None, backend=default_backend()
        )


def sign_request(private_key, method: str, path: str) -> tuple[str, str]:
    """
    Returns (timestamp_ms_str, base64_signature).
    path must include /trade-api/v2 prefix and NO query string.
    """
    timestamp = str(int(time.time() * 1000))
    path_no_query = path.split('?')[0]
    message = f"{timestamp}{method}{path_no_query}".encode('utf-8')
    signature = private_key.sign(
        message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH
        ),
        hashes.SHA256()
    )
    return timestamp, base64.b64encode(signature).decode('utf-8')
