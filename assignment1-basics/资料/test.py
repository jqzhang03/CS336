def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])

if __name__ == "__main__" :
    print(decode_utf8_bytes_to_str_wrong("hello,你好".encode("utf-8")))