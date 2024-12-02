import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

print("Qdrant API Key:", os.getenv('Qdrant_API_KEY'))
print("Qdrant URL:", os.getenv('Qdrant_URL'))
print("Collection Name:", os.getenv('Collection_Name'))
