from app.core.config import settings
from app.core.constants import MODEL_PATH
from app.core.logger import app_logger

print(settings.app_name)
print(settings.port)
print(MODEL_PATH)

app_logger.info("Configuration loaded successfully")