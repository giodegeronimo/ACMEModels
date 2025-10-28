import logging
import os
from pathlib import Path

from src.logging_config import configure_logging
from src.utils.env import load_dotenv
from src.metrics.tree_score import TreeScoreMetric

# Ensure logging is configured for this script run.
# Load local .env next to this file and set defaults if missing.
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
os.environ.setdefault("LOG_LEVEL", "2")
os.environ.setdefault("GEN_AI_STUDIO_API_KEY", "AIzaSyDUgg_XoEdry3pFusXVpUKptE-JMZD1qQE")
os.environ['ACME_IGNORE_FAIL'] = '1'
os.environ["LOG_FILE"] = str(BASE_DIR / "data" / "acme.log")
print(f"[temp_test] LOG_LEVEL={os.environ.get('LOG_LEVEL')} -> file={os.environ['LOG_FILE']}")
configure_logging()
_LOGGER = logging.getLogger(__name__)
_LOGGER.info(
    "temp_test starting; logging to %s",
    str(Path(os.environ["LOG_FILE"]).resolve()),
)
print(f"[temp_test] ACME_IGNORE_FAIL={os.environ.get('ACME_IGNORE_FAIL')}")


# after testing, the biggest issue is that the dataset quality metric will
# always return 0.0 for since they don't have datasets attached.
# Also, the bus_factor and performance metrics always return 0.5 since
# they are just placehodlers right now.

metric = TreeScoreMetric()

print()
print(f"DeepSeek-OCR Test")
url_record = {"hf_url": "https://huggingface.co/deepseek-ai/DeepSeek-OCR"}
score = metric.compute(url_record)
_LOGGER.info("TreeScore for %s: %.3f", url_record.get("hf_url"), float(score))
print(f"TreeScore for {url_record['hf_url']}: {score}")
print()

print(f"Google BERT Test")
url_record = {"hf_url": "https://huggingface.co/google-bert/bert-base-uncased"}
score = metric.compute(url_record)
_LOGGER.info("TreeScore for %s: %.3f", url_record.get("hf_url"), float(score))
print(f"TreeScore for {url_record['hf_url']}: {score}")
print()

print(f"Nanonets OCR2 3B Test")
url_record = {"hf_url": "https://huggingface.co/nanonets/Nanonets-OCR2-3B"}
score = metric.compute(url_record)
_LOGGER.info("TreeScore for %s: %.3f", url_record.get("hf_url"), float(score))
print(f"TreeScore for {url_record['hf_url']}: {score}")
