import re
from fastwarc.stream_io import FileStream, GZipStream
from fastwarc.warc import ArchiveIterator, WarcRecordType
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding
import fasttext

# --- Configuration & Constants ---
# Defining these at the top level makes the code cleaner and easier to update
EMAIL_PATTERN = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
EMAIL_PLACEHOLDER = "|||EMAIL_ADDRESS|||"

PHONE_PATTERN = (
    r"(?<!\d)(?:\+?1[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)"
)
PHONE_PLACEHOLDER = "|||PHONE_NUMBER|||"

# IPv4 strictly limits each octet to 0-255.
# \b ensures we don't accidentally match part of a longer number.
IP_PATTERN = r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
IP_PLACEHOLDER = "|||IP_ADDRESS|||"

LANG_MODEL = "/workdir/model/lid.176.bin"
HATE_MODEL = "/workdir/model/jigsaw_fasttext_bigrams_hatespeech_final.bin"
NSFW_MODEL = "/workdir/model/jigsaw_fasttext_bigrams_nsfw_final.bin"

def extract_text_from_html_bytes(html_bytes: bytes) -> str:
  """Extracts plain text from a byte string containing raw HTML.

  Args:
      html_bytes: A byte string containing raw HTML data.

  Returns:
      A Unicode string containing the extracted plain text.
  """
  encoding = detect_encoding(html_bytes)
  html_string = html_bytes.decode(encoding, errors="replace")
  extracted_text = extract_plain_text(html_string)
  return extracted_text


def warc_to_txt(warc_file: str, n_records: int = 10, record_id: int = 0):
  stream = GZipStream(FileStream(warc_file, "rb"))
  for i, record in enumerate(
      ArchiveIterator(stream, record_types=WarcRecordType.response)
  ):
    if i < record_id:
      continue
    if n_records <= 0:
      break
    n_records -= 1

    yield extract_text_from_html_bytes(record.reader.read())


# --- Common Core Logic ---
def _apply_mask(text: str, pattern: str, placeholder: str) -> tuple[str, int]:
  """Helper function to find regex patterns and replace them with a placeholder.

  Returns the masked string and the number of replacements made.
  """
  return re.subn(pattern, placeholder, text)


# --- Specific PII Functions ---
def mask_emails(text: str) -> tuple[str, int]:
  """Identifies and masks email addresses."""
  return _apply_mask(text, EMAIL_PATTERN, EMAIL_PLACEHOLDER)


def mask_phone_numbers(text: str) -> tuple[str, int]:
  """Identifies and masks common US phone numbers."""
  return _apply_mask(text, PHONE_PATTERN, PHONE_PLACEHOLDER)


def mask_ips(text: str) -> tuple[str, int]:
  """Identifies and masks IPv4 addresses."""
  return _apply_mask(text, IP_PATTERN, IP_PLACEHOLDER)

def classify_text(text: str, model_path: str) -> tuple[str, float]:
    """
    Generic fastText classifier wrapper.
    1. Cleans text (removes newlines).
    2. Loads model (Note: for scale, load models outside this function).
    3. Returns (label, confidence).
    """
    # fastText expects single lines for best performance
    clean_text = text.replace("\n", " ")
    
    # In a real pipeline, you'd load this once and pass the object, 
    # but for a direct replacement of your functions:
    model = fasttext.load_model(model_path)
    
    labels, scores = model.predict(clean_text, k=1)
    
    # Strip '__label__' prefix and return
    label = labels[0].replace("__label__", "")
    return label, float(scores[0])

# --- Refined Interface Functions ---
# These now just call the helper with different paths

def identify_language(text: str):
    return classify_text(text, LANG_MODEL)

def identify_nsfw(text: str):
    return classify_text(text, NSFW_MODEL)

def identify_toxic_speech(text: str):
    return classify_text(text, HATE_MODEL)