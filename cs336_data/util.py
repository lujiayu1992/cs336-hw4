from fastwarc.stream_io import FileStream, GZipStream
from fastwarc.warc import ArchiveIterator, WarcRecordType
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding


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
