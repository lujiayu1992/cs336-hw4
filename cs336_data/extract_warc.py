from util import warc_to_txt

if __name__ == "__main__":
  test_task = "html_to_txt"
  WARC_path = "/workdir/data/warc.gz"
  if test_task == "html_to_txt":
    for txt in warc_to_txt(WARC_path, n_records=2, record_id=200):
      print(txt)
      print("content length: ", len(txt))
