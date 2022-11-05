# coding=utf-8
import os
import requests
from tqdm import tqdm
from mindsearch.utils.logger import Logger

logger = Logger(__name__).get_logger()


def download_url(url, save_path):
    """
    Download file from remote `url` to local directory, the file will be named `path` locally.
    """
    if os.path.dirname(save_path) != '':
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
    req = requests.get(url, stream=True, verify=False)
    if req.status_code != 200:
        logger.error("Exception when downloading {}, response code {}".format(url, req.status_code))
        req.raise_for_status()
        return

    download_filepath = f"{save_path}_part"
    with open(download_filepath, "wb") as writer:
        content_length = req.headers.get('Content-Length')
        total = int(content_length) if content_length is not None else None
        progress = tqdm(unit="B", total=total, unit_scale=True)
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:
                writer.write(chunk)
                progress.update(len(chunk))
                
    os.rename(download_filepath, save_path)
    progress.close()
