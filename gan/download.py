"""
Modified version of https://github.com/carpedm20/DCGAN-tensorflow/blob/master/download.py
Downloads the following datasets:
- Celeb-A
- LSUN
- MNIST
- Fashion MNIST
- Pokemon
"""

from __future__ import print_function

import click
import json
import os
import subprocess
import sys
import zipfile
import requests
from six.moves import urllib
from tqdm import tqdm
from gan.utils.misc import REPO_ROOT


def download(url, dirpath):
    filename = url.split('/')[-1]
    filepath = os.path.join(dirpath, filename)
    u = urllib.request.urlopen(url)
    f = open(filepath, 'wb')
    filesize = int(u.headers["Content-Length"])
    print("Downloading: %s Bytes: %s" % (filename, filesize))

    downloaded = 0
    block_sz = 8192
    status_width = 70
    while True:
        buf = u.read(block_sz)
        if not buf:
            print('')
            break
        else:
            print('', end='\r')
        downloaded += len(buf)
        f.write(buf)
        status = (
                ("[%-" + str(status_width + 1) + "s] %3.2f%%") % (
            '=' * int(float(downloaded) / filesize * status_width) + '>',
            downloaded * 100. / filesize)
        )
        print(status, end='')
        sys.stdout.flush()
    f.close()
    return filepath


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def save_response_content(response, destination, chunk_size=32 * 1024):
    total_size = int(response.headers.get('content-length', 0))
    with open(destination, "wb") as f:
        for chunk in tqdm(response.iter_content(chunk_size), total=total_size,
                          unit='B', unit_scale=True, desc=destination):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def unzip(filepath):
    print("Extracting: " + filepath)
    dirpath = os.path.dirname(filepath)
    with zipfile.ZipFile(filepath) as zf:
        zf.extractall(dirpath)
    os.remove(filepath)


def download_celeb_a(dirpath):
    data_dir = 'celebA'
    if os.path.exists(os.path.join(dirpath, data_dir)):
        print('Found Celeb-A - skip')
        return

    filename, drive_id = "img_align_celeba.zip", "0B7EVK8r0v71pZjFTYXZWM3FlRnM"
    save_path = os.path.join(dirpath, filename)

    if os.path.exists(save_path):
        print('[*] {} already exists'.format(save_path))
    else:
        download_file_from_google_drive(drive_id, save_path)

    with zipfile.ZipFile(save_path) as zf:
        zip_dir = zf.namelist()[0]
        zf.extractall(dirpath)
    os.remove(save_path)
    os.rename(os.path.join(dirpath, zip_dir), os.path.join(dirpath, data_dir))


def _list_categories(tag):
    url = 'http://lsun.cs.princeton.edu/htbin/list.cgi?tag=' + tag
    f = urllib.request.urlopen(url)
    return json.loads(f.read())


def _download_lsun(out_dir, category, set_name, tag):
    url = 'http://lsun.cs.princeton.edu/htbin/download.cgi?tag={tag}' \
          '&category={category}&set={set_name}'.format(**locals())
    print(url)
    if set_name == 'test':
        out_name = 'test_lmdb.zip'
    else:
        out_name = '{category}_{set_name}_lmdb.zip'.format(**locals())
    out_path = os.path.join(out_dir, out_name)
    cmd = ['curl', url, '-o', out_path]
    print('Downloading', category, set_name, 'set')
    subprocess.call(cmd)


def download_lsun(dirpath):
    data_dir = os.path.join(dirpath, 'lsun')
    if os.path.exists(data_dir):
        print('Found LSUN - skip')
        return
    else:
        os.mkdir(data_dir)

    tag = 'latest'
    # categories = _list_categories(tag)
    categories = ['bedroom']

    for category in categories:
        _download_lsun(data_dir, category, 'train', tag)
        _download_lsun(data_dir, category, 'val', tag)
    _download_lsun(data_dir, '', 'test', tag)


def download_dataset_from_url(data_dir, url_fmt, file_names):
    for file_name in file_names:
        url = url_fmt.format(file_name)
        out_path = os.path.join(data_dir, file_name)
        cmd = ['curl', url, '-o', out_path]
        click.secho('Downloading ... ' + url, fg='green')
        subprocess.call(cmd)


def prepare_dataset_dir(dataset_name, folder_name='data'):
    data_dir = os.path.join(REPO_ROOT, folder_name, dataset_name)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
    elif click.confirm("Overwrite %s?" % data_dir, abort=True):
        print("Continue downloading files to", data_dir)
    else:
        sys.exit(0)

    return data_dir


VALID_DATASETS = ['celebA', 'lsun', 'mnist', 'fashion-mnist']


@click.command()
@click.argument('name', nargs=1, type=click.Choice(VALID_DATASETS))
def main(name):
    """
    The input dataset 'name' should be one of:
        ['celebA', 'lsun', 'mnist', 'fashion-mnist']
    """
    assert name in ['celebA', 'lsun', 'mnist', 'fashion-mnist']
    prepare_dataset_dir(name)
    data_dir = os.path.join(REPO_ROOT, name)

    if name == 'celebA':
        download_celeb_a(data_dir)
    elif name == 'lsun':
        download_lsun(data_dir)
    elif name == 'mnist':
        url_fmt = 'http://yann.lecun.com/exdb/mnist/{}'
        file_names = ['train-images-idx3-ubyte.gz',
                      'train-labels-idx1-ubyte.gz',
                      't10k-images-idx3-ubyte.gz',
                      't10k-labels-idx1-ubyte.gz']
        download_dataset_from_url(data_dir, url_fmt, file_names)
    elif name == 'fashion-mnist':
        url_fmt = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/{}'
        file_names = ['train-images-idx3-ubyte.gz',
                      'train-labels-idx1-ubyte.gz',
                      't10k-images-idx3-ubyte.gz',
                      't10k-labels-idx1-ubyte.gz']
        download_dataset_from_url(data_dir, url_fmt, file_names)
    elif name == 'pokemon':
        raise NotImplementedError()


if __name__ == '__main__':
    main()
