#!/usr/bin/python3

import argparse
import json
import requests
import getpass
import os
import tarfile
from urllib.parse import urlparse
from tqdm import tqdm
from joblib import Parallel, delayed
import shutil

BASE_URL = "https://o9k5xn5546.execute-api.us-east-1.amazonaws.com/v1/archives/v1.0/"

FILES_TRAIN = ["v1.0-trainval_meta.tgz",
         "v1.0-trainval01_blobs.tgz",
         "v1.0-trainval02_blobs.tgz",
         "v1.0-trainval03_blobs.tgz",
         "v1.0-trainval04_blobs.tgz",
         "v1.0-trainval05_blobs.tgz",
         "v1.0-trainval06_blobs.tgz",
         "v1.0-trainval07_blobs.tgz",
         "v1.0-trainval08_blobs.tgz",
         "v1.0-trainval09_blobs.tgz",
         "v1.0-trainval10_blobs.tgz",]

FILES_TEST = [
    "v1.0-test_meta.tgz",
    "v1.0-test_blobs.tgz",
]

HEADERS = {
    "content-type": "application/x-amz-json-1.1",
    "x-amz-target": "AWSCognitoIdentityProviderService.InitiateAuth",
}

COGNITO_CLIENT_ID = "7fq5jvs5ffs1c50hd3toobb3b9"

DOWNLOAD_CHUNK_SIZE = 1000 # 1 kB

def userlogin(username: str, password: str) -> str:
    data = {
    "AuthFlow": "USER_PASSWORD_AUTH",
    "AuthParameters": {
        "PASSWORD": password,
        "USERNAME": username
    },
    "ClientId": COGNITO_CLIENT_ID,
    "ClientMetadata": {}
    }
    
    response = requests.post(
        "https://cognito-idp.us-east-1.amazonaws.com/",
        headers=HEADERS,
        data=json.dumps(data)
    )
    
    if response.status_code != requests.codes.ok:
        print("Error logging in. Check username and password.")
        exit(1)

    return response.json()['AuthenticationResult']['IdToken']

def download_file(token: str, url: str, save_dir: str, n: int):
    headers = {
        "authorization": f"Bearer {token}"
    }

    query_params = {
        "region": "us",
        "project": "nuScenes",
    }

    response = requests.get(
        url,
        headers=headers,
        params=query_params
    )

    download_url = response.json()["url"]

    # download file and unzip to save_dir
    filename = os.path.basename(urlparse(download_url).path)
    tar_save_path = os.path.join(save_dir, filename)

    r = requests.get(download_url, stream=True)
    
    with open(tar_save_path, 'wb') as fd:
        progress_bar = tqdm(total=int(r.headers['Content-Length']), unit='B', unit_scale=True, unit_divisor=DOWNLOAD_CHUNK_SIZE, position=n, desc=f"{filename}", colour="green")
        for chunk in r.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE): # download in 1 KB chunks
            fd.write(chunk)
            progress_bar.update(len(chunk))

    # if filename.endswith(".tgz"):
    #     with tarfile.open(tar_save_path, 'r:gz') as tar:
    #         for member in tqdm(iterable=tar.getmembers(), total=len(tar.getmembers()), position=n, desc=f"Extracting {filename}", colour="green"):
    #             tar.extract(member=member, path=save_dir)
    #     os.remove(tar_save_path) # remove tar file
    # else:
    #     exit(1)

def main():
    parser = argparse.ArgumentParser(description="Download NuScenes dataset from CLI")
    parser.add_argument("savedir",
        help="Directory to save dataset"
        )
    
    args = parser.parse_args()
    save_dir = os.path.abspath(args.savedir)
    create_dir = False

    if not os.path.exists(save_dir):
        create_dir = True
    else:
        if os.listdir(save_dir):
            print("Directory is not empty. Aborting.")
            exit(1)

    try:
        username = input("nuScenes username [email]: ")
        password = getpass.getpass(prompt=f"nuScenes password: ")
    except KeyboardInterrupt:
        print("Exiting.")
        exit(1)

    if create_dir:
        os.makedirs(save_dir)
        
    with requests.Session() as s:
        # login and get token
        auth_token = userlogin(username, password)
        files = FILES_TRAIN + FILES_TEST
        file_urls = [BASE_URL + file for file in files]

        # download files and uncompress in parallel
        try:
            Parallel(n_jobs=-1, backend='threading')( 
            delayed(download_file)(auth_token, url, save_dir, n) for n, url in enumerate(file_urls)
            )
            print("Done.")
        except KeyboardInterrupt:
            print("Stopping download.")
            shutil.rmtree(save_dir)
            exit(1)

if __name__ == "__main__":
    main()
