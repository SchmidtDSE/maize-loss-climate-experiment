import os
import time

import boto3
import botocore


def save_file_tmp(bucket_name, tiff_filename, access_key, access_secret,
    local_filename=None):

    if local_filename is None:
        local_filename = tiff_filename

    def inner(local_filename):
        s3 = boto3.resource(
            's3',
            aws_access_key_id=access_key,
            aws_secret_access_key=access_secret
        )

        s3_object = s3.Object(bucket_name, tiff_filename)

        temp_file_path = '/tmp/' + tiff_filename

        try:
            s3_object.download_file(temp_file_path)
        except botocore.exceptions.ClientError as e:
            raise RuntimeError('Failed to download ' + tiff_filename)

        assert os.path.isfile(temp_file_path)

        return temp_file_path

    try:
        return inner(local_filename)
    except:
        time.sleep(5)
        return inner(local_filename)


def get_bucket_files(bucket_name, access_key, access_secret):
    s3 = boto3.resource(
        's3',
        aws_access_key_id=access_key,
        aws_secret_access_key=access_secret
    )

    bucket = s3.Bucket(bucket_name)

    all_objects = bucket.objects.all()
    all_keys = map(lambda x: x.key, all_objects)

    return all_keys
