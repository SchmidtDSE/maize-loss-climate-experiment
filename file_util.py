"""Utilities to interact with files.

License:
    BSD
"""
import os
import time

import boto3
import botocore


def save_file_tmp(source_location, source_filename, access_key='', access_secret='',
    local_filename=None):
    """Ensure a file is available locally.
    
    Ensure a file is available locally and, if it is transferred from remote, place it in a
    temporary filesystem.

    Args:
        source_location: The name of the S3 bucket where the resource can be found or the path to
            where the file can be found locally. Interpreted as a local path if access_key or
            access_secret are not provided (an empty string).
        source_filename: The filename of the resource to make available.
        access_key: The AWS or remote access key or empty string if the file should be found
            locally. Defaults to empty string.
        access_secret: The AWS or remote access secret or empty string if the file should be found
            locally. Defaults to empty string.
        local_filename: The filename to use for this resource when saving locally. This is a
            suggestion and may be ignored depending on host system and conditions.
    
    Returns:
        Full local path where the resource can be found.
    """

    using_local = access_key == '' or access_secret == ''

    if using_local:
        return os.path.join(source_location, source_filename)

    if local_filename is None:
        local_filename = source_filename

    def inner(local_filename):
        s3 = boto3.resource(
            's3',
            aws_access_key_id=access_key,
            aws_secret_access_key=access_secret
        )

        s3_object = s3.Object(source_location, source_filename)

        temp_file_path = '/tmp/' + source_filename

        try:
            s3_object.download_file(temp_file_path)
        except botocore.exceptions.ClientError:
            raise RuntimeError('Failed to download ' + source_filename)

        assert os.path.isfile(temp_file_path)

        return temp_file_path

    try:
        return inner(local_filename)
    except:
        time.sleep(5)
        return inner(local_filename)


def get_bucket_files(source_location, access_key='', access_secret=''):
    """Get all of the files in a location where there is a collection of resources.
    
    Args:
        source_location: The name of the S3 buckets where the resource can be found or the path to
            where the files can be found locally. Interpreted as a local path if access_key or
            access_secret are empty strings.
        access_key: The AWS or remote access key or empty string if the collection should be found
            locally. Defaults to empty string.
        access_secret: The AWS or remote access secret or empty string if the collection should be
            found locally. Defaults to empty string.
    
    Returns:

    """
    using_local = access_key == '' or access_secret == ''

    if using_local:
        all_items = os.listdir(source_location)
        all_files = filter(lambda x: os.path.isfile(x), all_items)
        return list(all_files)
    else:
        s3 = boto3.resource(
            's3',
            aws_access_key_id=access_key,
            aws_secret_access_key=access_secret
        )

        bucket = s3.Bucket(source_location)

        all_objects = bucket.objects.all()
        all_keys = map(lambda x: x.key, all_objects)

        return all_keys


def remove_temp_file(temp_file_path, access_key='', access_secret=''):
    """Remove a local temporary file if in a temporary filesystem location.
    
    Remove a local temporary file if in a temporary filesystem location or do nothing if the file is
    in the regular file system.

    Args:
        temp_file_path: Full path to the temporary file.
        access_key: The AWS or remote access key or empty string if the file was originally found
            locally. Defaults to empty string.
        access_secret: The AWS or remote access secret or empty string if the file was originally
            found locally. Defaults to empty string.
    """
    using_local = access_key == '' or access_secret == ''

    if using_local:
        return
    else:
        os.remove(temp_file_path)
