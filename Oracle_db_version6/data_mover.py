# -*- coding: utf-8 -*-

"""
Data Moving and Deleting Module, Made by BJH
"""


import os
import shutil

def copytree(src, dst, symlinks=False, ignore=None):
    try:
        if not os.path.exists(dst):
            os.makedirs(dst)
        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(dst, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, symlinks, ignore)
            else:
                shutil.copy2(s, d)
        print('file copy finished')
    except FileExistsError:
        pass


def on_copytree(src, dst, chmod_path, symlinks=False, ignore=None):
    try:
        print('file copy from ',src, ' to',  dst)
        if not os.path.exists(dst):
            os.makedirs(dst)

        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(dst, item)


            if os.path.isdir(s):
                shutil.copytree(s, d, symlinks, ignore)
            else:
                shutil.copy2(s, d)

        for root, dirs, files in os.walk(chmod_path):
            for d in dirs:
                os.chmod(os.path.join(root, d), mode=0o777)
            for f in files:
                os.chmod(os.path.join(root, f), mode=0o777)

        print('file copy finished')
    except FileExistsError:
        pass


def nas_to_dlserver(frompath, topath):
    copytree(frompath, topath)


def dlserver_to_nas(frompath, topath):
    shutil.move(frompath, topath)


def _data_delete(path):
    shutil.rmtree(path)


