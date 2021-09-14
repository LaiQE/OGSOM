'''
Description: In User Settings Edit
Author: Qianen
Date: 2021-09-14 18:50:21
LastEditTime: 2021-09-14 19:08:14
LastEditors: Qianen
'''
import sys
import pickle
from pathlib import Path
from ogsom.ogmanager import OGManager

SCALE = 0.001
STEP = 0.005


def process(obj_path):
    print('start to process file:', obj_path.as_posix())
    ogm = OGManager.from_obj_file(obj_path, step=STEP, scale=SCALE)
    pkl_path = obj_path.with_suffix('.pkl')
    if pkl_path.exists():
        print('remove file:', pkl_path.as_posix())
        pkl_path.unlink()
    with open(pkl_path, 'wb') as f:
        pickle.dump(ogm, f)


if __name__ == '__main__':
    for arg in sys.argv[1:]:
        p = Path(arg).resolve()
        if not p.exists():
            print(p.as_posix(), 'is not a file(suffix .obj) or folder')
        if p.is_dir():
            for fp in p.iterdir():
                if fp.is_file() and fp.suffix == '.obj':
                    process(fp)
        elif p.is_file() and p.suffix == '.obj':
            process(p)
