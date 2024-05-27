import os

file_dir = "OfficeHome_dataset/test/"

for (_, directory, _) in os.walk(file_dir):
    for dir_ in directory:
        for (path, dir, files) in os.walk(f'{file_dir}/{dir_}'):
            for dirname in dir:
                os.rename(f'{file_dir}/{dir_}/{dirname}', f'{file_dir}/{dirname}_{dir_}')
