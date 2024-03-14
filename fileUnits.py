import os


class File:
    def __init__(self):
        None

    
    # 得到要保存的文件路径，并判断是否放进文件夹里
    def get_file_path(self,file_name, putTofolder=False, folder=None):
        current_dir = os.getcwd()
        target_dir_path = current_dir

        if putTofolder == True:
            target_dir_name = folder
            target_dir_path = os.path.join(current_dir, target_dir_name)
            os.makedirs(target_dir_path, exist_ok=True)

        file_path = os.path.join(target_dir_path, file_name)
        return file_path
