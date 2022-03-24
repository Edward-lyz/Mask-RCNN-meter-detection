# -*- coding: utf-8 -*-
#定义一个删除非5的整数次训练的权重
import re
import os
import argparse
def check_file(filePath):
	# 统计删除总数
    count = 0
    # 
    for file_path, empty_list, file_name_list in os.walk(filePath):
        # file_name_list该列表是存放目标目录中所有文件名
        for file_name in file_name_list:
            # 正则匹配需要删除的文件--根据需求修改正则表达式
            if re.match(r'mask_rcnn_shapes_0[0-9][0-9][0-9]', file_name):
                # 删除匹配到的文件
                if not re.match(r'mask_rcnn_shapes_0[0-9][0-9][0,5]', file_name):
                  os.remove(file_path + file_name)
                  # 每删除一个文件＋1
                  count += 1

    return count


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='file_path')
    parser.add_argument('--filepath', type=str, default = None)
    args = parser.parse_args()
    cnt=check_file(args.filepath+"/")
    print("Successfully delete "+str(cnt)+" checkpoints!")
    