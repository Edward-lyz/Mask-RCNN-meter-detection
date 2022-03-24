def img_16to8():
    from PIL import Image
    import numpy as np
    import shutil
    import os

    src_dir = r'D:\OneDrive\STUDY\大四下\毕设\Mask_RCNN\dataset\test\labelme_json'
    dest_dir = r'D:\OneDrive\STUDY\大四下\毕设\Mask_RCNN\dataset\test\cv2_mask'
    for child_dir in os.listdir(src_dir):
        new_name = child_dir.split('_')[0] + '.png'
        old_mask = os.path.join(os.path.join(src_dir, child_dir), 'label.png')
        img = Image.open(old_mask)
        # img = Image.fromarray(np.uint8(np.array(img))) #由于新版本labelme不需要转化了，所以这里去掉
        new_mask = os.path.join(dest_dir, new_name)
        img.save(new_mask)

if __name__=='__main__':
    img_16to8()