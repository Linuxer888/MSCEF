import os
import cv2 as cv



def get_filelist(dir_path):
    return os.listdir(dir_path)

def after_image(src_filelist, src_dir, res_filelist, res_dir, save_dir):
    for srcimg, resimg in zip(src_filelist, res_filelist):
        src1 = cv.imread(src_dir + srcimg)
        src2 = cv.imread(res_dir + resimg)
        out = cv.cvtColor(src2, cv.COLOR_BGR2GRAY)
        # #结果图映射为0/1
        ret, out = cv.threshold(out, 0, 1, cv.THRESH_BINARY)
        h, w = out.shape
        for i in range(h):
            for j in range(w):
                if out[i][j] == 1:
                    for c in range(3):
                        src1[i, j, c] = src2[i, j, c]
        print(f"===============save img {resimg} to {save_dir + resimg} ====================")
        cv.imwrite(save_dir + resimg, src1)

if __name__ == '__main__':
    #验证集原图相信息
    src_list = get_filelist("../datasets/laser/leftImg8bit/val/allll")
    src_dir = "../datasets/laser/leftImg8bit/val/allll/"
    #验证效果相关信息
    res_list = get_filelist("../runs/pred_pic/unet3plus_dropencoding_mbga/")
    res_dir = "../runs/pred_pic/unet3plus_dropencoding_mbga/"
    #保存路径
    save_dir = "../runs/result/unet3plus_dropencoding_mbga/"
    after_image(src_list, src_dir, res_list, res_dir, save_dir)
    print("==================end===================")
