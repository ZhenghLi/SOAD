# calculate and visualize CNR and MSR of normal B-scans
import cv2
import os
import numpy as np
import xml.etree.ElementTree as ET

exp_root = 'data'
label_root = 'data/ROIs'
exp_names = ['OCTA']
Volumes = ['Volume0']

if not os.path.exists('mark_cnr_msr_normal'):
    os.mkdir('mark_cnr_msr_normal')

for exp_name in exp_names:
    mark_root = os.path.join('mark_cnr_msr_normal', exp_name)

    if not os.path.exists(mark_root):
        os.mkdir(mark_root)

    cnr_list_all = []
    msr_list_all = []
    font = cv2.FONT_HERSHEY_SIMPLEX

    exp_path = os.path.join(exp_root, exp_name)
    
    for v in Volumes:
        img_dir = os.path.join(exp_path, v)
        
        label_dir = os.path.join(label_root, v + '_normal')

        mark_dir = os.path.join('mark_cnr_msr_normal', os.path.join(exp_name, v))
        if not os.path.exists(mark_dir):
            os.mkdir(mark_dir)

        files = os.listdir(label_dir)

        cnr_volume = []
        msr_volume = []

        for f in files:
            cnrs = []
            msrs = []

            img_f = f[:-3] + 'tif'
            img_path = os.path.join(img_dir, img_f)
            label_path = os.path.join(label_dir, f)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype('float')
            img_marked = cv2.imread(img_path)

            tree = ET.parse(label_path)
            root = tree.getroot()

            roi_objs = root.findall(".//object[name= \'roi\']")
            bg_objs = root.findall(".//object[name= \'bg\']")

            assert len(roi_objs) == 9
            assert len(bg_objs) == 5

            bg_cat = None

            for bg_obj in bg_objs:
                bg_box = bg_obj.find('bndbox')

                bg_xmin = int(bg_box.find('xmin').text)
                bg_xmax = int(bg_box.find('xmax').text)
                bg_ymin = int(bg_box.find('ymin').text)
                bg_ymax = int(bg_box.find('ymax').text)

                bg = img[bg_ymin:bg_ymax, bg_xmin:bg_xmax]

                img_marked = cv2.rectangle(img_marked, (bg_xmin,bg_ymin), (bg_xmax,bg_ymax), (255, 0, 0), 1)

                if bg_cat is not None:
                    bg_cat = np.concatenate((bg_cat, bg.reshape(-1)))
                else:
                    bg_cat = bg.reshape(-1)

            bg_mean = bg_cat.mean()
            bg_std = bg_cat.std()

            for roi_obj in roi_objs:

                roi_box = roi_obj.find('bndbox')

                roi_xmin = int(roi_box.find('xmin').text)
                roi_xmax = int(roi_box.find('xmax').text)
                roi_ymin = int(roi_box.find('ymin').text)
                roi_ymax = int(roi_box.find('ymax').text)

                roi = img[roi_ymin:roi_ymax, roi_xmin:roi_xmax]
                roi_mean = roi.mean()
                roi_std = roi.std()

                cnr_r = np.abs(roi_mean-bg_mean) / np.sqrt(0.5*(roi_std**2 + bg_std**2))

                img_marked = cv2.rectangle(img_marked, (roi_xmin,roi_ymin), (roi_xmax,roi_ymax), (0, 0, 255), 1)
                img_marked = cv2.putText(img_marked, str(round(cnr_r, 2)), (roi_xmin, roi_ymin-5), font, 0.4, (0, 0, 255), 1)
                # img_marked = cv2.putText(img_marked, str(i), (roi_xmin, roi_ymax+10), font, 0.4, (0, 0, 255), 1)

                cnrs.append(cnr_r)
                cnr_list_all.append(cnr_r)

                msr_r = roi_mean / roi_std
                img_marked = cv2.putText(img_marked, str(round(msr_r, 2)), (roi_xmin, roi_ymax+10), font, 0.4, (0, 255, 0), 1)

                msrs.append(msr_r)
                msr_list_all.append(msr_r)
            
            cnr_img = np.mean(cnrs)
            # print(f, cnr_img)
            cnr_volume.append(cnr_img)

            msr_img = np.mean(msrs)
            # print(f, msr_img)
            msr_volume.append(msr_img)

            mark_img_path = os.path.join(mark_dir, img_f[:-3] + 'png')

            cv2.imwrite(mark_img_path, img_marked)

        cnr_volume = np.mean(cnr_volume)
        # print(v, cnr_volume)
        msr_volume = np.mean(msr_volume)
        # print(v, msr_volume)

    cnr = np.mean(cnr_list_all)
    print(exp_name, 'CNR:', cnr)

    msr = np.mean(msr_list_all)
    print(exp_name, 'MSR:', msr)