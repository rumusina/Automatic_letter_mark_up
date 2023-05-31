import os, fnmatch
import cv2 as cv
import numpy as np
import math
import shutil

drawing = False
wait_letter = False
ix,iy,iw,ih = -1,-1,-1,-1
out_img = None

def mouse_events(event, x, y, flags, param):
    global ix, iy, iw, ih, drawing, out_img, wait_letter, update_scene
    if event == cv.EVENT_LBUTTONDOWN:
        if drawing:
            iw = abs(x - ix)
            ih = abs(y - iy)
            ix = min(x, ix)
            iy = min(y, iy)
            drawing = False
            wait_letter = True
            cv.rectangle(out_img, (ix, iy), (x, y), (0, 0, 255), 2)
            cv.putText(out_img, 'Press letter:' , (4, 4), cv.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 1, cv.LINE_AA )
            update_scene = True
        else:
            ix = x
            iy = y
            drawing = True
            wait_letter = False

def getBoxByI(label):
    top = int(img_height * (label['y_center'] - label['height'] / 2))
    left = int(img_width * (label['x_center'] - label['width'] / 2))
    bottom = int(img_height * (label['y_center'] + label['height'] / 2))
    right = int(img_width * (label['x_center'] + label['width'] / 2))
    return left, top, right, bottom

def getMassCenter(img):
    x = img.shape[1] / 2
    y = img.shape[0] / 2
    tmp1 = 0.0
    tmp2 = 0.0
    tmp3 = 0.0
    for i in range(img.shape[1]):
        for j in range(img.shape[0]):
            pix = 255 - img[j, i, 0]
            tmp1 = tmp1 + pix * (i - x)
            tmp2 = tmp2 + pix * (j - y)
            tmp3 = tmp3 + pix
    return tmp1 / tmp3 , tmp2 / tmp3, tmp3 / (img.shape[1] * img.shape[0])

def find(pattern, path):
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                return os.path.join(root, name)
    return ''

def sharpFilter(roi):
    kernel =np.array([[-1, -1, -1],
                      [-1, 9, -1],
                      [-1, -1, -1]])
    shrp_img = cv.filter2D(roi,-1,kernel)
    return shrp_img

def findLetter(label, img, tmpl, k):
    w = img.shape[1]
    h = img.shape[0]
    add_area = 25
    top = int(h * (label['y_center'] - label['height'] / 2))
    left = int(w * (label['x_center'] - label['width'] / 2))
    bottom = int(h * (label['y_center'] + label['height'] / 2))
    right = int(w * (label['x_center'] + label['width'] / 2))
    _tmpl = tmpl
    _img = img[top- add_area:bottom + add_area,left- add_area:right + add_area]
    # if len(_tmpl.shape) > 2:
    #     _tmpl = cv.cvtColor(_tmpl, cv.COLOR_BGR2GRAY)
    # if len(_img.shape) > 2:
    #     _img = cv.cvtColor(_img, cv.COLOR_BGR2GRAY)
    _tmpl = cv.resize(_tmpl, ((int)(_tmpl.shape[1] * k), int(_tmpl.shape[0] * k)))
    res = cv.matchTemplate(tmpl, _img , cv.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    new_left = max_loc[0] + left - add_area
    new_top = max_loc[1] + top - add_area
    return new_left / w + label['width'] / 2, new_top / h + label['height'] / 2

# tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
# tess_option = '--oem 1--psm 6 -c tessedit_char_whitelist=0123456789ABCDEFJHIJKLMNOPQRSTUVWXYZ_'
rootdirname = 'data/'
imgdirname = 'images2/'
lbldirname = 'images2/'
goodimgdirname = 'images/'
goodlbldirname = 'labels/'
classesfilepath = 'classes.txt'

# iterate over files in
# that directory
classes = []
labelsfilecontent = []
need_save = False
s_line = []
labels = []
prev_labels = []
prev_images = []
with open(os.path.join(rootdirname,classesfilepath)) as classesfile:
    classes = [line.rstrip() for line in classesfile]
for dirname in os.listdir(os.path.join(rootdirname,imgdirname)):
    #tboxfile = find('_box*.txt', os.path.join(rootdirname,imgdirname,dirname))
    tboxfile = os.path.join(rootdirname,imgdirname,dirname, '_box.txt')
    #r = int(int(os.path.split(tboxfile)[1][5:-4])/2)
    #D = 2000 - r
    letter_mode = False
    change_letter = False
    i = 0
    for filename in os.listdir(os.path.join(rootdirname,imgdirname, dirname)):
        scale = 1
        imgfilepath = os.path.join(rootdirname, imgdirname, dirname, filename)
        labelfilename = os.path.splitext(filename)[0] + '.txt'
        lblfilepath = os.path.join(rootdirname, lbldirname, dirname, labelfilename)
        goodimgfilepath = os.path.join(rootdirname, goodimgdirname, filename)
        goodlblfilepath = os.path.join(rootdirname, goodlbldirname, labelfilename)
        if os.path.isfile(imgfilepath) and os.path.splitext(filename)[1] == '.png':
            labels = []
            img = cv.imread(imgfilepath)
            k = 1920 / img.shape[1]
            img= cv.resize(img, ((int)(img.shape[1] * k), int(img.shape[0] * k)))
            img_width = img.shape[1]
            img_height = img.shape[0]
            if not os.path.isfile(lblfilepath) and os.path.isfile(tboxfile):
                fileA = open(tboxfile, 'rb')
                fileB = open(lblfilepath, 'wb')
                shutil.copyfileobj(fileA, fileB)
                fileA.close()
                fileB.close()
            update_scene = True
            while True:
                if update_scene:
                    labels = []
                    update_scene = False
                    out_img = img.copy()
                    if os.path.isfile(lblfilepath):
                        with open(lblfilepath) as labelsfile:
                            for line in labelsfile:
                                s_line = line.rstrip().split(' ')
                                if (len(s_line) == 5):
                                    labels.append({'class': int(s_line[0]), 'x_center': float(s_line[1]), 'y_center': float(s_line[2]),
                                                          'width': float(s_line[3]), 'height': float(s_line[4])})
                        boxes = []
                        for j, label in enumerate(labels):
                            top = int(img_height * (label['y_center'] - label['height'] / 2))
                            left = int(img_width * (label['x_center'] - label['width'] / 2))
                            bottom = int(img_height * (label['y_center'] + label['height'] / 2))
                            right = int(img_width * (label['x_center'] + label['width'] / 2))
                            boxes.append({'class': classes[label['class']], 'left': left, 'top': top,'right': right,'bottom': bottom})
                            clr = (255,0,0)
                            if i == j and (letter_mode or change_letter):
                                clr = (255,255,0)
                                x1, y1, x2, y2 = getBoxByI(label)
                                xx, yy, mass = getMassCenter(img[y1:y2, x1:x2])
                                cv.putText(out_img, str(round(xx, 1)) + '/' + str(round(yy, 1)) + '/' + str(round(mass)), (left, top - 38),
                                           cv.FONT_HERSHEY_SIMPLEX,
                                           0.3, (12, 255, 12), 1, cv.LINE_AA)
                                # if len(prev_images)>0:
                                #     out_img[0:prev_images[i].shape[0], 0:prev_images[i].shape[1]] = prev_images[i]
                            cv.rectangle(out_img, (left, top), (right, bottom), clr, 2)
                            cv.putText(out_img, classes[label['class']] , (left , top - 4), cv.FONT_HERSHEY_SIMPLEX,0.5, (255,255,0), 1, cv.LINE_AA )
                            cv.putText(out_img, str(label['class']), (left+12, top - 12), cv.FONT_HERSHEY_SIMPLEX, 0.3,
                                       (12, 12, 255), 1, cv.LINE_AA)
                            cv.putText(out_img, str(j + 1), (left + 12, top - 4), cv.FONT_HERSHEY_SIMPLEX,
                                       0.3,(12, 255, 255), 1, cv.LINE_AA)

                    # tess = text(img)
                    if wait_letter:
                        cv.rectangle(out_img, (ix, iy), (ix+iw, iy+ih), (0, 0, 255), 2)
                        cv.putText(out_img, 'Press letter:', (4, 24), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 1,
                                   cv.LINE_AA)
                    if change_letter:
                        cv.putText(out_img, 'New letter:', (44, 24), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 1,
                                   cv.LINE_AA)
                    if letter_mode:
                        cv.putText(out_img, 'I: ' + str(i + 1), (2, 24), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 1, cv.LINE_AA)
                    cv.imshow(filename, out_img)
                    # cv.moveWindow(filename, 0 , 0)
                    # cv.resizeWindow(filename, out_img.shape[1], out_img.shape[0])
                    cv.setMouseCallback(filename, mouse_events)
                cvkey = cv.waitKey(10)
                if cvkey == 27:
                    cv.destroyWindow(filename)
                    break
                if cvkey == ord('u'):
                    update_scene = True
                if not wait_letter:
                    # if cvkey == ord('o'):
                    #     update_scene = True
                    #     labels_tess = pytesseract.image_to_boxes(img, config= tess_option, lang="eng").split('\n')
                    #     with open(lblfilepath, 'w') as f:
                    #         for lt in labels_tess:
                    #             l = lt.split(' ')
                    #             try:
                    #                 c = str(classes.index(l[0]))
                    #                 left = float(l[1])
                    #                 top = float(l[2])
                    #                 width = float(l[3]) - left
                    #                 height = float(l[4]) - top
                    #                 x_center = str((left + width / 2.0)/img_width)
                    #                 y_center = str(1 - (top + height / 2.0) / img_height)
                    #                 width = str(width/img_width)
                    #                 height = str(height / img_height)
                    #                 f.write(str(c) + ' ' + x_center + ' ' + y_center + ' ' + width + ' ' + height + '\n')
                    #             except:
                    #                 print("ooops")
                    if cvkey == ord('b'):
                        fileA = open(tboxfile, 'wb')
                        fileB = open(lblfilepath, 'rb')
                        shutil.copyfileobj(fileB, fileA)
                        fileA.close()
                        fileB.close()
                    if cvkey == ord('r'):
                        if os.path.isfile(lblfilepath):
                            os.remove(lblfilepath)
                        if os.path.isfile(imgfilepath):
                            os.remove(imgfilepath)
                        cv.destroyWindow(filename)
                        break
                if len(labels)>0:
                    if not wait_letter:
                        if cvkey == ord('f'):
                            if letter_mode:
                                x, y = findLetter(labels[i], img,  prev_images[i], scale)
                                labels[i]['x_center'] = x
                                labels[i]['y_center'] = y
                            else:
                                for ii, l in enumerate(labels):
                                    x, y = findLetter(l, img, prev_images[ii], scale)
                                    l['x_center'] = x
                                    l['y_center'] = y
                            update_scene = True
                            need_save = True
                        if cvkey == ord('l'):
                            update_scene = True
                            change_letter = True
                        if cvkey == ord('c') and not wait_letter and letter_mode:
                            del labels[i]
                            labels = sorted(labels, key=lambda x: x['x_center'])
                            need_save = True
                            update_scene = True
                        if cvkey == ord('g') and not wait_letter:
                            prev_images.clear()
                            for label in labels:
                                top = int(img_height * (label['y_center'] - label['height'] / 2))
                                left = int(img_width * (label['x_center'] - label['width'] / 2))
                                bottom = int(img_height * (label['y_center'] + label['height'] / 2))
                                right = int(img_width * (label['x_center'] + label['width'] / 2))
                                prev_images.append(img[top:bottom, left:right])
                            # os.rename(imgfilepath, goodimgfilepath)
                            # os.rename(lblfilepath, goodlblfilepath)
                            # cv.destroyWindow(labelfilename)
                            # break
                        if cvkey == ord('a'):
                            update_scene = True
                            if letter_mode:
                                labels[i]['x_center'] = labels[i]['x_center'] - 0.001
                            else:
                                for l in labels:
                                    l['x_center'] = l['x_center'] - 0.001
                            need_save = True
                        if cvkey == ord('d'):
                            update_scene = True
                            if letter_mode:
                                labels[i]['x_center'] = labels[i]['x_center'] + 0.001
                            else:
                                for l in labels:
                                    l['x_center'] = l['x_center'] + 0.001
                            need_save = True
                        if cvkey == ord('w'):
                            update_scene = True
                            if letter_mode:
                                labels[i]['y_center'] = labels[i]['y_center'] - 0.005
                            else:
                                for l in labels:
                                    l['y_center'] = l['y_center'] - 0.005
                            need_save = True
                        if cvkey == ord('s'):
                            update_scene = True
                            if letter_mode:
                                labels[i]['y_center'] = labels[i]['y_center'] + 0.005
                            else:
                                for l in labels:
                                    l['y_center'] = l['y_center'] + 0.005
                            need_save = True
                        if cvkey == ord('+'):
                            update_scene = True
                            if letter_mode:
                                k = 1.01
                                labels[i]['height'] = labels[i]['height']  * k
                                labels[i]['width'] = labels[i]['width']  * k
                            else:
                                k = 1.001
                                scale = scale * k
                                for l in labels:
                                    l['x_center'] = (l['x_center'] - 0.5) * k + 0.5
                                    l['y_center'] = (l['y_center'] - 0.5) * k + 0.5
                                    l['height'] = l['height'] * k
                                    l['width'] = l['width'] * k
                            need_save = True
                        if cvkey == ord('-'):
                            update_scene = True
                            if letter_mode:
                                k = 0.99
                                labels[i]['height'] = labels[i]['height']  * k
                                labels[i]['width'] = labels[i]['width']  * k
                            else:
                                k = 0.999
                                scale = scale * k
                                for l in labels:
                                    l['x_center'] = (l['x_center'] - 0.5) * k + 0.5
                                    l['y_center'] = (l['y_center'] - 0.5) * k + 0.5
                            need_save = True
                        if cvkey == ord('t'):
                            update_scene = True
                            letter_mode = not letter_mode
                            i = 0
                        if cvkey == ord('*') and letter_mode:
                            update_scene = True
                            i = (i + 1) % len(labels)
                        if cvkey == ord('/') and letter_mode:
                            update_scene = True
                            i = (i - 1) % len(labels)
                        if cvkey == ord('x') and not wait_letter:
                            update_scene = True
                            labels = sorted(labels, key=lambda x: x['x_center'])
                            need_save = True
                        if cvkey == ord('y') and not wait_letter:
                            update_scene = True
                            labels = sorted(labels, key=lambda x: x['y_center'])
                            need_save = True
                if ((65 <= cvkey <= 90) or cvkey == 95 or (48 <= cvkey <= 57)) and wait_letter:
                    wait_letter = False
                    update_scene = True
                    try:
                        labels.append(
                            {'class': classes.index(chr(cvkey)),
                             'x_center': (float(ix) + float(iw) / 2) / img_width,
                             'y_center': (float(iy) + float(ih) / 2) / img_height,
                             'width': float(iw) / img_width,
                             'height': float(ih) / img_height
                             })
                        labels = sorted(labels, key=lambda x: x['x_center'])
                        need_save = True
                    except:
                        pass
                if ((65 <= cvkey <= 90) or cvkey == 95 or (48 <= cvkey <= 57)) and change_letter:
                    change_letter = False
                    update_scene = True
                    try:
                        labels[i]['class'] = classes.index(chr(cvkey))
                        labels = sorted(labels, key=lambda x: x['x_center'])
                        need_save = True
                    except:
                        pass
                elif cvkey == ord(' '):
                    wait_letter = False
                    update_scene = True
                if need_save:
                    need_save = False
                    with open(lblfilepath, 'w') as f:
                        for l in labels:
                            f.write(
                                str(l['class']) + ' ' + str(l['x_center']) + ' ' + str(l['y_center']) + ' ' + str(
                                    l['width']) + ' ' + str(l['height']) + '\n')




