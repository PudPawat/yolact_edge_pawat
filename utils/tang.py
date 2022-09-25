import cv2
import numpy as np
import math
import os
import time

def cart2pol(x, y):
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho

def pol2cart(theta, rho):
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y

def rotate_contour(cnt, angle, cen_point=None):
    if cen_point != None:
        cx, cy = cen_point
    else:
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        else:
            cx = 0
            cy = 0
    cnt_norm = cnt - [cx, cy]

    coordinates = cnt_norm[:, 0, :]
    xs, ys = coordinates[:, 0], coordinates[:, 1]
    thetas, rhos = cart2pol(xs, ys)

    thetas = np.rad2deg(thetas)
    thetas = (thetas + angle) % 360
    thetas = np.deg2rad(thetas)

    xs, ys = pol2cart(thetas, rhos)

    cnt_norm[:, 0, 0] = xs
    cnt_norm[:, 0, 1] = ys
    # print("xs,ys",(xs,ys))
    # cv2.circle(img,(xs,ys),6,[0,0,255],-1)
    # cv2.imshow("points",img)
    # cv2.waitKey(0)
    cnt_rotated_1 = cnt_norm + [cx, cy]
    cnt_rotated = cnt_rotated_1.astype(np.int32)

    return cnt_rotated, [cx, cy], cnt_rotated_1

def find_rbbox(img, mask, angle):
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(mask, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, 2, 1)
    # cnt = contours[0]
    cnt = max(contours, key=cv2.contourArea)
    # cv2.drawContours(img,cnt,-1,(0,255,0),10)
    # -----------S1 rect, rect_rotated--------------
    cnt_rotated, center_point, _ = rotate_contour(cnt, angle, None)
    # print("asdafdaf",cnt_rotated)
    # cv2.drawContours(img,cnt_rotated,-1,(255,0,0),10)
    x, y, w, h = cv2.boundingRect(cnt_rotated)
    rect_rotated = np.array([[[x, y]], [[x + w, y]], [[x + w, y + h]], [[x, y + h]]])
    # cv2.circle(img,(x, y),5,[255,0,0],-1)
    # cv2.circle(img,(x+w, y),5,[255,0,0],-1)
    # cv2.circle(img,(x+w, y+h),5,[255,0,0],-1)
    # cv2.circle(img,[x, y+h],5,[255,0,0],-1)
    rect_out, _, _ = rotate_contour(rect_rotated, -angle, center_point)  # rotate back
    # -----------S2 origin_rotated-------------------
    length = w
    # convex_hull ==> far_points
    hull = cv2.convexHull(cnt_rotated,
                          returnPoints=False)  # returnPoints = False while finding convex hull, in order to find convexity defects.
    try:
        defects = cv2.convexityDefects(cnt_rotated, hull)
    except:
        cnt = cv2.approxPolyDP(cnt_rotated, 0.01 * cv2.arcLength(cnt, True), True)
        hull = cv2.convexHull(cnt, returnPoints=False)
        defects = cv2.convexityDefects(cnt, hull)
        # pass
    far_points = []
    for i in range(defects.shape[0]):
        s, e, f, d = defects[
            i, 0]  # [ start point, end point, farthest point, approximate distance to farthest point ].
        far = tuple(cnt_rotated[f][0])
        far_points.append(far)
        # cv2.circle(img,far,5,[0,0,255],-1)
    # print(far_points)

    # filter convex_hull in 1/3 rect_rotated (x,x+length *1/3), (y,y+h) percent =33%
    percent = 10
    far_points_filtered = []
    for point in far_points:
        if (x <= point[0] <= x + int(length * percent / 100)) and (y <= point[1] <= y + h):
            far_points_filtered.append(point)
        # elif
        #     far_points_filtered.append(np.max(point))
    y_min, y_max = 10000, 0
    for point in far_points_filtered:
        if point[1] < y_min:
            y_min = point[1]
        if point[1] > y_max:
            y_max = point[1]
    if not far_points_filtered:
        origin_rotated = np.array([[[x, center_point[1]]]])
        print("cannot defect hull")
    else:
        origin_rotated = np.array([[[x, (y_min + y_max) // 2]]])
    # print("center point", center_point)
    origin, _, diem = rotate_contour(origin_rotated, -angle, center_point)
    # print("diem",origin)
    # cv2.drawContours(img, [diem], 0, (0, 255, 255), 20)
    origin = tuple(origin.squeeze())

    # origin + angle + length ==> end_point
    x_end = int(origin[0] + length * math.cos(math.radians(angle)))
    y_end = int(origin[1] - length * math.sin(math.radians(angle)))
    end_point = (x_end, y_end)
    # Draw vector
    # -----------Other part---------
    # y1, x1, y2, x2 = roi
    # cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,0),2)
    caption = str(angle)
    x_add = int(30 * math.cos(math.radians(angle)))
    y_add = int(10 * math.sin(math.radians(angle)))
    # cv2.putText(img, caption, (x_end + x_add, y_end -y_add), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 1)
    # ------------------------------
    # cv2.drawContours(img, [rect_out], 0, (255, 0, 0), 2)
    cv2.circle(img, origin, 3, [0, 0, 255], -1)
    cv2.arrowedLine(img, origin, end_point, (255, 0, 0), 2)

    # cv2.drawContours(img, contours, 0, (255, 0, 0), 3)
    # cv2.drawContours(img, [cnt_rotated], 0, (0, 255, 0), 3)
    # cv2.drawContours(img, [rect_rotated], 0, (0, 255, 0), 2)

    # cv2.imwrite('out.png',img)
    # cv2.imshow('img',img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return origin, length

def convert_ori_len_angle2_twopoints(ori,length,angle):
    x,y = ori
    x_tip = int(x + length * math.cos(math.radians(angle)))
    y_tip = int(y - length * math.sin(math.radians(angle)))
    tip_xy = (x_tip,y_tip)
    return tip_xy

if __name__ == '__main__':
    path2mask_result = "../orchids_mask/"

    path2ori_image = "/home/ddr910/Desktop/Pawat/Pawat/yolact_edge_pawat/data/test"
    names = os.listdir(path2mask_result)
    names = sorted(names)
    ori_name = names[0].split("_")[0]
    print(ori_name)
    old_name = ori_name
    datas = []
    data = []
    for name in names:

        img = cv2.imread(os.path.join(path2mask_result,name))
        name_only = name.split(".")
        name_part = name_only[0].split("_")
        # print(name_part)
        # print(img.shape)

        if old_name == name_part[0]:
            angle = int(name_part[-1])
            ori, length = find_rbbox(img,img,angle)
            tip_coord = convert_ori_len_angle2_twopoints(ori,length,angle)
            data.append([name_part[0],ori,length,angle, tip_coord])
            old_name = name_part[0]
        else:

            datas.append(data)
            data = []
            angle = int(name_part[-1])
            ori, length = find_rbbox(img, img, angle)
            tip_coord = convert_ori_len_angle2_twopoints(ori, length, angle)
            data.append([name_part[0], ori, length, angle, tip_coord])
            old_name = name_part[0]
            # old_name = name_part[0]
        # print(length, ori)
    datas.append(data)
    print(datas)
    print(datas[-1])
    print(len(datas))
    # print(len(datas))
    for data in datas:

        print(len(data))
        img = cv2.imread(os.path.join(path2ori_image,data[0][0]+".jpg"))
        print(img.shape)
        for info in data:
            name, orinal, length, angle, tips = info
            cv2.line(img,orinal,tips,(255,0,0),3)
        print(data)

        cv2.imshow("ori",img)
        cv2.waitKey(0)