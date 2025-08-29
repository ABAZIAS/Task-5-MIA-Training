import cv2 as cv
import numpy as np
import os

'''BLUE_LOWER = np.array([90, 60, 50], dtype=np.uint8) 
BLUE_UPPER = np.array([140, 255, 255], dtype=np.uint8) '''
BLUE_LOWER = np.array([100, 100, 50], dtype=np.uint8) # Adjusted H, S, V
BLUE_UPPER = np.array([120, 255, 255], dtype=np.uint8) # Adjusted H

RED_LOWER1 = np.array([0, 100, 50], dtype=np.uint8) 
RED_UPPER1 = np.array([10, 255, 255], dtype=np.uint8) 
RED_LOWER2 = np.array([170, 100, 50], dtype=np.uint8) 
RED_UPPER2 = np.array([180, 255, 255], dtype=np.uint8)

KERNEL = cv.getStructuringElement(cv.MORPH_ELLIPSE, (13,13))  # big enough to fill internal patterns
MIN_CIRCULARITY = 0.55
MIN_SOLIDITY = 0.7       
MIN_AREA_FRAC = 0.001  
MIN_AREA_ABS = 500  
MIN_roundess=0.6

img_dir = "balls"
out_dir = "balls_out_contours2"
os.makedirs(out_dir, exist_ok=True)

labels_dir = "labels"
os.makedirs(labels_dir, exist_ok=True)


def circularity(contour):
    area = cv.contourArea(contour)
    if area <= 0: return 0.0
    perim = cv.arcLength(contour, True)
    if perim == 0: return 0.0
    return 4.0 * np.pi * area / (perim*perim)

def make_mask(hsv, color):
    if color == "blue":
        return cv.inRange(hsv, BLUE_LOWER, BLUE_UPPER)
    if color == "red":
        m1 = cv.inRange(hsv, RED_LOWER1, RED_UPPER1)
        m2 = cv.inRange(hsv, RED_LOWER2, RED_UPPER2)
        return cv.bitwise_or(m1, m2)
    raise ValueError

def merge_detections(dets, overlap_thresh=0.6):
    # dets = list of (x,y,r,color,area)
    keep = []
    used = [False]*len(dets)
    for i,(x,y,r,c,a) in enumerate(dets):
        if used[i]: continue
        best = i
        for j,(x2,y2,r2,c2,a2) in enumerate(dets):
            if i==j or used[j]: continue
            d = np.hypot(x-x2, y-y2)
            if d < max(r,r2)*overlap_thresh:
                # keep the larger area
                if a2 > a:
                    best = j
                    break
                else:
                    used[j] = True
        keep.append(dets[best])
        used[best] = True
    return keep



# ---------- main ----------
debug_dir = os.path.join(out_dir, "debug")
os.makedirs(debug_dir, exist_ok=True)

for fname in sorted(os.listdir(img_dir)):
    if not fname.lower().endswith(('.png','.jpg','.jpeg','.bmp')): continue
    path = os.path.join(img_dir, fname)
    img = cv.imread(path)
    if img is None:
        print("Failed to load", path); continue
    h,w = img.shape[:2]
    min_area = max(MIN_AREA_ABS, int(h*w*MIN_AREA_FRAC))
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    output = img.copy()
    detections = []

    print("Processing", fname)
    for color in ("blue","red"):
        mask = make_mask(hsv, color)
        # morphology & blur (as in your original)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, KERNEL, iterations=2)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, KERNEL, iterations=1)
        mask = cv.GaussianBlur(mask, (7,7), 0)

        # IMPORTANT: threshold to binary again for findContours stability
        _, mask_bin = cv.threshold(mask, 127, 255, cv.THRESH_BINARY)

        contours, _ = cv.findContours(mask_bin, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contour_vis = img.copy()
        cv.drawContours(contour_vis, contours, -1, (0,255,0), 2)  
        cv.imwrite(os.path.join(debug_dir, f"debug_{os.path.splitext(fname)[0]}_{color}_contours.png"), contour_vis)

        for c in contours:
            area = cv.contourArea(c)
            if area < min_area:
                continue
            circ = circularity(c)

            hull = cv.convexHull(c)
            hull_area = cv.contourArea(hull) if hull is not None else 0
            solidity = (area / hull_area) if hull_area>0 else 0

            (x,y), r = cv.minEnclosingCircle(c)
            cx,cy,r = int(x), int(y), int(r)

            circle_area = np.pi * (r ** 2)
            roundness = (area / circle_area) if circle_area>0 else 0

            if circ <= MIN_CIRCULARITY or solidity <= MIN_SOLIDITY or roundness <= MIN_roundess:
                 continue

            detections.append((cx,cy,r,color,area))

    # merge overlapping detections (keep largest)
    final = merge_detections(detections)

    base = os.path.splitext(fname)[0]
    txt_path = os.path.join(labels_dir, base + ".txt")
    with open(txt_path, "w") as f:
        # final is list of tuples (cx,cy,r,color,area)
        for (cx,cy,r,color,area) in final:
            class_id = 0 if color == "blue" else 1
            # write as integers: class_id x_center y_center radius
            f.write(f"{class_id} {cx} {cy} {r}\n")


    # draw final detections
    for (cx,cy,r,color,area) in final:
        col = (255,150,0) if color=="blue" else (150,150,255)
        cv.circle(output, (cx,cy), r, col, 3)
        cv.circle(output, (cx,cy), 3, (0,0,255), -1)
        cv.putText(output, color, (max(0,cx-r), max(10,cy-r-10)), cv.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)

    outpath = os.path.join(out_dir, fname)
    cv.imwrite(outpath, output)
    print(" ->", outpath, "detections:", len(final))

print("done")
