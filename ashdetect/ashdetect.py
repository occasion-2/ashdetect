import cv2
import os
import pathlib
import pandas as pd


def count_blobs(image_path):
    # Load image, grayscale, Otsu's threshold
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(image_path)
    thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1]


    # Morph open using elliptical shaped kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)


    # Find unconnected blobs
    cnts = cv2.findContours(opening, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    dots = 0
    dots_area = 0
    for c in cnts:
        area = cv2.contourArea(c)
        approx = cv2.approxPolyDP(c,0.01*cv2.arcLength(c,True),True)
        # if area > 150 and area <= 350 and len(approx) > 8:
        if area > 150 and area <= 350:
            cv2.drawContours(image,[c], -1, (0, 255, 255), 2)
            dots += 1
            dots_area += area
    avg_dot_area = dots_area/dots


    # Find connected blobs joint contours
    cnts = cv2.findContours(opening, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    

    # Calculate number of dots in joint contours
    for c in cnts:
        area = cv2.contourArea(c)
        if area > 350 and area < 24000:
            cv2.drawContours(image,[c], -1, (255, 0, 0), 2)
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255,0), 2)
            center = (x,y)
            number = area/avg_dot_area
            cv2.putText(image, str(round(number)), center, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0), 1, cv2.LINE_AA)
            dots += round(number)


    # Save visualization of the detected dots
    cv2.putText(image, str(dots), (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0), 2, cv2.LINE_AA)
    cv2.imwrite(f'{image_path[:-4]}_result.jpg', image)
    return dots

if __name__ == "__main__":
    results_df = pd.DataFrame()
    for root, dirs, files in os.walk("."):
        for name in files:
            if name.endswith(".tif"):
                dots = count_blobs(os.path.join(root, name))
                path = pathlib.PurePath(os.path.join(root, name))
                results_df[f'{path.parent.name} / {name}'] = pd.Series(dots)

    results_df.reindex(sorted(results_df.columns), axis=1).to_csv('results.csv')
