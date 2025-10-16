#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2025-09-21
#     Author: Martin Cífka <martin.cifka@cvut.cz>
#
from stringprep import c22_specials
from typing import List
from numpy.typing import ArrayLike
import numpy as np
import cv2  # noqa


def find_hoop_homography(images: ArrayLike, hoop_positions: List[dict]) -> np.ndarray:
    all_circles = []
    
    for src_image in images:
        gray_scale = cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray_scale, 7)

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            1,
            200,
            param1=100,
            param2=100,
            minRadius=0,
            maxRadius=5000
        )

        if circles is not None:
            all_circles.append(np.array([circles[0][0][0], circles[0][0][1], 1]))

    H, _ = cv2.findHomography(np.asarray(all_circles), np.asarray([[hoop["translation_vector"][0], hoop["translation_vector"][1], 1] for hoop in hoop_positions]), cv2.LMEDS)
    return H
