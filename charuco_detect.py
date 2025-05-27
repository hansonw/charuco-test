import argparse
import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Detect ChArUco board and draw bounding polygon"
    )
    parser.add_argument("--input-image", required=True, help="Path to input image")
    parser.add_argument("--output-image", required=True, help="Path to save output image")
    args = parser.parse_args()

    # Load image
    image = cv2.imread(args.input_image)
    if image is None:
        raise SystemExit(f"Could not read image {args.input_image}")

    # Create the ChArUco board description
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    board = cv2.aruco.CharucoBoard((5, 7), 1.0, 0.7, aruco_dict)

    # Detect the board
    charuco = cv2.aruco.CharucoDetector(board)
    charucoCorners, charucoIds, markerCorners, markerIds = charuco.detectBoard(image)

    if charucoIds is None or len(charucoIds) == 0:
        print("No ChArUco board detected")
        return

    # Estimate pose using solvePnP
    objPoints, imgPoints = board.matchImagePoints(charucoCorners, charucoIds)
    cameraMatrix = np.array(
        [[1000.0, 0.0, image.shape[1] / 2.0],
         [0.0, 1000.0, image.shape[0] / 2.0],
         [0.0, 0.0, 1.0]],
        dtype=float,
    )
    distCoeffs = np.zeros((5, 1))
    ok, rvec, tvec = cv2.solvePnP(objPoints, imgPoints, cameraMatrix, distCoeffs)
    if ok:
        print("Rotation vector (rvec):", rvec.ravel())
        print("Translation vector (tvec):", tvec.ravel())
    else:
        print("Pose estimation failed")

    # Draw blue polygon around the board using an oriented bounding box
    points = charucoCorners.reshape(-1, 2)
    rect = cv2.minAreaRect(points)
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    cv2.polylines(image, [box], True, (255, 0, 0), 2)  # BGR blue

    cv2.imwrite(args.output_image, image)


if __name__ == "__main__":
    main()
