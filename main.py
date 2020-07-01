import cv2
import numpy as np
import process as p


def main():
  face_cascade, eye_cascade, detector = p.init_cv()

  cap = cv2.VideoCapture(0)
  frame_width = int(cap.get(3))
  frame_height = int(cap.get(4))
  fourcc = cv2.VideoWriter_fourcc(*'MJPG')

  # Specify the FourCC codec
  # output = cv2.VideoWriter('output.avi',fourcc, 30, (frame_width,frame_height))
  cv2.namedWindow('image')
  # cv2.createTrackbar('threshold', 'image', 0, 255, p.nothing)
  previous_right_blob_area = previous_right_keypoints = previous_left_blob_area = previous_left_keypoints = None



  right_eye_threshold = 25
  left_eye_threshold = 25


  while True:
      _, frame = cap.read()
      gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

      face_frame, face_frame_gray, left_eye_estimated_position, right_eye_estimated_position, _, _ = p.detect_face(
                  frame, gray_frame, face_cascade)

      if face_frame is not None:
          left_eye_frame, right_eye_frame, left_eye_frame_gray, right_eye_frame_gray = p.detect_eyes(face_frame,
                                                                                                          face_frame_gray,
                                                                                                          left_eye_estimated_position,
                                                                                                          right_eye_estimated_position,
                                                                                                          eye_cascade)

          if right_eye_frame is not None:
            right_keypoints, previous_right_keypoints, previous_right_blob_area = p.get_keypoints(
                detector, right_eye_frame, right_eye_frame_gray, right_eye_threshold,
                previous_area=previous_right_blob_area,
                previous_keypoint=previous_right_keypoints)
            # print(f"right_eye_frame: {right_eye_frame}, right_keypoints: {right_keypoints}")
            p.draw_blobs(right_eye_frame, right_keypoints)


          if left_eye_frame is not None:
            left_keypoints, previous_left_keypoints, previous_left_blob_area = p.get_keypoints(
                detector, left_eye_frame, left_eye_frame_gray, left_eye_threshold,
                previous_area=previous_left_blob_area,
                previous_keypoint=previous_left_keypoints)
            # print(f"left_eye_frame: {left_eye_frame}, left_keypoints: {left_keypoints}")
            p.draw_blobs(left_eye_frame, left_keypoints)

          cv2.imshow('frame', frame)

      if cv2.waitKey(1) & 0xFF == ord('q'):
          break
  cap.release()
  # output.release()
  cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
