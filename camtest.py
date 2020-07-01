import cv2
import numpy as np

# rawvid = 'proj04_vidA_960x540.mp4'
capture = cv2.VideoCapture(0)
frame_width = int(capture.get(3))
frame_height = int(capture.get(4))
# fourcc = cv2.VideoWriter_fourcc(*'MJPG')

# Specify the FourCC codec
# output = cv2.VideoWriter('transcode.avi',fourcc, 24, (frame_width,frame_height))

while(True):
  ret, frame = capture.read()
  if ret == True:
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY )
    cv2.imshow('frame',gray)

    # output.write(gray)
    # q to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  else:
    break

capture.release()
# output.release()

cv2.destroyAllWindows()
