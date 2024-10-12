# poses

## classification
Save 2 poses using 'R' and 'S' keys. A pose takes 2 seconds to save so that you can click the keyboard and then go to your pose.

This app keeps track of how many times you switch from the S position to the R position. For example, if you are counting push ups you can set the R as the relaxed position and S as the contracted position.

You can click the 'X' key to reset the counter. You can also change the 'R' and 'S' keys as many times as you'd like.

The counter goes up when going from the 'S' position to the 'R' position.

## feedback
Save a pose using the 'S' key, and toggle feedback using the 'F' key. You should be able to see in the console if there is anything you are doing incorrectly.

You can also click the 'G' key to see a guide (reference), including the pose and image. 

## data collection
Save a pose using the 'S' key, and download it using the 'D' key. If you don't want to repeatedly press 'S' and 'D' back and forth, you can click 'L' toggle loop mode, then 'S' to start saving a pose every 2 seconds. It will save the image and coordinates of your relevant body parts in a JSON file.

You can also run to save in a directory for a specific argument. Instead of running "python .\pose_detection.py", use "python .\pose_dection.py DIR_NAME" to save the images to poses/DIR_NAME. 