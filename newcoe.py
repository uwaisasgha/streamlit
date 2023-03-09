import cv2
import numpy as np
import streamlit as st
from lib.func import order_vertices_ccw, point_in_triangle

scores_dict={
    "0":"19",
    "1":"7",
    "2":"16",
    "3":"8",
    "4":"11",
    "5":"14",
    "6":"9",
    "7":"12",
    "8":"5",
    "9":"20",
    "10":"1",
    "11":"18",
    "12":"4",
    "13":"13",
    "14":"6",
    "15":"10",
    "16":"15",
    "17":"2",
    "18":"17",
    "19":"3"
}

lower_black = np.array([0, 0, 0])
upper_black = np.array([180, 255, 50])

uploaded_file = st.file_uploader("Select an image", type=['jpeg','png'])

if uploaded_file is not None:
    # Read the image data into a bytes object
    image_bytes = uploaded_file.read()

    # Convert the image bytes to a numpy array
    image_np = np.frombuffer(image_bytes, np.uint8)

    # Decode the numpy array to an image using OpenCV
    img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)



    # Get the dimensions of the image
    height, width, channels = img.shape

    # Determine the longer dimension of the image
    longer_dim = max(height, width)

    # Calculate the scale factor to resize the longer dimension to 240 pixels
    scale_factor = 640.0 / longer_dim

    # Resize the image using the scale factor
    img = cv2.resize(img, (int(width * scale_factor), int(height * scale_factor)))

    # Creating a copy of original image
    image = img.copy()

    # Create a white image
    white_img = np.ones_like(img) * 255

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to remove noise
    blur = cv2.GaussianBlur(gray, (15, 15), 0)

    # Detect the circles using Hough circle transform
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1, minDist=500, param1=50, param2=30, minRadius=170, maxRadius=180)

    # list of 20 triangles to find the location of the "point" 
    triangles_list = []
    points = []

    # Draw the detected circle on a copy of the original image
    circle_img = img.copy()
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            # cv2.circle(circle_img, (x, y), r, (255, 255, 150), 5)
            
            # making a filled circle on the white image
            cv2.circle(white_img, (x, y), r, (0,0,0), -1)

            # Create a mask that is white inside the circle and black outside the circle
            mask = cv2.bitwise_not(white_img)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)[1]
            result = cv2.bitwise_and(circle_img, circle_img, mask=mask)
            
            # CONVERTING to gray
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            gray = cv2.blur(gray, (5,5))
            # Define lower and upper bounds for black color in HSV
            lower = np.array([0])
            upper = np.array([100])
            
            # Create a mask 
            black_portion = cv2.inRange(gray, lower, upper) # pass only black color

            # Apply the mask to the original image
            result = cv2.bitwise_and(result, result, mask=black_portion)

            cpy_gray = gray.copy()
            cpy_gray= cv2.blur(cpy_gray, (5,5))
            _, target = cv2.threshold(cpy_gray, 215, 255, cv2.THRESH_BINARY)

            # Find contours of white regions
            contours, _ = cv2.findContours(target, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Draw bounding boxes around white regions
            for contour in contours:
                x_c, y_c, w_c, h_c = cv2.boundingRect(contour)
                # cv2.rectangle(img, (x_c, y_c), (x_c + w_c, y_c + h_c), (255, 0, 0), 2)
                cv2.circle(result, (x_c + w_c // 2, y_c + h_c // 2),7,(255, 255, 255),-1)
                cv2.circle(result, (x_c + w_c // 2, y_c + h_c // 2),4,(255, 0, 0),-1)
                cv2.circle(result, (x_c + w_c // 2, y_c + h_c // 2),1,(255, 255, 0),-1)
                points.append([(x_c + w_c // 2, y_c + h_c // 2), (x_c, y_c)])  # center point of rectangle, start point of rectangle

            x_prev = 0
            y_prev = 0
            
            # # Draw 20 lines from the center to the edge of the circle
            for i in range(20):
                angle = i * 18 + 9 # calculate the angle between each line in degrees
                x1 = int(x + r * np.cos(np.deg2rad(angle)))
                y1 = int(y + r * np.sin(np.deg2rad(angle)))
                cv2.line(result, (x, y), (x1+7, y1), (0, 255, 0), 2)
                cv2.putText(result, str(i), (x1-10, y1+20), 2, 0.5, (255, 255, 255)) # to get the triangle number
                cv2.putText(result,str(x1)+" ,"+str(y1),(x1,y1),2,0.5,(255, 0, 0)) # to show each vertic of the triangle
                if i == 0 :
                    x_0 = x1
                    y_0 = y1
                elif i == 19:
                    triangles_list.append(order_vertices_ccw((x, y), (x1, y1), (x_prev, y_prev)))
                    triangles_list.append(order_vertices_ccw((x, y), (x1, y1), (x_0, y_0)))
                else:
                    triangles_list.append(order_vertices_ccw((x, y), (x1, y1), (x_prev, y_prev)))
                x_prev = x1
                y_prev= y1    
        
        # writting scores for each dart
        for point in points:
            dart, st_point = point[0], point[1]
            for index, i in enumerate(triangles_list):
                if point_in_triangle(dart, i):
                    cv2.putText(img, "{}".format(scores_dict[str(index)]), (st_point[0]-5, st_point[1]-5),cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 2)

        st.write("img")
        st.image(img)
        # st.write("Processing")
        # st.image(result)
    else:
        st.write("Dart Board Not Detected.")



# cv2.imwrite(f"results/{aa.split('_')[1].split('.')[0]}_ORG.jpeg",image)
# cv2.imwrite(f"results/{aa.split('_')[1].split('.')[0]}_MOD.jpeg",img)
# cv2.imwrite(f"{image_path.split('.')[0]}_MOD.jpeg", img)
# Display the images
# cv2.imshow('Original Image', img)
# cv2.imshow('Detected Circle', circle_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
