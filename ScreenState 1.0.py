import cv2
import numpy as np
import pyautogui
import pygetwindow as gw
import sys
import keyboard
import time
import threading

start_point = [270, 70]
end_point = [335, 135]
width = 64
stop_threads = False
board = [[0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0]]

def Bgr_to_colour_name(bgr_colour):
    '''
    Converts a BGR colour to a human-readable colour name using predefined colour mappings.

    Args:
        bgr_colour (tuple): BGR colour tuple in the format (B, G, R), where B, G, and R are integers in the range 0-255.

    Returns:
        str: The corresponding colour name based on predefined mappings, or 'Unknown' if no match is found.
    '''
    # Define colour name mappings for various BGR colours
    colour_names = {
        (255, 255, 255): 'White',   # W
        (0, 0, 255): 'Red',         # Primary
        (0, 255, 0): 'Green',       # Primary
        (255, 0, 0): 'Blue',        # Primary
        (0, 255, 255): 'Yellow',    # Secondary
        (0, 128, 255): 'Orange',    # Tertiary
        (128, 0, 255): 'Purple',    # Tertiary
    }
    
    colours = np.array(list(colour_names.keys()))  # Array of BGR colours from the colour_names dictionary
    colour_names_array = np.array(list(colour_names.values()))  # Array of colour names corresponding to the BGR colours

    distances = np.linalg.norm(colours - np.array(bgr_colour), axis=1)  # Calculate the Euclidean distances between the 
                                                                        # input BGR colour and all colours in the dictionary

    nearest_colour_index = np.argmin(distances)  # Index of the nearest colour in the distances array

    # Check if the nearest colour is within a threshold distance
    if distances[nearest_colour_index] < 250:  
        return colour_names_array[nearest_colour_index]  # Return the name of the nearest colour
    else:
        return 'Unknown'  # Return 'Unknown' if no suitable colour is found within the threshold

def Average_colour(frame,start_point,end_point):
    roi = frame[start_point[1]+20:end_point[1]-20, start_point[0]+20:end_point[0]-20]
    
    # Calculate the average color of the ROI
    avg_color_per_row = np.average(roi, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    
    # Convert the BGR average color to a color name
    return Bgr_to_colour_name(avg_color)

def Get_grid(frame):
    start_point = [270, 70]
    end_point = [330, 130]
    font = cv2.FONT_HERSHEY_SIMPLEX#testing
    org = (10, 500)#testing
    fontScale = .25#testing
    color = (0, 0, 0)#testing
    thickness = 1#testing
    for i in range(len(board)):
        for j in range(len(board)):
        
            frame = cv2.rectangle(frame, start_point, end_point, color, thickness)#testing
            board[i][j] = Average_colour(frame,start_point,end_point)
            
            frame = cv2.putText(frame, board[i][j], (start_point[0]+15,start_point[1]+30), font, fontScale, color, thickness, cv2.LINE_AA) #testing
            
            start_point[0]+=width
            end_point[0]+=width
            
            
        start_point[0] = 270
        end_point[0] = 330
        start_point[1]+=width
        end_point[1]+=width

def Match():
    global stop_threads
    paused = False
    doubles = []
    triples = []
    quads = []
    while not stop_threads:
        if paused:
            if keyboard.is_pressed('esc'):
                paused = False
                print("Unpaused")
                time.sleep(0.5)
        else:
            
            for i in range(len(board)):
                horizontal = []
                vertical = []
                for j in range(len(board)):  
                    horizontal.append([board[i][j],[i,j]])
                    vertical.append([board[j][i],[j,i]])
                    if len(horizontal)>2:
                        if horizontal[0][0] == horizontal[1][0]:
                            tempH = [horizontal[0][0], horizontal[0][1], horizontal[1][1]]
                            if tempH not in doubles:
                                doubles.append(tempH)
                        elif horizontal[0][0] == horizontal[2][0]:
                            tempH = [horizontal[0][0], horizontal[0][1], horizontal[2][1]]
                            if tempH not in doubles:
                                doubles.append(tempH)
                        horizontal.pop(0)
                        
                    if len(vertical)>2:
                        if vertical[0][0] == vertical[1][0]:
                            tempV = [vertical[0][0], vertical[0][1], vertical[1][1]]
                            if tempV not in doubles:
                                doubles.append(tempV)
                        elif vertical[0][0] == vertical[2][0]:
                            tempV = [vertical[0][0], vertical[0][1], vertical[2][1]]
                            if tempV not in doubles:
                                doubles.append(tempV)
                        vertical.pop(0)
                    #if LOGIC
                        #triples.append([COLOUR, COORDS_1, COORDS_2, COORDS_3])
                    #if LOGIC
                        #quads.append([COLOUR, COORDS_1, COORDS_2, COORDS_3, COORDS_4])
            Check(doubles, triples, quads)
            doubles = []

            if keyboard.is_pressed('esc'):
                print(doubles)
                paused = True
                print("Paused")
                time.sleep(0.5)
    print(f"Matching thread is stopping")


def check_and_append(Ofocus, focus, direction, gem, board, Selection):
    """Helper function to check if a move is valid and append it to Selection."""
    if 0 <= focus[0] < len(board) and 0 <= focus[1] < len(board[0]):
        #print(Ofocus, focus, direction, gem, Selection)
        if board[focus[0]][focus[1]] == gem:
            Selection.append([Ofocus, direction])

def Check(doubles, triples, quads):
    Selection = []
    for gems in doubles:
        gem_color = gems[0]
        x1, y1 = gems[1]
        x2, y2 = gems[2]

        if x1 == x2:  # Horizontal check
            if (y1 + y2) % 2 == 0:  # Between
                focus = [x1, y1+1]
                check_and_append(focus,[focus[0]-1 , focus[1]], 'up', gem_color, board, Selection)
                check_and_append(focus,[focus[0]+1 , focus[1]], 'down', gem_color, board, Selection)
            else:  # l/r/u/d
                focusL = [x1, y1-1]
                check_and_append(focusL,[focusL[0]- 1 , focusL[1]], 'up', gem_color, board, Selection)
                check_and_append(focusL,[focusL[0]+ 1 , focusL[1]], 'down', gem_color, board, Selection)
                check_and_append(focusL,[focusL[0], focusL[1]- 1], 'left', gem_color, board, Selection)
                
                focusR = [x1, y2+1]
                check_and_append(focusR,[focusR[0]- 1, focusR[1]], 'up', gem_color, board, Selection)
                check_and_append(focusR,[focusR[0]+ 1 , focusR[1]], 'down', gem_color, board, Selection)
                check_and_append(focusR,[focusR[0], focusR[1]+ 1], 'right', gem_color, board, Selection)
        
        else:  # Vertical check
            if (x1 + x2) % 2 == 0:  # Between
                focus = [x1 + 1, y1]
                check_and_append(focus,[focus[0], focus[1]- 1], 'left', gem_color, board, Selection)
                check_and_append(focus,[focus[0], focus[1]+ 1], 'right', gem_color, board, Selection)
            else:  # l/r/u/d
                focusU = [x1- 1, y1]
                check_and_append(focusU,[focusU[0], focusU[1]-1], 'left', gem_color, board, Selection)
                check_and_append(focusU,[focusU[0], focusU[1]+1], 'right', gem_color, board, Selection)
                check_and_append(focusU,[focusU[0]-1, focusU[1]], 'up', gem_color, board, Selection)
                
                focusD = [x2+1, y1]
                check_and_append(focusD,[focusD[0], focusD[1]-1], 'left', gem_color, board, Selection)
                check_and_append(focusD,[focusD[0], focusD[1]+1], 'right', gem_color, board, Selection)
                check_and_append(focusD,[focusD[0]+1, focusD[1]], 'down', gem_color, board, Selection)
    
    if not Selection:
        time.sleep(1)
        print("no matches")
    else:
        Swap(Selection[0])
        Selection = []
    

def Swap(Selection):
    Reset()
    print(Selection)
    for i in range(Selection[0][1]):
        keyboard.send('right')
    for i in range(Selection[0][0]):
        keyboard.send('down')
    if Selection[1] == 'up':
        keyboard.send('w')
    elif Selection[1] == 'down':
        keyboard.send('s')
    elif Selection[1] == 'left':
        keyboard.send('a')
    elif Selection[1] == 'right':
        keyboard.send('d')
    time.sleep(1)

def Reset():
    for i in range(8):
        keyboard.send('left')
        keyboard.send('up')
    
def Code_Start():
    keyboard.wait('q')
    time.sleep(1)
    Reset()
    Match()
    
def Record():
    global stop_threads
    
    # search for the window, getting the first matched window with the title
    w = gw.getWindowsWithTitle("Bejeweled 3")[0]
    # activate the window
    w.activate()


    # Define video codec and frames per second
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    fps = 15
    record_seconds = 60

    # Create the video writer object
    out = cv2.VideoWriter("video_out.mp4v", fourcc, fps, tuple(w.size))
    while not stop_threads:
        # Capture a screenshot
        img = pyautogui.screenshot(region=(w.left, w.top, w.width, w.height))

        # Convert to a numpy array
        frame = np.array(img)

        # Convert colors from BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Draw the rectangle on the image
        Get_grid(frame)
        
        # Write the frame to the video file
        out.write(frame)
    cv2.destroyAllWindows()
    out.release()
    print(f"Recording thread is stopping")

recording_thread = threading.Thread(target=Record, args=())
matching_thread = threading.Thread(target=Code_Start, args=())
recording_thread.start()
matching_thread.start()

try:
    while True:
        time.sleep(0.1)        
except KeyboardInterrupt:
    stop_threads = True
    recording_thread.join()
    matching_thread.join()
    print("Program terminated")
    



