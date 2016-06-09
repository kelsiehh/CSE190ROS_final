import cv2
import numpy as np
import os
from read_config import read_config

from PIL import Image, ImageFont, ImageDraw

row = (read_config()["map_size"][0])
col = (read_config()["map_size"][1])
goal = read_config()["goal"]
pits = read_config()["pits"]
walls = read_config()["walls"]

reward_for_reaching_goal = read_config()['reward_for_reaching_goal']
reward_for_falling_in_pit = read_config()['reward_for_falling_in_pit']
reward_for_hitting_wall = read_config()['reward_for_hitting_wall']

lw = 3 #line width
mg = 5 #margin

width = col * (100 + lw) - lw
height = row * (100 + lw) - lw

goalImg = Image.open(os.path.dirname(os.path.abspath(__file__)) + "/../img/goal.jpg")
goalImg = goalImg.resize((60, 60), Image.ANTIALIAS)
wallImg = Image.open(os.path.dirname(os.path.abspath(__file__)) + "/../img/wall.jpg")
wallImg = wallImg.resize((60, 60), Image.ANTIALIAS)
pitImg = Image.open(os.path.dirname(os.path.abspath(__file__)) + "/../img/pit.jpg")
pitImg = pitImg.resize((60, 60), Image.ANTIALIAS)


def save_image_for_iteration(qvalues, curState, action, iteration):
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for x in range(0, col):
        for y in range(0, row):
            if [y, x] == goal or [y, x] in pits or [y, x] in walls:
                continue
            if [y, x] == curState:
                if action == "E":
                    draw.polygon([(x * (100 + lw) + 85, y * (100 + lw) + 50),\
                    (x * (100 + lw) + 65, y * (100 + lw) + 40), (x * (100 + lw) + 65, y * (100 + lw) + 60)], fill = (0,255,0,50))
                elif action == "W":
                    draw.polygon([(x * (100 + lw) + 15, y * (100 + lw) + 50),\
                    (x * (100 + lw) + 35, y * (100 + lw) + 40), (x * (100 + lw) + 35, y * (100 + lw) + 60)], fill = (0,255,0,50))
                elif action == "N":
                    draw.polygon([(x * (100 + lw) + 50, y * (100 + lw) + 15),\
                    (x * (100 + lw) + 40, y * (100 + lw) + 35), (x * (100 + lw) + 60, y * (100 + lw) + 35)], fill = (0,255,0,50))
                else:
                    draw.polygon([(x * (100 + lw) + 50, y * (100 + lw) + 85),\
                    (x * (100 + lw) + 40, y * (100 + lw) + 65), (x * (100 + lw) + 60, y * (100 + lw) + 65)], fill = (0,255,0,50))
            
            draw.line(( (x * (100 + lw), y * (100 + lw)), (x * (100 + lw) + 100 - 1, y * (100 + lw) + 100 - 1) ), fill=(0, 0,0))
            draw.line(( (x * (100 + lw), y * (100 + lw) + 100 - 1), (x * (100 + lw) + 100 - 1, y * (100 + lw)) ), fill=(0, 0,0))
            draw.text( (x * (100 + lw) + 40, y * (100 + lw) + mg), str("%.2f" % qvalues[y][x].N), fill="#000000")
            draw.text( (x * (100 + lw) + mg, y * (100 + lw) + 48), str("%.2f" % qvalues[y][x].W), fill="#000000")
            draw.text( (x * (100 + lw) + 50 + 2 * mg, y * (100 + lw) + 48), str("%.2f" % qvalues[y][x].E), fill="#000000")
            draw.text( (x * (100 + lw) + 40, y * (100 + lw) + 100 - 3 * mg), str("%.2f" % qvalues[y][x].S), fill="#000000")

    for x in range(1, col):
        draw.polygon([(x * (100 + lw) - lw, 0), (x * (100 + lw) - lw, row * (100 + lw) - lw),\
            (x * (100 + lw), row * (100 + lw) - lw), (x * (100 + lw), 0)], fill = (0,0,0,255))

    for y in range(1, row):
        draw.polygon([(0, y * (100 + lw) - lw), (col * (100 + lw) - lw, y * (100 + lw) - lw),\
            (col * (100 + lw) - lw, y * (100 + lw)), (0, y * (100 + lw))], fill = (0,0,0,255))

    img.paste(goalImg, (goal[1] * (100 + lw) + 20, goal[0] * (100 + lw) + 20))
    draw.text( (goal[1] * (100 + lw) + 40, goal[0] * (100 + lw) + 10), str(reward_for_reaching_goal), fill=(255,0,0))
    
    for pit in pits:
        img.paste(pitImg, (pit[1] * (100 + lw) + 20, pit[0] * (100 + lw) + 20))
        draw.text( (pit[1] * (100 + lw) + 40, pit[0] * (100 + lw) + 10), str(reward_for_falling_in_pit), fill=(255,0,0))
    for wall in walls:
        img.paste(wallImg, (wall[1] * (100 + lw) + 20, wall[0] * (100 + lw) + 20))
        draw.text( (wall[1] * (100 + lw) + 40, wall[0] * (100 + lw) + 45), str(reward_for_hitting_wall), fill=(255,0,0))

    draw.ellipse( (curState[1] * (100 + lw) + 45, curState[0] * (100 + lw) + 45,\
        curState[1] * (100 + lw) + 55, curState[0] * (100 + lw) + 55), fill=(0,0,255))
    # draw.text( (curState[1] * (100 + lw) + 47, curState[0] * (100 + lw) + 47), action, fill=(0,0,255))

    img.save(os.path.dirname(os.path.abspath(__file__)) + "/../saved_video/qvalues_iteration_" + str(iteration) + ".jpg")

def generate_video(no_of_iterations):
    video = cv2.VideoWriter(os.path.dirname(os.path.abspath(__file__)) + "/../saved_video/video_qvalues.avi", cv2.cv.CV_FOURCC('m', 'p', '4', 'v'), 1, (width, height))

    for i in range(no_of_iterations):
        file_name = os.path.dirname(os.path.abspath(__file__)) + "/../saved_video/qvalues_iteration_" + str(i) + ".jpg"
        img = cv2.imread(file_name)
        video.write(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
        #This removes the image after stitching it to the video. Please comment this if you want the images to be saved
        os.remove(file_name)
    video.release()


def save_final_image(qvalues):
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for x in range(0, col):
        for y in range(0, row):
            if [y, x] == goal or [y, x] in pits or [y, x] in walls:
                continue
            draw.line(( (x * (100 + lw), y * (100 + lw)), (x * (100 + lw) + 100 - 1, y * (100 + lw) + 100 - 1) ), fill=(0, 0,0))
            draw.line(( (x * (100 + lw), y * (100 + lw) + 100 - 1), (x * (100 + lw) + 100 - 1, y * (100 + lw)) ), fill=(0, 0,0))
            draw.text( (x * (100 + lw) + 40, y * (100 + lw) + mg), str("%.2f" % qvalues[y][x].N), fill="#000000")
            draw.text( (x * (100 + lw) + mg, y * (100 + lw) + 48), str("%.2f" % qvalues[y][x].W), fill="#000000")
            draw.text( (x * (100 + lw) + 50 + 2 * mg, y * (100 + lw) + 48), str("%.2f" % qvalues[y][x].E), fill="#000000")
            draw.text( (x * (100 + lw) + 40, y * (100 + lw) + 100 - 3 * mg), str("%.2f" % qvalues[y][x].S), fill="#000000")

    for x in range(1, col):
        draw.polygon([(x * (100 + lw) - lw, 0), (x * (100 + lw) - lw, row * (100 + lw) - lw),\
            (x * (100 + lw), row * (100 + lw) - lw), (x * (100 + lw), 0)], fill = (0,0,0,255))

    for y in range(1, row):
        draw.polygon([(0, y * (100 + lw) - lw), (col * (100 + lw) - lw, y * (100 + lw) - lw),\
            (col * (100 + lw) - lw, y * (100 + lw)), (0, y * (100 + lw))], fill = (0,0,0,255))

    img.paste(goalImg, (goal[1] * (100 + lw) + 20, goal[0] * (100 + lw) + 20))
    draw.text( (goal[1] * (100 + lw) + 40, goal[0] * (100 + lw) + 10), str(reward_for_reaching_goal), fill=(255,0,0))
    
    for pit in pits:
        img.paste(pitImg, (pit[1] * (100 + lw) + 20, pit[0] * (100 + lw) + 20))
        draw.text( (pit[1] * (100 + lw) + 40, pit[0] * (100 + lw) + 10), str(reward_for_falling_in_pit), fill=(255,0,0))
    for wall in walls:
        img.paste(wallImg, (wall[1] * (100 + lw) + 20, wall[0] * (100 + lw) + 20))
        draw.text( (wall[1] * (100 + lw) + 40, wall[0] * (100 + lw) + 45), str(reward_for_hitting_wall), fill=(255,0,0))

    img.save(os.path.dirname(os.path.abspath(__file__)) + "/../saved_video/final_qvalues" + ".jpg")

