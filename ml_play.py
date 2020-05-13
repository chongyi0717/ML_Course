# -*- coding: utf-8 -*-
"""
Created on Mon May 11 23:05:54 2020

@author: chongyi
"""

from mlgame.communication import ml as comm
from os import path
import pickle    
import numpy as np

def ml_loop(side: str):
    file="rfc_"+side+".sav"
    filename = path.join(path.dirname(__file__), 'save', file)
    with open(filename, 'rb') as file:
        clf = pickle.load(file)
    ball_served=False
           

    # 2. Inform the game process that ml process is ready
    comm.ml_ready()

    # 3. Start an endless loop
    while True:
        # 3.1. Receive the scene information sent from the game process
        scene_info = comm.recv_from_game()

        # 3.2. If either of two sides wins the game, do the updating or
        #      resetting stuff and inform the game process when the ml process
        #      is ready.
        if scene_info["status"] != "GAME_ALIVE":
            # Do some updating or resetting stuff
            ball_served = False

            # 3.2.1 Inform the game process that
            #       the ml process is ready for the next round
            comm.ml_ready()
            continue

        # 3.3 Put the code here to handle the scene information

        # 3.4 Send the instruction for this frame to the game process
        if not ball_served:
            comm.send_to_game({"frame": scene_info["frame"], "command": "SERVE_TO_LEFT"})
            ball_served = True
        else:
            if side == "1P": 
                feature=[]
                feature.append(scene_info["platform_1P"][0])
                feature.append(scene_info["ball"][0])
                feature.append(scene_info["ball"][1])
                feature.append(scene_info["ball_speed"][0])
                feature.append(scene_info["ball_speed"][1])
                feature.append(scene_info["blocker"][0])
                feature.append(scene_info["blocker"][1])
                feature.append(scene_info["frame"])
                feature=np.array(feature)
                feature=feature.reshape(-1,8)
                pred=clf.predict(feature)                
                #command = ml_loop_for_1P()
                command=pred    
            else:
                feature=[]
                feature.append(scene_info["platform_2P"][0])
                feature.append(scene_info["ball"][0])
                feature.append(scene_info["ball"][1])
                feature.append(scene_info["ball_speed"][0])
                feature.append(scene_info["ball_speed"][1])
                feature.append(scene_info["blocker"][0])
                feature.append(scene_info["blocker"][1])
                feature.append(scene_info["frame"])
                feature=np.array(feature)
                feature=feature.reshape(-1,8)
                pred=clf.predict(feature)
                command=pred
            if command == 0:
                comm.send_to_game({"frame": scene_info["frame"], "command": "NONE"})
            elif command == 1:
                comm.send_to_game({"frame": scene_info["frame"], "command": "MOVE_RIGHT"})
            elif command==2 :
                comm.send_to_game({"frame": scene_info["frame"], "command": "MOVE_LEFT"})
        
