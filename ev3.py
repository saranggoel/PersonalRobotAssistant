#!/usr/bin/env python3
import time
import requests
from ev3dev2.motor import OUTPUT_A, OUTPUT_B, MediumMotor, OUTPUT_C, SpeedPercent, MoveTank
tank_drive = MoveTank(OUTPUT_A, OUTPUT_B)
rotate = MediumMotor(OUTPUT_C)

x = True
while x:
        pos = {'device':1}
        response = requests.post('http://saranggoel.pythonanywhere.com/', json=pos)
        if response.text == "0": #Turning Left
            tank_drive.on_for_seconds(SpeedPercent(10), SpeedPercent(5), 3)
        elif response.text == "1":  #Turning Right
            tank_drive.on_for_seconds(SpeedPercent(5), SpeedPercent(10), 3)
        elif response.text == "2":  #Forward
            tank_drive.on_for_seconds(SpeedPercent(10), SpeedPercent(10), 3)
        elif response.text == "3": #Backward
            tank_drive.on_for_seconds(SpeedPercent(-10), SpeedPercent(-10), 3)
        elif response.text == "4": #Rotate Left
            rotate.on_for_degrees(SpeedPercent(5), 20)
        elif response.text == "5": #Rotate Right
            rotate.on_for_degrees(SpeedPercent(-5), 20)
        elif response.text == "6":
            x = False
        time.sleep(0.1)


