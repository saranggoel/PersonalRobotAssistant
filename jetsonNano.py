import RPi.GPIO as GPIO
import time
import requests

# Pin Definitions
motor_pin_i1 = 16  # Green
motor_pin_i2 = 18  # Yellow
motor_pin_i3 = 22  # Orange
motor_pin_i4 = 24  # Red
#MOTOR 1: GOING UP is LOW/HIGH (2 sec) GOING DOWN is HIGH/LOW (2 sec)
#MOTOR 2: OPEN is LOW/HIGH (1 sec) CLOSE is HIGH/LOW (1 sec)

# Pin Setup:
# Board pin-numbering scheme
GPIO.setmode(GPIO.BOARD)
# Set both pins LOW to keep them stationary
# You can keep one of them HIGH and the LOW to start with rotation in one direction
GPIO.setup(motor_pin_i1, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(motor_pin_i2, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(motor_pin_i3, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(motor_pin_i4, GPIO.OUT, initial=GPIO.LOW)

x = True
while x:
        pos = {'device':0}
        response = requests.post('http://saranggoel.pythonanywhere.com/', json=pos)
        if response.text == "0": #GOING UP MOTOR 1
            GPIO.output(motor_pin_i1, GPIO.LOW)
            GPIO.output(motor_pin_i2, GPIO.HIGH)
            time.sleep(2)
        elif response.text == "1":  # GOING DOWN MOTOR 1
            GPIO.output(motor_pin_i1, GPIO.HIGH)
            GPIO.output(motor_pin_i2, GPIO.LOW)
            time.sleep(2)
        elif response.text == "2":  # OPENING MOTOR 2
            GPIO.output(motor_pin_i3, GPIO.LOW)
            GPIO.output(motor_pin_i4, GPIO.HIGH)
            time.sleep(1)
        elif response.text == "3":
            GPIO.output(motor_pin_i3, GPIO.HIGH)
            GPIO.output(motor_pin_i4, GPIO.LOW)
            time.sleep(1)
        GPIO.output(motor_pin_i1, GPIO.LOW)
        GPIO.output(motor_pin_i2, GPIO.LOW)
        GPIO.output(motor_pin_i3, GPIO.LOW)
        GPIO.output(motor_pin_i4, GPIO.LOW)
        if response.text == "4":
            GPIO.cleanup()
            x=False
        time.sleep(0.1)


