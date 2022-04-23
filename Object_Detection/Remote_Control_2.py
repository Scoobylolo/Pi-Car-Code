from pynput import keyboard
import Actuation as act
import tty, sys, termios

filedescriptors = termios.tcgetattr(sys.stdin)
tty.setcbreak(sys.stdin)
x = 0

while 1:
    x=sys.stdin.read(1)[0]
    print("You pressed", x)
    
    if x == "w":
        act.drive()
    elif x == "s":
        act.reverse()
    elif x == "d":
        act.right()
    elif x == "a":
        act.left()
    elif x == "q":
        act.stop()
        break
    else:
        act.stop()

  
 
        
termios.tcsetattr(sys.stdin, termios.TCSADRAIN,filedescriptors)
