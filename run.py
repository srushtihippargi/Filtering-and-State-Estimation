from system.RobotSystem import *
from world.world2d import *

# sudo spctl --master-disable

def main():
    
    world = world2d()
    
    robot_system = RobotSystem(world)

    robot_system.run_filter()

if __name__ == '__main__':
    main()