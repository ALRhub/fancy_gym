## GENERAL
    
    - This is a modification (5 links) of Mujoco Gym's Reacher (2 links)

    - Creating a custom Mujoco Gym according to this guides: https://github.com/openai/gym/blob/master/docs/creating-environments.md

## INSTALL
    - NOTE: you should look into envs/reacher_env.py and change the link to the correct path of .xml file on your computer.  
    
    - Install: go to "../reacher_5_links"        
        ``` pip install -e reacher_5_links ```
    - Use (see example.py): 
        ``` env = gym.make('reacher:ALRReacherEnv-v0')```