from fancy_gym.envs.mujoco.box_pushing.box_pushing_env import BoxPushingEnvBase


class BoxPushingRandom2RandomEnvBase(BoxPushingEnvBase):
    def __init__(self, frame_skip: int = 10, xml_name="box_pushing.xml"):
        super().__init__(frame_skip=frame_skip, xml_name=xml_name)
