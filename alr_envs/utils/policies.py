class PDController:
    def __init__(self, p_gains, d_gains):
        self.p_gains = p_gains
        self.d_gains = d_gains

    def get_action(self, env, des_pos, des_vel):
        # TODO: make standardized ALRenv such that all of them have current_pos/vel attributes
        cur_pos = env.current_pos
        cur_vel = env.current_vel
        if len(des_pos) != len(cur_pos):
            des_pos = env.extend_des_pos(des_pos)
        if len(des_vel) != len(cur_vel):
            des_vel = env.extend_des_vel(des_vel)
        trq = self.p_gains * (des_pos - cur_pos) + self.d_gains * (des_vel - cur_vel)
        return trq
