
import numpy as np
import time
import sys
import tkinter as tk


UNIT = 50
MAZE_H = 4
MAZE_W = 4

class Maze(tk.Tk,object):
    def __init__(self):
        super(Maze,self).__init__()
        self.action_space = ['u','d','l','r']
        self.n_actions = len(self.action_space)
        self.n_features = 2
        self.title('maze')
        self.geometry("{}x{}".format(MAZE_H * UNIT,MAZE_W * UNIT))
        self._build_maze()


    def _build_maze(self):
        self.canvas = tk.Canvas(self,bg='white',
                                height = MAZE_H * UNIT,
                                width = MAZE_W * UNIT)


        # create grids
        for c in range(0,MAZE_W*UNIT,UNIT):
            x0,y0,x1,y1 = c,0,c,MAZE_H * UNIT
            self.canvas.create_line(x0,y0,x1,y1)

        for r in range(0,MAZE_H*UNIT,UNIT):
            x0,y0,x1,y1 = 0,r,MAZE_W * UNIT,r
            self.canvas.create_line(x0,y0,x1,y1)

        origin = np.array([25,25])
        hell1_center = origin + np.array([UNIT * 2,UNIT])

        self.hell1 = self.canvas.create_rectangle(
            hell1_center[0] - 20,hell1_center[1] - 20,
            hell1_center[0] + 20,hell1_center[1] + 20,
            fill = 'black'
        )

        oval_center = origin + UNIT * 2

        self.oval = self.canvas.create_oval(
            oval_center[0] - 20,oval_center[1] - 20,
            oval_center[0] + 20,oval_center[1] + 20,
            fill = 'red'
        )

        self.rect = self.canvas.create_rectangle(
            origin[0] - 20,origin[1] - 20,
            origin[0] + 20,origin[1] + 20
        )

        self.canvas.pack()


    def reset(self):
        self.update()
        time.sleep(0.1)
        self.canvas.delete(self.rect)

        origin = np.array([25,25])
        self.rect = self.canvas.create_rectangle(
            origin[0] - 20,origin[1] - 20,
            origin[0] + 20,origin[1] + 20,
            fill = 'yellow'
        )
        return (np.array(self.canvas.coords(self.rect)[:2]) - np.array(self.canvas.coords(self.oval)[:2])) / (MAZE_H * UNIT)



    def step(self,action):
        s = self.canvas.coords(self.rect)
        base_action = np.array([0,0])

        if action == 0:
            if s[1] > UNIT:
                base_action[1] -= UNIT

        elif action == 1:
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT

        elif action == 2:
            if s[0] < (MAZE_W - 1 ) *UNIT:
                base_action[0] += UNIT

        elif action == 3:
            if s[0] > UNIT:
                base_action[0] -= UNIT

        self.canvas.move(self.rect,base_action[0],base_action[1])

        next_coords = self.canvas.coords(self.rect)

        if next_coords == self.canvas.coords(self.oval):
            reward = 1
            done = True

        elif next_coords in [self.canvas.coords(self.hell1)]:
            reward = -1
            done = True
        else:
            reward = 0
            done = False

        s_ = (np.array(next_coords[:2]) - np.array(self.canvas.coords(self.oval)[:2])) / (MAZE_H * UNIT)
        return s_, reward, done

    def render(self):
        # time.sleep(0.01)
        self.update()

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

np.random.seed(1)
tf.set_random_seed(1)

class SumTree(object):

    def __init__(self,capacity):
        self.capacity = capacity
        self.data_pointer = 0
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity,dtype=object)

    def add(self,p,data):
        #tree_idx和data_pointer一直存在如下关系
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        #根据优先级来更新tree_idx
        self.update(tree_idx,p)

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0


    def update(self,tree_idx,p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p

        while tree_idx!=0:
            tree_idx = (tree_idx - 1) // 2#这里刚好是他的父节点
            self.tree[tree_idx] += change


    #这个可以画个细致的图来讲解
    def get_leaf(self,v):
        parent_idx = 0
        while True:
            cl_idx = 2 * parent_idx + 1
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx,self.tree[leaf_idx],self.data[data_idx]


    @property
    def total_p(self):
        return self.tree[0]  # the root


class Memory(object):


    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.epsilon = 0.01  # small amount to avoid zero priority
        self.alpha = 0.6  # [0~1] convert the importance of TD error to priority
        self.beta = 0.4  # importance-sampling, from initial value increasing to 1
        self.beta_increment_per_sampling = 0.001
        self.abs_err_upper = 1.  # clipped abs error

    #新加入的数据，我们认为他的优先级是最大的
    def store(self, transition):
        #max_p 是序列的最值
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        #print("max_p is {0}\t self.tree.tree[-self.tree.capacity:] is {1}".format(max_p, self.tree.tree[-self.tree.capacity:]))
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)   # set the max p for new p

    def sample(self,n):
        print("self.tree.data[0].size is {0}".format(self.tree.data[0].size))
        b_idx,b_memory,ISWeights = np.empty((n,),dtype=np.int32),np.empty((n,self.tree.data[0].size)),np.empty((n,1))

        pri_seg = self.tree.total_p / n
        print("pri_seg is {0}".format(pri_seg))

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p  # for later calculate ISweight

        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            #这里的v为什么要这样计算？，在划分的这些段里面随机取一个值
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            b_idx[i], b_memory[i, :] = idx, data
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            #第一个值是tree_idx，第二个值是权重
            self.tree.update(ti, p)


class DQNPrioritizedReplay():
    def __init__(self,
            n_actions,
            n_features,
            learning_rate=0.005,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=500,
            memory_size=10000,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
            prioritized=True,
            sess=None,
                 ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        self.prioritized = prioritized  # decide to use double q or not

        self.learn_step_counter = 0

        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]


        if self.prioritized:
            self.memory = Memory(capacity=memory_size)
        else:
            self.memory = np.zeros((self.memory_size, n_features*2+2))

        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = sess

        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.cost_his = []



    def _build_net(self):
        def build_layers(s, c_names, n_l1, w_initializer, b_initializer, trainable):
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names, trainable=trainable)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names,  trainable=trainable)
                l1 = tf.nn.relu(tf.matmul(s, w1) + b1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names,  trainable=trainable)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names,  trainable=trainable)
                out = tf.matmul(l1, w2) + b2
            return out


        #---------------------input----------------------
        self.s = tf.placeholder(tf.float32,[None,self.n_features],name='s')
        self.q_target = tf.placeholder(tf.float32,[None,self.n_actions],name='Q_target')
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')

        if self.prioritized:
            self.ISWeights = tf.placeholder(tf.float32,[None,1],name='IS_weights')


        # ---------------------eval net -----------------
        with tf.variable_scope('eval_net'):
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 20, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            self.q_eval = build_layers(self.s, c_names, n_l1, w_initializer, b_initializer, True)

        # --------------------target net----------------
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            self.q_next = build_layers(self.s_, c_names, n_l1, w_initializer, b_initializer, False)


        # --------------------loss and train -----------
        with tf.variable_scope('loss'):
            if self.prioritized:
                self.abs_errors = tf.reduce_sum(tf.abs(self.q_target - self.q_eval), axis=1)    # for updating Sumtree
                self.loss = tf.reduce_mean(self.ISWeights * tf.squared_difference(self.q_target, self.q_eval))
            else:
                self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)




    def choose_action(self, observation):
        observation = observation[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action


    def store(self,s,a,r,s_):
        if self.prioritized:
            transition = np.hstack((s, [a, r], s_))
            #print("transition is {0}".format(transition))
            self.memory.store(transition)

            """
        else:  # random replay
            if not hasattr(self, 'memory_counter'):
                self.memory_counter = 0
            transition = np.hstack((s, [a, r], s_))
            index = self.memory_counter % self.memory_size
            self.memory[index, :] = transition
            self.memory_counter += 1
            """

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        if self.prioritized:
            tree_idx, batch_memory, ISWeights = self.memory.sample(self.batch_size)
            print("tree_idx is {0} batch_memory is {1} ISWeights is {2}".format(tree_idx, batch_memory, ISWeights))
        """
        else:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
            batch_memory = self.memory[sample_index, :]
        """

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={self.s_: batch_memory[:, -self.n_features:],
                       self.s: batch_memory[:, :self.n_features]})

        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)


        if self.prioritized:
            _, abs_errors, self.cost = self.sess.run([self._train_op, self.abs_errors, self.loss],
                                         feed_dict={self.s: batch_memory[:, :self.n_features],
                                                    self.q_target: q_target,
                                                    self.ISWeights: ISWeights})
            self.memory.batch_update(tree_idx, abs_errors)     # update priority
        else:
            _, self.cost = self.sess.run([self._train_op, self.loss],
                                         feed_dict={self.s: batch_memory[:, :self.n_features],
                                                    self.q_target: q_target})

        self.cost_his.append(self.cost)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()
        


def run_maze():
    step = 0
    for episode in range(30):
        observation = env.reset()
        print("observation is {0} episode is {1}".format(observation, episode))

        while True:
            env.render()
            action = RL.choose_action(observation)
            observation_,reward,done = env.step(action)
            RL.store(observation, action, reward, observation_)
            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
            step += 1

    print('game over')
    env.destroy()

if __name__ == '__main__':
    tf.reset_default_graph()
    env = Maze()
    RL = DQNPrioritizedReplay(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      )

    env.after(100, run_maze)
    env.mainloop()
    RL.plot_cost()
