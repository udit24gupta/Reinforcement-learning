import numpy as np
#R matrix
R = np.matrix([[-1, -1, -1, -1, 0, -1],
               [-1, -1, -1 ,0, -1, 100],
               [-1, -1, -1, 0, -1, -1],
               [-1, 0, 0, -1, 0, -1],
               [-1, 0, 0, -1, -1, 100],
               [-1, 0, -1, -1, 0, 100]])
R
matrix([[ -1,  -1,  -1,  -1,   0,  -1],
        [ -1,  -1,  -1,   0,  -1, 100],
        [ -1,  -1,  -1,   0,  -1,  -1],
        [ -1,   0,   0,  -1,   0,  -1],
        [ -1,   0,   0,  -1,  -1, 100],
        [ -1,   0,  -1,  -1,   0, 100]])
#Q Matrix
Q = np.matrix(np.zeros([6,6]))
Q
matrix([[0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.]])
gamma = 0.8
initial_state = 1
def available_actions(state):
  current_state_row=R[state,]
  av_act = np.where(current_state_row>=0)[1]
  return av_act

available_act  = available_actions(initial_state)
available_act
array([3, 5])
R
matrix([[ -1,  -1,  -1,  -1,   0,  -1],
        [ -1,  -1,  -1,   0,  -1, 100],
        [ -1,  -1,  -1,   0,  -1,  -1],
        [ -1,   0,   0,  -1,   0,  -1],
        [ -1,   0,   0,  -1,  -1, 100],
        [ -1,   0,  -1,  -1,   0, 100]])
#this function chooses at random which action to be performend with
def sample_next_action(available_actions_range):
  next_action=int(np.random.choice(available_act,1))
  return next_action
#sample next action to be performed
action=sample_next_action(available_act)
action
def update(current_state, action, game):
  max_index = np.where(Q[action,] == np.max(Q[action,]))[1]

  if max_index.shape[0]>1:
    max_index=int(np.random.choice(max_index,size=1))
  else:
    max_index=int(max_index)
  max_value = Q[action, max_index]  

  #Q learning formula
  Q[current_state,action] = R[current_state,action] + gamma*max_value


update(initial_state, action, gamma)  


           
Q
matrix([[0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.]])
#Training
#Train over 10,000 iterations
for i in range(10000):
  current_state = np.random.randint(0, int(Q.shape[0]))
  available_act = available_actions(current_state)
  action = sample_next_action(available_act)
  update(current_state, action, gamma)
print("Trained Q matrix")
print(Q/np.max(Q)*100)
Trained Q matrix
[[  0.    0.    0.    0.   80.    0. ]
 [  0.    0.    0.   64.    0.  100. ]
 [  0.    0.    0.   64.    0.    0. ]
 [  0.   80.   51.2   0.   80.    0. ]
 [  0.   80.   51.2   0.    0.  100. ]
 [  0.   80.    0.    0.   80.  100. ]]
#Testing
#goal_state =5
current_state =1
steps=[current_state]

while current_state != 5:
  next_step_index=np.where(Q[current_state,] == np.max(Q[current_state],))[1]
  if next_step_index.shape[0]>1:
    next_step_index=int(np.random.choice(next_step_index,size=1))
  else:
    next_step_index=int(next_step_index)

  steps.append(next_step_index)
  current_state = next_step_index    
#print selected steps
steps
[1, 5]
Q
matrix([[  0.,   0.,   0.,   0., 400.,   0.],
        [  0.,   0.,   0., 320.,   0., 500.],
        [  0.,   0.,   0., 320.,   0.,   0.],
        [  0., 400., 256.,   0., 400.,   0.],
        [  0., 400., 256.,   0.,   0., 500.],
        [  0., 400.,   0.,   0., 400., 500.]])

