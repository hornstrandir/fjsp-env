id_job 1
id_activity 2
id_operation / alternatives 3

last_activity is counting from 0

operation/ action 

machines start with 0. this is different from the instances where they start from 1

needed machine for non-existing operations is -1

legal_machines does not indicate if machine is free or not


## WHEN IS AN ACTION LEGAL?
1. id_activity == todo_activity
2. machine_legal[needed_machine]
3. There is no other alternative already performing


## ALGORITHM:

1. reset()
2. step()
3. while no legal machine:
4.    increase_time_step()
5. check_no_op()
6. prio_non_final()


step(action):
if action == No_op:
    set all actions to illegal 
    set all machines to illegal
    increase time step until there is a free machine
    reward -= idle time of the machines 
else:
    reward += process time for that action
    end_action = current_time_step + time_needed[action]
    next_time_steps.insert(end_action) # queue
    set needed machine for that action to illegal
    set all alternatives of that action to illegal
    set machines illegal if alternatives where the only one needed this machine
    reward -= idle time of the machines until the next time step
    while no legal machine:
        increase time step
    prio_non_final()
    check_no_op()

increase_time_step():
current_time_step = next_time_steps.pop()


