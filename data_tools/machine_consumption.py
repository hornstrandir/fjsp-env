import random

nb_machines = [4, 5, 6, 8, 10, 15]
total_range_consumption = [i for i in range(2,20)] #kW

random.seed(2023)

for nb_machines in nb_machines:
    lst = random.choices(total_range_consumption, k=nb_machines)
    print(lst)

