from climbing_machine_configuration import Climbing_machine_configuration



climbing_machine_c = Climbing_machine_configuration(max_prepro = 2)
##### use function run to test specific combination
#climbing_machine_c.run(mode = "random_detection", prepro_index = [3, 4],
#                       discriminant_index = 4, input_index = 'r', verbose = 2)


######################################################################################################
#choose one mode to turn on climbers
######################################################################################################
#mode = "360_detection"
#mode = "random_movement"
mode = "random_detection"
best_climbers_list = climbing_machine_c.run_all_combinations(mode = mode)
