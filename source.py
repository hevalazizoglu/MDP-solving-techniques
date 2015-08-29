
from copy import deepcopy
from math import fabs

class Node:

	def __init__(self, n_id, n_type):
		self.id = n_id
		self.type = n_type


class Transition:

	def __init__(self, node1_id, node2_id, reward):
		self.node1_id = node1_id
		self.node2_id = node2_id
		self.reward = reward

class Action:

	def __init__(self, reward, prob_list):
		self.reward = reward
		self.prob_list = prob_list


class QLearning:

	def __init__(self, round_nodes_list, room_nodes_list, cross_nodes_list, 
		discount_factor, transitions, learning_rate):
		self.round_nodes_list = round_nodes_list
		self.room_nodes_list = room_nodes_list
		self.cross_nodes_list = cross_nodes_list
		self.discount_factor = discount_factor
		self.transitions = transitions
		self.learning_rate = learning_rate

		self.r_matrix = {}
		self.q_matrix = {}
		self.v_matrix = {}

		self.fillRMatrix()
		self.fillQMatrix()
		self.fillVMatrix()

	def fillRMatrix(self):
		for transition in self.transitions:
			self.r_matrix.update({(transition.node1_id, transition.node2_id): transition.reward})

	def fillQMatrix(self):
		for key in self.r_matrix.keys():
			self.q_matrix.update({key:0})

	def fillVMatrix(self):
		self.v_matrix.clear()
		self.v_matrix.update(deepcopy(self.q_matrix))

	def get_max_q(self, list_of_next):
		value_list = []
		for key in list_of_next:
			value_list.append(self.v_matrix[key])
		if len(value_list) != 0:
			r = max(value_list)
			if r > 0:
				return r
			else:
				return 0
		else:
			return 0

	def update_q_matrix(self, episode):
		current_index = 0
		while True:
			if int(episode[current_index]) not in [node.id for node in room_nodes_list+cross_nodes_list] and current_index!=len(episode)-1:
				reward = self.r_matrix[(int(episode[current_index]),int(episode[current_index+1]))]
				list_of_next = [key for key in self.q_matrix.keys() if key[0]==int(episode[current_index+1])]
				value_of_state = self.get_max_q(list_of_next)
				exploitation = (1-self.learning_rate)*self.q_matrix[(int(episode[current_index]),int(episode[current_index+1]))]
				exploration = learning_rate * (reward + self.discount_factor * value_of_state)
				self.q_matrix[(int(episode[current_index]),int(episode[current_index+1]))] = exploitation + exploration
				current_index = current_index+1
			else:
				break
		self.fillVMatrix()
		self.pretty_print()

	def pretty_print(self):
		length = len(round_nodes_list) + len(room_nodes_list) + len(cross_nodes_list)
		all_possible_keys = [(i,j) for i in range(0,length) for j in range(0,length)]
		available_keys = self.q_matrix.keys()
		print_string = ""
		for key in all_possible_keys:
			if key[1] != length-1:
				if key not in available_keys:
					print_string += "_ "
				else:
					print_string += str((self.q_matrix[key])) + " "
			else:
				if key not in available_keys:
					print_string += "_ "
				else:
					print_string += str(self.q_matrix[key]) + " "			
				print print_string
				print_string = ""
		print "\n"


class PI:

	def __init__(self, room_nodes_list, star_nodes_list, goal_node, discount_factor,
		actions_for_nodes, actions):
		self.room_nodes_list = room_nodes_list
		self.star_nodes_list = star_nodes_list
		self.goal_node = goal_node
		self.discount_factor = discount_factor
		self.actions = actions
		self.actions_for_nodes = actions_for_nodes

		self.value_vector = {}
		self.policy_vector = {}

		self.fillValueVector()
		self.fillPolicyVector()

	def fillValueVector(self):
		for node in room_nodes_list+star_nodes_list+[goal_node]:
			self.value_vector.update({node.id:0})

	def fillPolicyVector(self):
		for key in self.actions_for_nodes.keys():
			self.policy_vector.update({key: min(self.actions_for_nodes[key])})

	"""
	def epsilon_difference(self, temp_values):
		epsilon = 0.001
		max_diff = 0
		for key in self.value_vector:
			if fabs(self.value_vector[key] - temp_values[key]) > max_diff:
				max_diff = fabs(self.value_vector[key] - temp_values[key])
		return max_diff > epsilon
	"""
	def policy_evaluation(self):
		while True:
			significant = False
			temp_values = self.value_vector
			for node in room_nodes_list+star_nodes_list:
				action = self.actions[(self.policy_vector[node.id], node.id)]
				reward = action.reward
				expected_value = 0
				for pair in action.prob_list:
					expected_value += (pair[1]/100.0) * temp_values[pair[0]]
				new = reward + self.discount_factor * expected_value
				diff = new - self.value_vector[node.id]
				if diff > 0.00001:
					significant = True
				self.value_vector[node.id] = new
			if not significant:
				break

	def policy_improvement(self):
		best_action = None
		best_value = float('-inf')
		for node in room_nodes_list+star_nodes_list:
			for act in self.actions_for_nodes[node.id]:
				action = self.actions[(act, node.id)]
				expected_value = 0
				for pair in action.prob_list:
					expected_value += (pair[1]/100.0) * self.value_vector[pair[0]]
				val = action.reward + self.discount_factor * expected_value
				if val > best_value:
					best_value = val
					best_action = act
			self.policy_vector[node.id] = best_action
			best_value = float('-inf')
			best_action = None

	def iterate(self):
		self.policy_evaluation()
		self.policy_improvement()
		self.pretty_print()

	def pretty_print(self):
		key_list = []
		for key in self.value_vector:
			key_list.append(key)
		key_list.sort()
		for key in key_list:
			print key, self.value_vector[key]
		print "\n"
		nodes = ""
		for key in key_list:
			if key != self.goal_node.id:
				for pair in actions[(self.policy_vector[key],key)].prob_list:
					nodes += str(pair[0]) + ","
				print key, nodes[:-1]
				nodes = ""
		print "\n"



if __name__ == "__main__":

	# ------------------  INPUT READ START  --------------------------------------
	file = open("hw4.inp", "r")

	nodes = file.readline().strip("\n")
	index = 0
	round_nodes_list = []
	room_nodes_list = []
	star_nodes_list = []
	cross_nodes_list = []
	goal_node = None

	for char in nodes:
		node = Node(index, char)
		if char=='R':
			round_nodes_list.append(node)
		elif char=='V':
			cross_nodes_list.append(node)
		elif char=='O':
			room_nodes_list.append(node)
		elif char=='S':
			star_nodes_list.append(node)
		else:
			goal_node=node
		index = index+1
	
	discount_factors = file.readline().split()
	learning_rate = float(discount_factors[0])
	discount_factor = float(discount_factors[1])
	
	deterministic_transitions = []
	num_transitions = int(file.readline().strip("\n"))
	while num_transitions>0:
		line = file.readline().split()
		transition = Transition(int(line[0]), int(line[1]), float(line[2]))
		deterministic_transitions.append(transition)
		num_transitions = num_transitions-1

	
	# input part for policy iteration

	num_of_actions = int(file.readline().strip("\n"))

	actions_for_nodes = {}
	match_count = len(room_nodes_list) + len(star_nodes_list)
	while match_count>0:
		line = file.readline().split()
		actions_for_nodes.update({int(line[0]): [int(x) for x in line[1:]]})
		match_count = match_count-1

	new_action = True
	new_tuple = True
	new_reward = False
	action_id = 0
	actions = {}
	pair = None
	reward = 0
	prob_list = []
	ac = None

	while True:
		line = file.readline().strip("\n")
		if line == 'E':
			break;
		if line == '#':
			new_action = True
		elif new_action:
			action_id = int(line.replace(" ","").split(':')[-1])
			new_action = False
		elif line == '$':
			new_tuple = True
			ac = Action(reward, prob_list)
			actions.update({pair:ac})
			prob_list = []
		elif new_tuple:
			pair = (action_id, int(line))
			new_reward = True
			new_tuple = False
		elif new_reward:
			reward = int(line)
			new_reward = False
		else:
			sl = line.split()
			prob_list += [(int(sl[0]),int(sl[1]))]


	# ------------------  INPUT READ ENDED  --------------------------------------

	# ------------------  INTERACTIVE SESSION HANDLING ---------------------------

	termination_flag = 0
	my_universe = QLearning(round_nodes_list, room_nodes_list, cross_nodes_list, discount_factor, deterministic_transitions, learning_rate)
	his_universe = PI(room_nodes_list, star_nodes_list, goal_node, discount_factor, actions_for_nodes, actions)
	while True:
		inp = raw_input()
		if inp == '$':
			termination_flag = termination_flag+1
		if termination_flag == 2:
			break

		if termination_flag == 0:
			episode = inp.split()
			my_universe.update_q_matrix(episode)

		elif termination_flag == 1:
			if inp=='c':
				his_universe.iterate()
				pass

	




