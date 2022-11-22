x_len = {"A": 208, "B": 76, "C": 76}
y_len = {"0": 1, "1": 161, "2": 233, "3": 74}
x_area_map = {0: "D", 1: "B", 2: "A", 3: "C", 4: "E"}
y_area_map = {0: "0", 1: "1", 2: "2", 3: "3"}

x_idx = [1]*x_len["B"]+[2]*x_len["A"]+[3]*x_len["C"]
y_idx = [3]*y_len["3"]+[2]*y_len["2"]+[1]*y_len["1"]+[0]*y_len["0"]

def x_in_range(x):
	global x_area_map, x_idx
	return str(x_area_map[x_idx[int(x)]])

def y_in_range(y):
	global y_area_map, y_idx
	return str(y_area_map[y_idx[int(y)]])

def x_out_of_range(x):
	if x >= 392:
		return "E"
	elif x < 32:
		return "D"

def y_out_of_range(y):
	return "4"

def reset_y(y):
	return 935-y

def reset_x(x):
	return 424-x

def judge(x, y):
	# y in range
	if y < 468 and y >= 0:
		# x in range
		if x < 392 and x >= 32:
			return x_in_range(x-32)+y_in_range(y)
		# x out of range
		else:
			return x_out_of_range(x)+y_in_range(y)
	# y out of range
	else:
		# x in range
		if x < 392 and x >= 32:
			return x_in_range(x-32)+y_out_of_range(y)
		# x out of range
		else:
			return x_out_of_range(x)+y_out_of_range(y)

#convert coordinate to area
def to_area(x_list, y_list):
	area_list = []
	for i in range(len(x_list)):
		x = x_list[i]
		y = y_list[i]

		# UP PLACE and middle line
		if y <= 468:
			area_list.append(judge(x, y))

		# DOWN PLACE
		elif y > 468:
			area_list.append(judge(reset_x(x), reset_y(y)))
		else:
			area_list.append('')
	return area_list