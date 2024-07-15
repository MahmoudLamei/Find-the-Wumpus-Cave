"""
    To use this implementation, you simply have to implement `agent_function` such that it returns a legal action.
    You can then let your agent compete on the server by calling
        python3 client_simple.py path/to/your/config.json
    
    The script will keep running forever.
    You can interrupt it at any time.
    The server will remember the actions you have sent.

    Note:
        By default the client bundles multiple requests for efficiency.
        This can complicate debugging.
        You can disable it by setting `single_request=True` in the last line.
"""
import itertools
import json
import logging

import requests
import time
import numpy as np


def agent_function(request_dict):
    print('I got the following request:')
    print(request_dict)
    
    # Putting the map in an array form.
    map = request_dict['map']
    map = map.split('\n')
    map = [[char for char in line] for line in map]
    map = np.array(map)
    observations = request_dict['observations']
    current_cell = observations['current-cell']
    humidity = None
    if 'humidity' in observations:
        humidity = observations['humidity']
    max_time = request_dict['max-time']
    actions = []
    starting_pos, humidity_map = get_info(map, current_cell, humidity)
    action_list = ["GO north", "GO east", "GO south", "GO west"]
    possible_plans = list(itertools.product(action_list, repeat = (int(max_time)+1)))
    
    actions, expected_time = evaluate_plans(map, starting_pos, humidity_map, current_cell, humidity, possible_plans, max_time)
    res = {"actions": actions, "expected-time":expected_time}

    return res 
    # {"actions": ["GO south", "GO east"], "expected-time": 1.5}


# Returns the best plan.
def evaluate_plans(map, starting_positions, humidity_map, current_cell, humidity, possible_plans, time_limit):
    # Has the expected time for each plan after being executed on all the starting positions.
    plans_eval = []
    probabilities = calculate_probabilities(map, starting_positions, humidity_map, current_cell, humidity)
    for plan in possible_plans:
        # Has the resulting time for a plan for each starting position.
        result_times = []
        weighted_times = []
        for i,pos in enumerate(starting_positions):
            time = execute_plan(map, pos, plan, time_limit)
            result_times.append(time)
            weighted_times.append(time*probabilities[i])

        plans_eval.append(sum(weighted_times))
    
    best_plan_indx = np.argmin(plans_eval)
    return possible_plans[best_plan_indx], plans_eval[best_plan_indx]
            

# Executes all the actions in the given plan.
def execute_plan(map, pos, plan, time_limit):
    time = 0
    new_pos = pos
    is_boot_on = False
    for i, step in enumerate(plan):
        # Starting cell
        if i ==0:
            # If the starting cell is a swamp.
            if is_within_map(map,new_pos) and  map[new_pos] == "S":
                is_boot_on = True
                time += 1
            # If the starting cell is not swamp
            else:
                time += 0.5
        # Remaining cells.                
        else:
            # If we're in a swamp cell.
            if is_within_map(map,new_pos) and  map[new_pos] == "S":
                # If the wumpus has his boots on already and going through a swamp.
                if is_boot_on:
                    time += 2
                # If the wumpus doesn't has his boots and going through a swamp.
                else:
                    is_boot_on = True
                    time += 3
            # If the wumpus is not in a swamp cell.
            else:
                # if wumpus isn't in swamp cell and has boots on.
                if is_boot_on:
                    time += 2
                    is_boot_on = False
                # If wumpus isn't in swamp cell and doesn't has boots on.
                else:
                    time += 1
        new_pos = take_step(new_pos, step)
        if is_within_map(map,new_pos) and map[new_pos] == "W" and time <= time_limit:
            return time
        if time > time_limit:
            return time_limit
    
    return time_limit


# Takes one step from the position and returns the new position.
def take_step(pos, action):
    x, y = pos
    match action:
        case "GO north": 
            pos = (x-1, y)
        case "GO east":
            pos = (x, y+1)
        case "GO south":
            pos = (x+1, y)
        case "GO west":
            pos = (x, y-1)
    return pos

# Returns a dictionary with the neighbors positions if inside the map, otherwise it's None.
def get_neighbors(map, loc):
    x, y = loc
    neighbor = {'north': None, 'east': None, 'south': None, 'west': None}
    neighbor['north'] = (x-1, y) if is_within_map(map, (x-1,y)) else None
    neighbor['east'] = (x, y+1) if is_within_map(map, (x, y+1)) else None
    neighbor['south'] = (x+1, y) if is_within_map(map, (x+1, y)) else None
    neighbor['west'] = (x, y-1) if is_within_map(map, (x, y-1)) else None
    return neighbor


# Gets all the starting points cordinates and their humidities from the map. 
def get_info(map, current_cell, humidity):
    humidity_map = None
    if current_cell in ("C", "B"):
        possible_positions = np.argwhere(np.logical_or(map == "C", map == "B"))
    else:
        possible_positions = np.argwhere(map == current_cell)
        
    starting_pos = [(x, y) for x, y in possible_positions]

    if humidity != None:
        humidity_map = []
        temp_pos = []
        for pos in starting_pos:
            current_humidity = calculate_humidity(map, pos)
            if current_humidity in (humidity-1, humidity, humidity+1):
                temp_pos.append(pos)
                humidity_map.append(current_humidity)
        starting_pos = temp_pos

    return starting_pos, humidity_map

# Calculating the probabilities by using Bayes rule.
def calculate_probabilities(map, starting_pos, humidity_map, oc, oh):
    temp_probabilities = []
    for i, pos in enumerate(starting_pos):
        ac = map[pos]
        # Initiation in case it's not C or B.
        cm = 1
        hm = 1  
        # If it's C or B and the wumpus was correct or not.
        cm = 0.8 if ac == oc and oc in ("C","B") else 0.2
        # If humidity is observed and equals the actual humidity.
        if oh != None:
            ah = humidity_map[i]
            hm = 0.8 if ah == oh else 0.1
        likelihood = cm * hm

        map_size = len(map)**2
        prior = 1/map_size

        temp_probabilities.append(likelihood*prior)
    
    # Normalization to calculate the denominator.
    alpha = 1/sum(temp_probabilities)
    probabilities = [x * alpha for x in temp_probabilities]

    return probabilities
    

# Checks if the position within the map boundaries or not.
def is_within_map(map, pos):
    if pos == None:
        return False
    x, y = pos
    rows, columns = map.shape
    return 0 <= x < columns and 0 <= y < rows


# Calculates the humidity of given cell.
def calculate_humidity(map, pos):
    humidity = 0
    if is_within_map(map, pos):        
        if map[pos] == 'S':
            humidity += 2

        # Get neighbors and then check if it's within the map, if so return it's type otherwise return meadow.
        neighbors = get_neighbors(map, pos)
        north = map[neighbors['north']] if is_within_map(map, neighbors['north']) else 'M'
        east = map[neighbors['east']] if is_within_map(map, neighbors['east']) else 'M'
        south = map[neighbors['south']] if is_within_map(map, neighbors['south']) else 'M' 
        west = map[neighbors['west']] if is_within_map(map, neighbors['west']) else 'M'

        # Count the ammount of swamps in neighbors.
        humidity += [north, east, south, west].count('S')
    return humidity


def run(config_file, action_function, single_request=False):
    logger = logging.getLogger(__name__)

    with open(config_file, 'r') as fp:
        config = json.load(fp)
    
    logger.info(f'Running agent {config["agent"]} on environment {config["env"]}')
    logger.info(f'Hint: You can see how your agent performs at {config["url"]}agent/{config["env"]}/{config["agent"]}')

    actions = []
    for request_number in itertools.count():
        logger.debug(f'Iteration {request_number} (sending {len(actions)} actions)')
        # send request
        response = requests.put(f'{config["url"]}/act/{config["env"]}', json={
            'agent': config['agent'],
            'pwd': config['pwd'],
            'actions': actions,
            'single_request': single_request,
        })
        if response.status_code == 200:
            response_json = response.json()
            for error in response_json['errors']:
                logger.error(f'Error message from server: {error}')
            for message in response_json['messages']:
                logger.info(f'Message from server: {message}')

            action_requests = response_json['action-requests']
            if not action_requests:
                logger.info('The server has no new action requests - waiting for 1 second.')
                time.sleep(1)  # wait a moment to avoid overloading the server and then try again
            # get actions for next request
            actions = []
            for action_request in action_requests:
                actions.append({'run': action_request['run'], 'action': action_function(action_request['percept'])})
        elif response.status_code == 503:
            logger.warning('Server is busy - retrying in 3 seconds')
            time.sleep(3)  # server is busy - wait a moment and then try again
        else:
            # other errors (e.g. authentication problems) do not benefit from a retry
            logger.error(f'Status code {response.status_code}. Stopping.')
            break


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    import sys
    run(sys.argv[1], agent_function, single_request=False)
