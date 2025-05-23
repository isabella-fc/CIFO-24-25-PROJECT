# Imports
import random
from copy import deepcopy
import pandas as pd
import os
import pickle
import numpy as np


#------------------------------- Genetic Algorithm ------------------------------

# Solution

def create_solution(guest_list, mutation_function, crossover_function):  #Initial configuration that will create the starting random distribution (after that crossovers and mutations will take it from there)
    shuffled = random.sample(guest_list, len(guest_list))
    return {
        'repr': shuffled,  #representation of the guest list shuffled
        'guest_list': guest_list,   #original list
        'table_size': 8, # 8 tables, at all times
        'mutation_function': mutation_function, 
        'crossover_function': crossover_function
    }

def get_tables(solution):   # Function to divide the guests by the number of tables (The guests are already randomized by create_solution) -> Used to calculate the score in the fitness function
    repr = solution['repr']
    table_size = solution['table_size']
    return [repr[i * table_size:(i + 1) * table_size] for i in range(len(repr) // table_size)]

def mutate_solution(solution, mut_prob):    #Applies the mutation to the current solution
    new_repr = solution['mutation_function'](solution['repr'], mut_prob)
    return { **solution, 'repr': new_repr }

def crossover_solutions(sol1, sol2):    #Applies the crossover to the current solution
    repr1, repr2 = sol1['crossover_function'](sol1['repr'], sol2['repr'])
    return (
        { **sol1, 'repr': repr1 },
        { **sol2, 'repr': repr2 }
    )

# Swap Mutation
# Switches 2 random guests in their positions
# These positions can be between tables or in the same table
# This only happens once per generation (one mutation per generation) and if the mutation probability allows it (we used 10%)
def swap_mutation(representation, mut_prob):
    new_repr = deepcopy(representation)
    if random.random() <= mut_prob:
        i, j = random.sample(range(len(new_repr)), 2)
        new_repr[i], new_repr[j] = new_repr[j], new_repr[i]
    return new_repr

# Inter table swap mutation
# Switches 2 random guests in their positions
# The main difference is that can only be guests from 2 different tables, making the exploration of new combinations possible
# Also only happens once per generation with the same probability
def inter_table_pair_swap_mutation(representation, mut_prob, table_size=8):
    new_repr = deepcopy(representation)
    if random.random() < mut_prob:
        num_tables = len(representation) // table_size
        t1, t2 = random.sample(range(num_tables), 2)
        i = random.randint(0, table_size - 1)
        j = random.randint(0, table_size - 1)

        idx1 = t1 * table_size + i
        idx2 = t2 * table_size + j

        new_repr[idx1], new_repr[idx2] = new_repr[idx2], new_repr[idx1]

    return new_repr

# Greedy Local replacement mutation
# Changes the guest with less afinity with the table of a random table
# This function chooses a random table, then identifies the guest with less afinity/score and then searchs for a better replacement in the other tables
# Makes the change only if the table score gets better, if not it cancels 
# If the table score gets better, but the main score don't it still does the trade (we choose this because we want to add diversity even at the cost of total score)
def greedy_local_replacement_mutation(representation, relationship_matrix, mut_prob=0.1, table_size=8):
    new_repr = deepcopy(representation)

    if random.random() >= mut_prob:
        return new_repr

    num_tables = len(representation) // table_size
    tables = [new_repr[i * table_size:(i + 1) * table_size] for i in range(num_tables)]

    target_table_idx = random.randint(0, num_tables - 1) #Chooses a table to improve randomly
    target_table = tables[target_table_idx]

    def guest_contribution(g, table):
        return sum(relationship_matrix[g][other] for other in table if other != g)

    weakest_guest = min(target_table, key=lambda g: guest_contribution(g, target_table)) #identifies the weakest guest in the table

    best_improvement = 0
    best_swap = None

    for other_table_idx in range(num_tables): # searches for a replacement with better affinity
        if other_table_idx == target_table_idx:
            continue
        other_table = tables[other_table_idx]

        for candidate in other_table:
            new_table = [g for g in target_table if g != weakest_guest] + [candidate]
            new_score = sum(
                relationship_matrix[g1][g2]
                for i, g1 in enumerate(new_table)
                for j, g2 in enumerate(new_table)
                if i < j
            )
            old_score = sum(
                relationship_matrix[g1][g2]
                for i, g1 in enumerate(target_table)
                for j, g2 in enumerate(target_table)
                if i < j
            )
            improvement = new_score - old_score

            if improvement > best_improvement:
                best_improvement = improvement
                best_swap = (candidate, other_table_idx)

    if best_swap: # only if it finds one does the trade
        candidate, candidate_table_idx = best_swap
        idx_weak = new_repr.index(weakest_guest)
        idx_candidate = new_repr.index(candidate)
        new_repr[idx_weak], new_repr[idx_candidate] = new_repr[idx_candidate], new_repr[idx_weak]

    return new_repr

# Table Based Crossover
# This crossover generates 2 childs, and 2 parents with each 64 guests distributed by the 8 tables (good because already has the structure)
# For child1 chooses randomly a number of tables to use from parent1
# This tables will be passed directly to the child and after that marks every guest of those tables as already used
# And then iterates every table from parent2 and gets every guest that has not been used yet to fill the remaining tables
# For child2 will do the same but based first on parent2
def table_based_crossover(parent1, parent2):
    num_tables = len(parent1) // 8

    # Convert flat parents to table form
    tables1 = [parent1[i * 8:(i + 1) * 8] for i in range(num_tables)]
    tables2 = [parent2[i * 8:(i + 1) * 8] for i in range(num_tables)]

    # Randomly select k tables from parent1
    k = random.randint(2, num_tables - 2)
    selected_indices = random.sample(range(num_tables), k)

    # Child1
    used_guests = set()
    offspring1_tables = []
    for idx in selected_indices:
        table = tables1[idx]
        offspring1_tables.append(table)
        used_guests.update(table)

    remaining_guests = [g for table in tables2 for g in table if g not in used_guests]
    random.shuffle(remaining_guests)

    while remaining_guests:
        table = remaining_guests[:8]
        offspring1_tables.append(table)
        remaining_guests = remaining_guests[8:]

    offspring1_repr = [g for table in offspring1_tables for g in table]

    # Child2
    used_guests = set()
    offspring2_tables = []
    for idx in selected_indices:
        table = tables2[idx]
        offspring2_tables.append(table)
        used_guests.update(table)

    remaining_guests = [g for table in tables1 for g in table if g not in used_guests]
    random.shuffle(remaining_guests)

    while remaining_guests:
        table = remaining_guests[:8]
        offspring2_tables.append(table)
        remaining_guests = remaining_guests[8:]

    offspring2_repr = [g for table in offspring2_tables for g in table]

    return offspring1_repr, offspring2_repr

# Wrapper to call the table based crossover to be later used on the model (its to use on create_solution so it has the same output)
def crossover_table_based(p1, p2):
    return table_based_crossover(p1, p2)

# Uniform guest crossover
# Creates 2 childs, from 2 parents
# Chooses randomly guest by guest if the guest will be added by parent1 or parent2, and only adds if the guests doesnt already exist(preventing duplication)
# After trying all guests, the remainng spots are filled by shuffling the remaning not chosen guests
# This will be done for child one and child two assuring diversity by the randomness of the solution, only switching the order to grant even more diversity
def uniform_guest_crossover(parent1, parent2):
    length = len(parent1)
    all_guests = set(range(length))

    # Child1
    child1 = []
    used_1 = set()
    for g1, g2 in zip(parent1, parent2):
        chosen = g1 if random.random() < 0.5 else g2
        if chosen not in used_1:
            child1.append(chosen)
            used_1.add(chosen)

    remaining_1 = list(all_guests - used_1) # Fill missing guests
    random.shuffle(remaining_1)
    child1.extend(remaining_1)

    # Child2
    child2 = []
    used_2 = set()
    for g2, g1 in zip(parent2, parent1):  # Switch parent priority, but since is 50/50 its still the same, its only the order that matters
        chosen = g2 if random.random() < 0.5 else g1
        if chosen not in used_2:
            child2.append(chosen)
            used_2.add(chosen)

    remaining_2 = list(all_guests - used_2) # Fill remaining guests
    random.shuffle(remaining_2)
    child2.extend(remaining_2)

    return child1, child2

# Another wrapper function to later utilize the uniform crossover in the create_solution code
def crossover_uniform(parent1, parent2):
    return uniform_guest_crossover(parent1, parent2)

def initialize_population(pop_size, guest_list, mutation_fn, crossover_fn): # Creates the random initial solutions of the GA, making use of the create_solution passed mutation and crossover functions
    return [create_solution(guest_list, mutation_fn, crossover_fn) for _ in range(pop_size)]

def evaluate_population(population, relationship_matrix): # Utilizes the fitness function to score each solution with the relationships of the guests
    return [(sol, fitness(sol, relationship_matrix)) for sol in population]

# Tournament Selection
# Chooses a group of 3 population from the parents randomly and calculates the fitness of each one of them
# Returns only the one with the biggest fitness
# This function will be used twice per crossover to choose the 2 parents of the solution
def tournament_selection(population, relationship_matrix, k=3): 
    selected = random.sample(population, k)
    return max(selected, key=lambda sol: fitness(sol, relationship_matrix))

# Roulette Selection
# This is a little different from the tournament since the highest fitness parent not always is chosen
# This choses the parent adding all the fitness of all the parents and making a probablistic view of each one is chosen
# So, if after adding the fitness parent1 has 10% of the fitness total, he has a 10% chance of being chosen
# In case the fitness is zero, it randoms it (We suppose that the hypotesis is near 0, but dont want to risk it) 
def roulette_selection(population, relationship_matrix):
    fitnesses = [fitness(sol, relationship_matrix) for sol in population]
    total_fitness = sum(fitnesses)

    if total_fitness == 0:
        return random.choice(population)  # avoid divide by zero

    probs = [f / total_fitness for f in fitnesses]
    return random.choices(population, weights=probs, k=1)[0]

# GA
# This is the main function that uses the other building functions and gets them all together
# Creates the initial population, executes the generations with selection, crossover and mutations, saves the best solution and returns the best score
# It also shows the best individual per 10 generations so we can have some loading progress on our side
def run_ga(relationship_matrix, mutation_fn, crossover_fn, pop_size, guest_list, num_generations, elite_size, mut_prob, selection_alg):
    population = initialize_population(pop_size, guest_list, mutation_fn, crossover_fn)
    fitness_history = []

    for gen in range(num_generations):
        scored_population = evaluate_population(population, relationship_matrix)
        sorted_population = [pair[0] for pair in sorted(scored_population, key=lambda x: x[1], reverse=True)]

        next_gen = sorted_population[:elite_size]

        while len(next_gen) < pop_size:
            parent1 = selection_alg(population, relationship_matrix)
            parent2 = selection_alg(population, relationship_matrix)
            child1, child2 = crossover_solutions(parent1, parent2)
            child1 = mutate_solution(child1, mut_prob)
            child2 = mutate_solution(child2, mut_prob)
            next_gen.append(child1)
            if len(next_gen) < pop_size:
                next_gen.append(child2)

        population = next_gen

        best = max(population, key=lambda sol: fitness(sol, relationship_matrix))
        best_fitness = fitness(best, relationship_matrix)
        fitness_history.append(best_fitness)

        if gen % 10 == 0 or gen == num_generations - 1:
            print(f"Generation {gen}: Best fitness = {best_fitness}")

    final_scores = evaluate_population(population, relationship_matrix)
    best_solution = max(final_scores, key=lambda x: x[1])[0]
    return best_solution, fitness_history

# Saving on pickle
# This funtion just saves the better solution in a pickle file, and after is called if there is another solution, compares it and decides if it should save or not
def save_best_solution_if_improved(current_best_solution, relationship_matrix, save_path="best_solution.pkl"):
    current_fitness = fitness(current_best_solution, relationship_matrix)

    if os.path.exists(save_path):
        with open(save_path, "rb") as f:
            previous_best = pickle.load(f)
        previous_fitness = fitness(previous_best, relationship_matrix)

        if current_fitness > previous_fitness:
            with open(save_path, "wb") as f:
                pickle.dump(current_best_solution, f)
            print(f"New best solution saved. Improved from {previous_fitness} to {current_fitness}")
        else:
            print(f"Not saved. Current fitness ({current_fitness}) ≤ previous ({previous_fitness})")
    else:
        with open(save_path, "wb") as f:
            pickle.dump(current_best_solution, f)
        print(f"No previous solution found. Best saved with fitness = {current_fitness}")


# Fitness Function
# One of the most important functions of our code, is the function that adds all the score from the relations between every pair of guests seated at the same table
# After that adds all the tables to show the final score of our GA
def fitness(solution, relationship_matrix):
    score = 0
    for table in get_tables(solution):
        for i in range(len(table)):
            for j in range(i + 1, len(table)):
                g1 = table[i]
                g2 = table[j]
                score += relationship_matrix[g1][g2]
    return score

#----------------------------------------------- Simulated Annealing ---------------------------------------------------

# Just generates a random solution of 64 guests per 8 tables to be used in simulated annealing and hillclimbing
def generate_random_seating():
    guests = list(range(64))
    random.shuffle(guests)
    return [guests[i * 8:(i + 1) * 8] for i in range(8)]

# Creates a "neighbor" that is a solution different from the current one, but with a guest traded between tables
def get_random_neighbor(seating):
    neighbor = deepcopy(seating)
    t1, t2 = random.sample(range(8), 2)
    i1, i2 = random.randint(0, 8 - 1), random.randint(0, 8 - 1)
    neighbor[t1][i1], neighbor[t2][i2] = neighbor[t2][i2], neighbor[t1][i1]
    return neighbor

# Works kind of like the wrapper function with the intention of retrieving a dictionary with a list of tables to be used in the same format as the GA
def to_solution_format(seating, table_size=8):
    return {
        'repr': [guest for table in seating for guest in table],
        'table_size': table_size
    }

#Simulated Annealing
# Starts with a random guest list, does the get_random_neighbor to try to create diverstity and iteration after iteration tries to converge to a good solution
def simulated_annealing(matrix, initial_seating, C=1000.0, L=10, H=1.0001, max_iter=100, verbose=True):
    sm_history = []
    current = initial_seating
    current_fitness = fitness(to_solution_format(current), matrix)

    # Track best solution
    best = deepcopy(current)
    best_fitness = current_fitness

    for iter in range(1, max_iter + 1):
        sm_history.append(current_fitness)
        for _ in range(L):
            neighbor = get_random_neighbor(current)
            neighbor_fitness = fitness(to_solution_format(neighbor), matrix)

            if neighbor_fitness >= current_fitness:
                current, current_fitness = neighbor, neighbor_fitness
            else:
                delta = current_fitness - neighbor_fitness
                p = np.exp(-delta / C)
                if random.random() < p:
                    current, current_fitness = neighbor, neighbor_fitness

            # Update best if needed
            if current_fitness > best_fitness:
                best, best_fitness = deepcopy(current), current_fitness

        C = C / H
        if verbose:
            print(f"Iteration {iter}: Fitness = {current_fitness}, Temperature = {C:.2f}")

    # Final output
    if verbose:
        print("Best seating arrangement found:")
        for i, table in enumerate(best):
            print(f"Table {i+1}: {table}")
        print(f"Total happiness score: {best_fitness}")

    return best, sm_history

#----------------------------------------- HillClimbing ------------------------------------------------

#HillCimbing
#Tries many restarts to do some local optimal optimization, since it is a greedy algorythm only takes another solution when the solution gets better 
def run_hill_climbing(matrix, num_restarts=1000, max_iterations=5000, local_trials=50, verbose=True):
    hc_history = []
    global_best_score = float('-inf')
    global_best_tables = None

    for restart in range(num_restarts):
        tables = generate_random_seating()
        current_score = fitness(to_solution_format(tables), matrix)

        for iter_ in range(max_iterations):
            hc_history.append(current_score)
            best_score = current_score
            best_tables = None

            for _ in range(local_trials):
                a, b = random.sample(range(8), 2)
                i = random.randint(0, 8 - 1)
                j = random.randint(0, 8 - 1)

                new_tables = deepcopy(tables)
                new_tables[a][i], new_tables[b][j] = new_tables[b][j], new_tables[a][i]

                score = fitness(to_solution_format(new_tables), matrix)
                if score > best_score:
                    best_score = score
                    best_tables = new_tables

            if best_tables:
                tables = best_tables
                current_score = best_score
            else:
                break

        if current_score > global_best_score:
            global_best_score = current_score
            global_best_tables = deepcopy(tables)

        if verbose and restart % 100 == 0:
            print(f"[Restart {restart}] Best score so far: {global_best_score:.2f}")

    # Final output
    print("Final Hill Climbing Result")
    print(f"Best score found: {global_best_score:.2f}")
    for idx, table in enumerate(global_best_tables, 1):
        print(f"Table {idx}: {table}")

    return to_solution_format(global_best_tables), global_best_score, hc_history
