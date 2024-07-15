# WS2324 Assignment 2.0.A (Warm-up): Find the Wumpus Cave


**Topic:** Wumpus has lost his way in the wumpus world and I had to write this program to help him get back to his cave.

**Dependencies:** You can download all the dependencies by running ```pip install -r requirements.txt```.

**Running:** Run the command ```python3 client_simple.py env-1.json``` you can choose env between 1-5.

# Description
The Wumpus World is a cell-based map where each cell has a special effect on the agent, the agent here would be the Wumpus itself. The wumpus can give some observation like the cell it's at and the humidity. Using those observations, the program should create a plan that would get the wumpus back home from any possible position given the observation.

There are 5 types of cells: M(Meadows), B(Broad-leaf tree), C(coniferous tree), S(swamps), and W(wumpus cave entrance). M is the basic cell and anything outside the map is to be dealt with as an M. B and C are trees which the wumpus has a 20% chance of identifying wrong. S cells cause a rise in the humidity, if the wumpus is on an S cell then the humidity is raised by 2, if there are any other neighboring S cells it will increase the humidity by 1 per each, the wumpus can also misidentify the humidity, 10% chance it added 1 or 10% chance of deducting 1, which leaves us with 80% chance of getting it correctly.

# Bayes theorem

P(A|B) = (P(B|A) * P(A)) / P(B)

Where:
- P(A|B) is the conditional probability of event A given event B.
- P(B|A) is the conditional probability of event B given event A (likelihood).
- P(A) and P(B) are the probabilities of events A and B respectively (prior).

I have used bayes rule to calculate the probability of starting at each possible cell to find the best plan for all the possible starting positions and especially the ones with higher probability. Using the bayes rule was not straight forward as I imagined since we do not have all the required probabilities for it, later after some research I found out that Normalization can be easily used to fix this problem. Most of the work for this assignment was done on paper just figuring out how to calculate the probability, and then the implementation was self explainatory. 