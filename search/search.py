# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """

    closedSet = set()
    previousStates = dict()

    openStack = util.Stack()
    openStack.push(problem.getStartState())
    while not openStack.isEmpty():
        currentState = openStack.pop()
        closedSet.add(currentState)

        if problem.isGoalState(currentState):
            return buildPath(previousStates, currentState)

        for (successorState, action, cost) in problem.getSuccessors(currentState):
            if successorState not in closedSet:
                previousStates[successorState] = (currentState, action)
                openStack.push(successorState)


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    startState = problem.getStartState()

    previousStates = dict()
    closedSet = set()
    openQueue = util.Queue()

    openQueue.push(startState)
    closedSet.add(startState)

    while not openQueue.isEmpty():
        currentState = openQueue.pop()

        if problem.isGoalState(currentState):
            return buildPath(previousStates, currentState)

        for (successorState, action, cost) in problem.getSuccessors(currentState):
            if successorState not in closedSet:
                previousStates[successorState] = (currentState, action)
                openQueue.push(successorState)
                closedSet.add(successorState)


def uniformCostSearch(problem):
    """Search the node of least total cost first."""

    return aStarSearch(problem, nullHeuristic)


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    startState = problem.getStartState()

    previousStates = dict()
    closedSet = set()

    costSoFar = dict()
    estimateTotalCost = dict()

    openQueue = util.PriorityQueue()
    openQueue.push(startState, 0)
    costSoFar[startState] = 0

    while not openQueue.isEmpty():
        currentState = openQueue.pop()
        closedSet.add(currentState)

        if problem.isGoalState(currentState):
            return buildPath(previousStates, currentState)

        for (successorState, action, cost) in problem.getSuccessors(currentState):
            if successorState not in closedSet:

                cumulativeCost = costSoFar[currentState] + cost

                if isBetterCost(cumulativeCost, costSoFar, successorState):
                    estimateCost = cumulativeCost + heuristic(successorState, problem)
                    previousStates[successorState] = (currentState, action)
                    costSoFar[successorState] = cumulativeCost
                    estimateTotalCost[successorState] = estimateCost
                    openQueue.update(successorState, estimateCost)


def isBetterCost(cumulativeCost, cumulativeCosts, state):
    if state not in cumulativeCosts:
        return True

    return cumulativeCost < cumulativeCosts[state]


def buildPath(previousStates, endState):
    actionStack = util.Stack()
    currentState = endState
    while currentState in previousStates:
        (previousState, action) = previousStates[currentState]
        actionStack.push(action)
        currentState = previousState

    actionList = []
    while not actionStack.isEmpty():
        action = actionStack.pop()
        actionList.append(action)

    return actionList


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
