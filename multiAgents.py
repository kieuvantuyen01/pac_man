# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action, BIGNUM=10000):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFoods = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        distancesToGhosts = []
        for ghostState in newGhostStates:
            ghostDistance = manhattanDistance(newPos, ghostState.getPosition())
            distancesToGhosts.append(ghostDistance)
        nearestGhost = min(distancesToGhosts)
        farthestGhost = max(distancesToGhosts)

        nearestGhostPoint = 1.0 / (nearestGhost * 1.0 + 1)
        farthestGhostPoint = 1.0 / (farthestGhost * 1.0 + 1)
        avgGhostPoint = (nearestGhostPoint + farthestGhostPoint) / 2.0

        distancesToFood = []
        nearestFood = 0
        if newFoods:
            for food in newFoods:
                foodDistance = manhattanDistance(newPos, food)
                distancesToFood.append(foodDistance)
            nearestFood = min(distancesToFood)

        nearestFoodPoint = 10.0 / (nearestFood * 1.0 + 1)
        remainFoodPoint = 50.0 / (len(newFoods) * 1.0 + 1)

        point = nearestFoodPoint + remainFoodPoint + avgGhostPoint

        if newScaredTimes[distancesToGhosts.index(nearestGhost)] <= 0:
            point -= 11.0 / (nearestGhost * 1.0 + 1)
        elif newScaredTimes[distancesToGhosts.index(nearestGhost)] > 20:
            point += 100

        return successorGameState.getScore() + point


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        numAgent = gameState.getNumAgents()
        ActionScore = []

        def _removeStop(List):
            return [x for x in List if x != 'Stop']

        def _alphaBeta(curGameState, depth, agentIndex, alpha, beta):
            if (depth >= self.depth and agentIndex == numAgent) or curGameState.isWin() or curGameState.isLose():
                return self.evaluationFunction(curGameState)
            if agentIndex == 0:  # Pacman max
                result = alpha
                for a in _removeStop(curGameState.getLegalActions(agentIndex)):
                    sdot = curGameState.generateSuccessor(agentIndex, a)
                    if agentIndex == numAgent - 1:
                        result = max(result, _alphaBeta(sdot, depth + 1, 0, alpha, beta))
                    else:
                        result = max(result, _alphaBeta(sdot, depth, agentIndex + 1, alpha, beta))
                    alpha = max(alpha, result)
                    if depth == 1:
                        ActionScore.append(result)
                    if beta < alpha:
                        break
            else:  # Ghost min
                result = beta
                for a in _removeStop(curGameState.getLegalActions(agentIndex)):
                    sdot = curGameState.generateSuccessor(agentIndex, a)
                    if depth < self.depth and agentIndex == numAgent - 1:
                        result = min(result, _alphaBeta(sdot, depth + 1, 0, alpha, beta))
                    else:
                        result = min(result, _alphaBeta(sdot, depth, agentIndex + 1, alpha, beta))
                    beta = min(beta, result)
                    if beta < alpha:
                        break
            return result

        result = _alphaBeta(gameState, 1, 0, -1e20, 1e20)
        return _removeStop(gameState.getLegalActions(0))[ActionScore.index(max(ActionScore))]
        util.raiseNotDefined()


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    # Tuyen
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
