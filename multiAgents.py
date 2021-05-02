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
import math

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

    def evaluationFunction(self, currentGameState, action):
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
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        ghostDistances = []
        for ghostState in newGhostStates:
            ghostDistance = manhattanDistance(newPos, ghostState.getPosition())
            ghostDistances.append(ghostDistance)
        nearestGhost = min(ghostDistances)
        farthestGhost = max(ghostDistances)

        nearestGhostScore = 1.0 / (nearestGhost * 1.0 + 1)
        farthestGhostScore = 1.0 / (farthestGhost * 1.0 + 1)
        avgGhostScore = (nearestGhostScore + farthestGhostScore) / 2.0

        newFoodList = newFood.asList()
        foodDistances = []
        nearestFood = 0
        if newFoodList:
            for food in newFoodList:
                foodDistance = manhattanDistance(newPos, food)
                foodDistances.append(foodDistance)
            nearestFood = min(foodDistances)

        nearestFoodScore = 10.0 / (nearestFood * 1.0 + 1)
        remainFoodScore = 50.0 / (len(newFoodList) * 1.0 + 1)

        score = nearestFoodScore + remainFoodScore + avgGhostScore

        if newScaredTimes[ghostDistances.index(nearestGhost)] <= 0:
            score -= 11.0 / (nearestGhost * 1.0 + 1)
        elif newScaredTimes[ghostDistances.index(nearestGhost)] > 20:
            score += 100

        return successorGameState.getScore() + score


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

        def miniMax(agentIndex, depth, gameState):
            if agentIndex == gameState.getNumAgents():
                if depth == self.depth or gameState.isWin() or gameState.isLose():
                    return self.evaluationFunction(gameState)  #When reaching certain game state condition, return point of Pacman 
                else:
                    return miniMax(0, depth + 1, gameState) #Return layer with depth = depth + 1 
            else:
                actions = gameState.getLegalActions(agentIndex) #List of legal actions for an agent
                if len(actions) == 0:                           
                    return self.evaluationFunction(gameState)   #If there is no action left, return point of Pacman
                distances = (miniMax(agentIndex + 1, depth,
                                     gameState.generateSuccessor(agentIndex, action)) for action in actions)  
                if agentIndex == 0:
                    return max(distances) 
                else:
                    return min(distances) 

        # Return best direction for Pacman
        bestScore = float("-inf")
        bestAction = Directions.LEFT
        for action in gameState.getLegalActions(0):
            score = miniMax(1, 1, gameState.generateSuccessor(0, action))
            if score > bestScore:
                bestScore = score
                bestAction = action
        return bestAction


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

        def removeStop(ListAction):
            actions = []
            for action in ListAction:
                if action != 'Stop':
                    actions.append(action)
            return actions

        def alphaBeta(currentState, depth, agentIndex, alpha, beta):
            if (depth >= self.depth and (
                    numAgent == 1 or agentIndex == numAgent)) or currentState.isWin() or currentState.isLose():
                return self.evaluationFunction(currentState)

            if agentIndex == 0:
                result = float('-inf')
                for action in removeStop(currentState.getLegalActions(agentIndex)):
                    nextState = currentState.generateSuccessor(agentIndex, action)
                    if agentIndex == numAgent - 1:
                        result = max(result, alphaBeta(nextState, depth + 1, 0, alpha, beta))
                    else:
                        result = max(result, alphaBeta(nextState, depth, agentIndex + 1, alpha, beta))
                    alpha = max(alpha, result)
                    if depth == 1:
                        ActionScore.append(result)
                    if beta < alpha:
                        break
                return result
            else:
                result = float('inf')
                for action in removeStop(currentState.getLegalActions(agentIndex)):
                    nextState = currentState.generateSuccessor(agentIndex, action)
                    if depth < self.depth and agentIndex == numAgent - 1:
                        result = min(result, alphaBeta(nextState, depth + 1, 0, alpha, beta))
                    else:
                        result = min(result, alphaBeta(nextState, depth, agentIndex + 1, alpha, beta))
                    beta = min(beta, result)
                    if beta < alpha:
                        break
                return result

        alphaBeta(gameState, 1, 0, float('-inf'), float('inf'))
        bestActionIndex = ActionScore.index(max(ActionScore))
        return removeStop(gameState.getLegalActions(0))[bestActionIndex]


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
        return self.expectimax(gameState, self.depth)[1]

    def isEndGame(self, gameState, depth):
        return gameState.isWin() or gameState.isLose() or depth == 0

    def expectimax(self, gameState, depth, agentIndex=0):
        utility = 0 if agentIndex != 0 else -math.inf
        expectimaxAction = None
        if self.isEndGame(gameState, depth):
            return self.evaluationFunction(gameState), expectimaxAction

        numAgents = gameState.getNumAgents()
        actionList = gameState.getLegalActions(agentIndex)

        for action in actionList:
            newState = gameState.generateSuccessor(agentIndex, action)
            if agentIndex == 0:
                newUtility, newAction = self.expectimax(newState, depth, agentIndex + 1)
                if newUtility > utility:
                    utility, expectimaxAction = newUtility, action
            else:
                if agentIndex == (numAgents - 1):
                    nextAgentIndex, nextDepth = 0, depth - 1
                else:
                    nextAgentIndex, nextDepth = agentIndex + 1, depth
                newUtility, newAction = self.expectimax(newState, nextDepth, nextAgentIndex)
                utility += 1.0 * newUtility / len(actionList)

        return utility, expectimaxAction


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    infinity = float('inf')
    score = currentGameState.getScore()
    pacmanPos = currentGameState.getPacmanPosition()
    ghosts = currentGameState.getGhostStates()
    foods = currentGameState.getFood().asList()
    capsules = currentGameState.getCapsules()

    if currentGameState.isWin():
        return infinity
    if currentGameState.isLose():
        return -infinity

    gScore = 0
    for ghost in ghosts:
        ghostDistance = manhattanDistance(pacmanPos, ghost.getPosition())
        if ghost.scaredTimer > 0:
            gScore += max(8 - ghostDistance, 0) * 18
        else:
            gScore -= max(5 - ghostDistance, 0) * 5

    foodDistance = [1.0 / manhattanDistance(pacmanPos, food) for food in foods]
    fScore = max(foodDistance)

    cScore = 0
    for capsule in capsules:
        capsDistance = manhattanDistance(pacmanPos, capsule)
        cScore = max(50.0 / capsDistance, cScore)

    return 1.4 * score + 0.5 * gScore + 4.2 * fScore + 0.9 * cScore


# Abbreviation
better = betterEvaluationFunction
