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


from optparse import Values
from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
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
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
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
        score = successorGameState.getScore()
        # Food score
        for food in newFood.asList():  # closer food is better
            # +1 to avoid division by 0
            score += 2 / (manhattanDistance(newPos, food) + 1)
        # Ghost score
        for ghost in newGhostStates:
            ghostPos = ghost.getPosition()
            ghostDistance = manhattanDistance(newPos, ghostPos)
            if ghost.scaredTimer > 0:  # Scared ghost is good, closer is better
                score += 2 / (ghostDistance + 1)
            else:  # Normal ghost is bad, closer is worse
                if ghostDistance < 2:  # If ghost is too close, run away
                    score -= 999
                score -= 2 / (ghostDistance + 1)
        # Action score
        if action == Directions.STOP:
            score -= 1  # Stop is bad
        return score


def scoreEvaluationFunction(currentGameState: GameState):
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

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
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

        PACMAN_INDEX = 0

        def getValue(state: GameState, depth: int, agentIndex: int):
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)
            values = []
            # the ghost's depth is the same as pacman's depth
            # the pacman's depth is one less than the last ghost's depth
            # but in case of no ghost...
            nextDepth = depth - 1 if agentIndex == state.getNumAgents() - 1 else depth
            nextAgentIndex = (agentIndex + 1) % state.getNumAgents()
            for action in state.getLegalActions(agentIndex):
                nextStates = state.generateSuccessor(agentIndex, action)
                values.append(getValue(nextStates, nextDepth, nextAgentIndex))
            if agentIndex == PACMAN_INDEX:  # Pacman, max
                return max(values)
            else:  # Ghost, min
                return min(values)

        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions(PACMAN_INDEX)

        # Pacman, max
        # Choose one of the best actions
        scores = [
            getValue(gameState.generateSuccessor(PACMAN_INDEX, action), self.depth, 1)
            for action in legalMoves
        ]
        bestScore = max(scores)
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        return legalMoves[chosenIndex]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        PACMAN_INDEX = 0

        def getValue(
            state: GameState, depth: int, agentIndex: int, alpha: float, beta: float
        ):
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)

            # the ghost's depth is the same as pacman's depth
            # the pacman's depth is one less than the last ghost's depth
            # but in case of no ghost...
            nextDepth = depth - 1 if agentIndex == state.getNumAgents() - 1 else depth
            nextAgentIndex = (agentIndex + 1) % state.getNumAgents()
            value = float("-inf") if agentIndex == PACMAN_INDEX else float("inf")
            if agentIndex == PACMAN_INDEX:  # Pacman, max
                for action in state.getLegalActions(agentIndex):
                    nextStates = state.generateSuccessor(agentIndex, action)
                    value = max(
                        value,
                        getValue(nextStates, nextDepth, nextAgentIndex, alpha, beta),
                    )
                    if value > beta:
                        return value
                    alpha = max(alpha, value)
            else:  # Ghost, min
                for action in state.getLegalActions(agentIndex):
                    nextStates = state.generateSuccessor(agentIndex, action)
                    value = min(
                        value,
                        getValue(nextStates, nextDepth, nextAgentIndex, alpha, beta),
                    )
                    if value < alpha:
                        return value
                    beta = min(beta, value)
            return value

        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions(PACMAN_INDEX)

        # Pacman, max
        # Choose one of the best actions
        bestActions = []
        alpha, beta = float("-inf"), float("inf")
        for action in legalMoves:
            nextState = gameState.generateSuccessor(PACMAN_INDEX, action)
            score = getValue(nextState, self.depth, 1, alpha, beta)
            if score > alpha:
                alpha = score
                bestActions = [action]
            elif score == alpha:
                bestActions.append(action)

        return random.choice(bestActions)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        PACMAN_INDEX = 0

        def getValue(state: GameState, depth: int, agentIndex: int):
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)
            values = []
            # the ghost's depth is the same as pacman's depth
            # the pacman's depth is one less than the last ghost's depth
            # but in case of no ghost...
            nextDepth = depth - 1 if agentIndex == state.getNumAgents() - 1 else depth
            nextAgentIndex = (agentIndex + 1) % state.getNumAgents()
            for action in state.getLegalActions(agentIndex):
                nextStates = state.generateSuccessor(agentIndex, action)
                values.append(getValue(nextStates, nextDepth, nextAgentIndex))
            if agentIndex == PACMAN_INDEX:  # Pacman, max
                return max(values)
            else:  # Ghost, expect(uniformly random)
                if len(values) == 0:  # No legal moves
                    return 0
                return sum(values) / len(values)

        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions(PACMAN_INDEX)

        # Pacman, max
        # Choose one of the best actions
        scores = [
            getValue(gameState.generateSuccessor(PACMAN_INDEX, action), self.depth, 1)
            for action in legalMoves
        ]
        bestScore = max(scores)
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        return legalMoves[chosenIndex]


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # Useful information you can extract from a GameState (pacman.py)
    pos = currentGameState.getPacmanPosition()
    allFood = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()

    "*** YOUR CODE HERE ***"
    score = currentGameState.getScore()
    # Ghost score
    for ghost in ghostStates:
        ghostPos = ghost.getPosition()
        ghostDistance = manhattanDistance(pos, ghostPos)
        if ghost.scaredTimer > 0:  # Scared ghost is good, closer is better
            score += 4 / (ghostDistance + 1)
        else:  # Normal ghost is bad, closer is worse
            if ghostDistance < 2:  # If ghost is too close, run away
                score -= 999
            score -= 4 / (ghostDistance + 1)
    # Capsule score
    capsules = currentGameState.getCapsules()
    for capsule in capsules:
        score += 2 / (manhattanDistance(pos, capsule) + 1)
    # Food score
    for food in allFood.asList():  # closer food is better
        # +1 to avoid division by 0
        score += 1 / (manhattanDistance(pos, food) + 1)

    return score


# Abbreviation
better = betterEvaluationFunction
