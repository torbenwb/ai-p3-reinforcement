# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        # Iterations
        for k in range(0, self.iterations):
            # temporary counter
            k1 = util.Counter()
            # for each state Vk+1 = max Q value from legal actions
            # If no legal actions Vk+1 = 0.0 (default counter value)
            for state in states:
                actions = self.mdp.getPossibleActions(state)
                q = -999999
                for action in actions:
                    nq = self.getQValue(state, action)
                    if nq > q:
                        q = nq

                if q != -999999:
                    k1[state] = q
            self.values = k1

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        nextStates = self.mdp.getTransitionStatesAndProbs(state, action)
        # sum of all actions
        qValue = 0
        for nextState, prob in nextStates:
            # prob = T(s, a, s')
            t = prob
            # r = R(s, a, s')
            r = self.mdp.getReward(state, action, nextState)
            # vksPrime = Vk(s')
            vksPrime = self.getValue(nextState)
            # T(s, a, s') * [(R(s, a, s') + gamma * Vk(s')]
            v = t * (r + self.discount * vksPrime)
            qValue += v
        return qValue

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # If no legal actions return None
        bestAction = None
        maxValue = -999999
        actions = self.mdp.getPossibleActions(state)
        # Simply find the max value and return action
        for action in actions:
            v = self.computeQValueFromValues(state, action)
            if v > maxValue:
                maxValue = v
                bestAction = action
        return bestAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        # Run iterations
        for k in range(0, self.iterations):
            # index represents the index of the state to update
            # this iteration
            index = k % len(states)
            state = states[index]
            actions = self.mdp.getPossibleActions(state)
            q = -999999
            for action in actions:
                nq = self.getQValue(state, action)
                if nq > q:
                    q = nq

            if q != -999999:
                self.values[state] = q

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        # compute predecessors
        states = self.mdp.getStates()
        predecessors = {}
        # predecessors stores key value pairs
        # key = predecessor state
        # value = set of successors
        for state in states:
            actions = self.mdp.getPossibleActions(state)
            for action in actions:
                successors = self.mdp.getTransitionStatesAndProbs(state, action)
                for nextState, prob in successors:
                    if nextState in predecessors:
                        predecessors[nextState].add(state)
                    else:
                        predecessors[nextState] = {state}

        # Priority Queue
        q = util.PriorityQueue()
        # absolute value of the difference between the current value
        # of s in self.value
        # and the highest q value across all possible actions from s
        # push s into the priority queue with priority -diff
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                # determine max Q value across actions
                maxValue = -999999
                for action in self.mdp.getPossibleActions(state):
                    qValue = self.computeQValueFromValues(state, action)
                    if qValue >= maxValue:
                        maxValue = qValue
                # diff = absolute value of difference between
                # the current Q value of state s and the max Q Value
                diff = abs(maxValue - self.values[state])
                # -diff --> Priority Queue implements wih Min Heap
                q.update(state, -diff)

        # for each iteration
        # if q is empty terminate
        # pop a state s off q
        # update s value (if not terminal)
        for k in range(0, self.iterations):
            if q.isEmpty():
                break
            state = q.pop()
            if not self.mdp.isTerminal(state):
                # update s value if not terminal
                # compute new max q value for state
                maxValue = -999999
                for action in self.mdp.getPossibleActions(state):
                    qValue = self.computeQValueFromValues(state, action)
                    if qValue > maxValue:
                        maxValue = qValue
                self.values[state] = maxValue

            # for each predecessor p of s
            # find the absolute value of the difference between the current
            # value of p in self.values and highest Q-value across all possible actions
            # from p. Call this number diff
            for p in predecessors[state]:
                if not self.mdp.isTerminal(p):
                    maxValue = -999999
                    for action in self.mdp.getPossibleActions(p):
                        qValue = self.computeQValueFromValues(p, action)
                        if qValue > maxValue:
                            maxValue = qValue
                    diff = abs(maxValue - self.values[p])
                    if diff > self.theta:
                        q.update(p, -diff)




