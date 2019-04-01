public class QLearner {  
    /**
     * This method performs Q-learning.
     * @param rewards an n x m matrix containing the reward from state i to state j
     * @param paths array of paths containing states to visit (in order)
     * @param gamma learning rate (discount)
     * @param noIterations number of iterations per path
     * @return a String indicating the preferred action per state: e.g. 321n indicates that the preferred action
     * for state 0 is 3, for 1 is 2, for 2 is 1 and the state 4 is a final state (no allowed actions)
     */
    public String execute(Integer[][] rewards, Integer[][] paths, Double gamma, Integer noIterations)
    {
        // find dimensions of Q
        int n = rewards.length;
        int m = rewards[0].length;
        // declare Q
        final Double[][] Q = new Double[n][m];
        
        // Initialize Q
        for (int i = 0; i < Q.length; i++) {
            for (int j = 0; j < Q[i].length; j++) {

                // if the move is not applicable leave it as null
                if (rewards[i][j] == null) {
                    continue;
                }

                // set arbitrarily to 0
                Q[i][j] = 0.0;
            }
        }
        
        // Do Q-learning
        for (Integer[] path : paths) {
            for (int iter = 1; iter <= noIterations; iter++) {
                execute(Q, rewards, path, gamma);
            }
        }
        
        return policy(Q);
    }

    /**
     * Does perform Q-learning for a single path
     * @param Q an n x m matrix containing Q values
     * @param rewards an n x m matrix containing the reward from state i to state j
     * @param path array of integers indicating which states to visit (in order)
     * @param gamma learning rate (discount)
     */
    private void execute(Double[][] Q, Integer[][] rewards, Integer[] path, Double gamma) {
        for (int i = 0; i < path.length - 1; i++) {
            // initialize max Q-value to min value
            Double maxq = -Double.MAX_VALUE;

            // current state
            int state = path[i];

            // action defines which state to visit next (successor state)
            int action = path[i+1];

            // find max Q of successor state
            for (Double q : Q[action]) {

                // if action is invalid, continue
                if (q == null) {
                    continue;
                }

                // else if q value of action is larger than all previous ones
                if (q > maxq) {
                    maxq = q;
                }
            }

            // Q(s,a) := r(s,a) + gamma * maxQ
            Q[state][action] = rewards[state][action] + gamma * maxq;
        }
    }

    /**
     * Reads Q and computes the corresponding policy
     * @param Q n x m matrix containing Q values
     * @return a String indicating the preferred action per state: e.g. 321n indicates that the preferred action
     * for state 0 is 3, for 1 is 2, for 2 is 1 and the state 4 is a final state (no allowed actions)
     */
    private String policy(Double[][] Q) {
        // initialize policy as empty string
        String policy = "";

        for (int i = 0; i < Q.length; i++) {
            // initialize q value of preferred action, before first iteration set to absolute MIN
            double q = -Double.MAX_VALUE;
            // preferred action, choose invalid action for cases in which state is final
            Integer prefAction = -1;

            // figure out which action would lead to highest q
            for (int action = 0; action < Q[i].length; action++) {

                // if Q value is null, continue (i.e. if the move is not allowed)
                if (Q[i][action] == null) {
                    continue;
                }

                // else if there exists a valid action with higher q value, pick that one as preferred action
                if (Q[i][action] >= q) {
                    q = Q[i][action];
                    prefAction = action;
                }
            }

            // if final state, append 'n'
            if (prefAction == -1) {
                policy = policy + "n ";
                continue;
            }

            // else assign preferred action for state i to policy
            policy = policy + prefAction.toString() + " ";
        }

        return policy;
    }
}