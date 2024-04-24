import math

class StepBetaScheduler:
    def __init__(self, T_max, end_value=0.001):
        """
        Set T_max to the maximum epoch
        """
        self.T_max = T_max//2
        self.end_value = end_value
        self.current_step = 0
        self.step_value = end_value / self.T_max
        self.beta = 0.0  # Initial value of beta

    def step(self):
        """Get the current beta value and update the step."""

        # Update the current step
        self.current_step += 1

        # Reset step when it reaches T_max
        # if the train reach 1/2, we fix the end_value to be half of the original
        if self.current_step >= self.T_max:
            self.current_step = 1
            self.end_value *= 0.5

        # Update beta only after the second call to step
        if self.current_step % 20 == 0:
            self.beta = min(self.current_step * self.step_value, self.end_value)


        return 0.000001
