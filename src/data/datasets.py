from scipy import io


class SessionDataset:
    """Simplifies access to fields from .mat file."""
    def __init__(self, filename):
        data = io.loadmat(filename)['SessionData'][0][0]
        self.num_trials = data['nTrial'].flatten()[0]
        self.rewarded = data['Rewarded'].flatten()
        self.animal_weight = data['AnimalWeight'].flatten()[0]
        self.trial_start_time = data['TrialStartTimestamp'][0]
        self.trial_end_time = data['TrialEndTimestamp'][0]
        self.punished = data['Punished'].flatten()
        self.did_not_choose = data['DidNotChoose'].flatten()
        self.did_not_lever = data['DidNotLever'].flatten() # Not relevant for our task.
        self.trial_stimulus = data['TrialStimulus'].flatten()
        self.iti_jitter = data['ITIjitter'].flatten()
        self.correct_side = data['CorrectSide'].flatten()
        self.stimulus_duration = data['stimDur'].flatten()
        self.decision_gap = data['decisionGap'].flatten()
        self.stimulus_on = data['stimOn'].flatten()
        self.block = data['Block'].flatten()
        self.decision_threshold = [d[0] for d in data['DecisionThreshold'].flatten()]
        self.assisted = data['Assisted'].flatten()
        self.single_spout = data['SingleSpout'].flatten()
        self.auto_reward = data['AutoReward'].flatten()
        self.response_side = data['ResponseSide'].flatten()
        self.ml_water_received = data['mLWaterReceived'].flatten()[0]
        self.performance = data['Performance'].flatten()
        self.left_performance = data['lPerformance'].flatten()
        self.right_performance = data['rPerformance'].flatten()
