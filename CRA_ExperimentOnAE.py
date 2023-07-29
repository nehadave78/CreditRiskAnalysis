# Experimentation with different Amplitude Estimator
epsilon = 0.01
alpha = 0.05
from qiskit.algorithms import IterativeAmplitudeEstimation,FasterAmplitudeEstimation,IterativeAmplitudeEstimation, EstimationProblem,AmplitudeEstimation,MaximumLikelihoodAmplitudeEstimation
from qiskit_aer.primitives import Sampler

def MaxLikelihoodAE(evaluation_schedule=3):
    # construct amplitude estimation
    mae = MaximumLikelihoodAmplitudeEstimation(
        evaluation_schedule=3,  # log2 of the maximal Grover power
        sampler=Sampler(run_options={"shots": 500,"seed" : 2}),
    )
    return mae

def FasterAE(delta=0.01,maxiter=3):
    # construct amplitude estimation
    fae = FasterAmplitudeEstimation(
        delta=0.01,  # target accuracy
        maxiter=3,  # determines the maximal power of the Grover operator
        sampler=Sampler(run_options={"shots": 500,"seed" : 2}),
        )
    return fae
    
def IterativeAE():
    # construct amplitude estimation
    iae = IterativeAmplitudeEstimation(
        epsilon_target=epsilon, alpha=alpha, sampler=Sampler(run_options={"shots": 500,"seed" :2})
    )
    return iae

def AE():
    # construct amplitude estimation
    ae = AmplitudeEstimation(
        num_eval_qubits=3, sampler=Sampler(run_options={"shots": 500,"seed" :2})
    )
    return ae