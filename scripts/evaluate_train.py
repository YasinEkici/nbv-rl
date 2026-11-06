"""Evaluate all frozen canonical policies on train, with no learning or selection."""
import nbv
from nbv.frozen_evaluation import evaluate_train

if __name__=='__main__':
    evaluate_train()
