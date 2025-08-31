# diagnostic_network.py (Final Corrected Version)
import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

class DiagnosticBayesianNetwork:
    """
    A class to build, train, and use a Bayesian Network for diagnosing
    mathematical misconceptions based on observed errors.
    """
    def __init__(self):
        """
        Initializes the Bayesian Network structure and inference engine.
        """
        self.conceptual_nodes = [
            'U_Order_Of_Operations', 'U_Distributive_Property',
            'U_Combining_Like_Terms', 'U_Exponent_Rules'
        ]
        self.model = self._create_network_structure()
        self._add_conditional_probability_distributions()
        self.inference = VariableElimination(self.model)

    def _create_network_structure(self):
        """
        Defines the nodes and edges of the Bayesian Network.
        """
        edges = [
            ('U_Order_Of_Operations', 'E_Ignores_PEMDAS'),
            ('U_Distributive_Property', 'E_Distribution_Error'),
            ('U_Distributive_Property', 'E_Partial_Distribution'),
            ('U_Combining_Like_Terms', 'E_Incorrect_Term_Combination'),
            ('U_Exponent_Rules', 'E_Incorrect_Exponent_Operation')
        ]
        model = DiscreteBayesianNetwork(edges)
        return model

    def _add_conditional_probability_distributions(self):
        """
        Defines and adds the Conditional Probability Distributions (CPDs) to the model.
        All evidence nodes (conceptual understandings) have 2 states (Mastery/Misconception),
        so their evidence_card must be [2].
        """
        # Priors for Conceptual Nodes (Latent Variables)
        cpd_U_Order_Of_Operations = TabularCPD('U_Order_Of_Operations', 2, [[0.80], [0.20]])
        cpd_U_Distributive_Property = TabularCPD('U_Distributive_Property', 2, [[0.70], [0.30]])
        cpd_U_Combining_Like_Terms = TabularCPD('U_Combining_Like_Terms', 2, [[0.75], [0.25]])
        cpd_U_Exponent_Rules = TabularCPD('U_Exponent_Rules', 2, [[0.85], [0.15]])

        # Conditional Probabilities for Error Nodes (Observable Variables)
        cpd_E_Ignores_PEMDAS = TabularCPD(
            'E_Ignores_PEMDAS', 2, [[0.95, 0.20], [0.05, 0.80]],
            evidence=['U_Order_Of_Operations'], evidence_card=[2]) # Corrected
            
        cpd_E_Distribution_Error = TabularCPD(
            'E_Distribution_Error', 2, [[0.98, 0.15], [0.02, 0.85]],
            evidence=['U_Distributive_Property'], evidence_card=[2]) # Corrected

        cpd_E_Partial_Distribution = TabularCPD(
            'E_Partial_Distribution', 2, [[0.97, 0.40], [0.03, 0.60]],
            evidence=['U_Distributive_Property'], evidence_card=[2]) # Corrected
            
        cpd_E_Incorrect_Term_Combination = TabularCPD(
            'E_Incorrect_Term_Combination', 2, [[0.95, 0.10], [0.05, 0.90]],
            evidence=['U_Combining_Like_Terms'], evidence_card=[2]) # Corrected

        cpd_E_Incorrect_Exponent_Operation = TabularCPD(
            'E_Incorrect_Exponent_Operation', 2, [[0.96, 0.25], [0.04, 0.75]],
            evidence=['U_Exponent_Rules'], evidence_card=[2]) # Corrected

        self.model.add_cpds(
            cpd_U_Order_Of_Operations, cpd_U_Distributive_Property,
            cpd_U_Combining_Like_Terms, cpd_U_Exponent_Rules,
            cpd_E_Ignores_PEMDAS, cpd_E_Distribution_Error,
            cpd_E_Partial_Distribution, cpd_E_Incorrect_Term_Combination,
            cpd_E_Incorrect_Exponent_Operation
        )
        
        if not self.model.check_model():
            raise ValueError("Model validation failed.")

    def diagnose_misconception(self, observed_errors):
        """
        Performs inference to find the most likely misconception(s) given observed errors.
        """
        if not observed_errors:
            print("No errors observed. Cannot perform diagnosis.")
            return None
            
        print(f"Observed Evidence: {observed_errors}")
        print("Calculating posterior probabilities for misconceptions...")
        
        results = {}
        for node in self.conceptual_nodes:
            posterior_factor = self.inference.query(variables=[node], evidence=observed_errors)
            prob_misconception = posterior_factor.values[1]
            results[node] = prob_misconception

        df = pd.DataFrame(
            list(results.items()),
            columns=['Misconception', 'Probability']
        ).sort_values(by='Probability', ascending=False).reset_index(drop=True)

        return df

if __name__ == '__main__':
    # --- USAGE EXAMPLE ---
    diagnostic_system = DiagnosticBayesianNetwork()
    print("Bayesian Network for Mathematical Diagnosis Initialized.")
    print("-" * 50)
    
    # Simulate an observation: An LLM identifies a distribution error.
    student_error_evidence = {'E_Distribution_Error': 1} # Error was observed

    # Use the network to diagnose the root cause
    diagnosis_results = diagnostic_system.diagnose_misconception(student_error_evidence)
    
    # Print the diagnostic output
    print("\n--- Diagnostic Report ---")
    print(diagnosis_results)
    print("-" * 50)
    
    if not diagnosis_results.empty:
        most_likely_cause = diagnosis_results.iloc[0]
        print(f"\nConclusion: The most probable root cause is a misconception in "
              f"'{most_likely_cause['Misconception']}' with a probability of {most_likely_cause['Probability']:.2%}.")

