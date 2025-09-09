class InferenceModel:
    """
    A base class for implementing inference models.

    This class provides a framework for building and evaluating inference models.
    It defines common attributes and methods that can be extended by specific
    model implementations.
    """
    def __init__(self):
        """
        Initializes the InferenceModel with placeholders for parameters, metrics, and diagnostics.
        """
        self.kind = "inference"
        self.parameters = None
        self.metrics = None
        self.diagnostics = None

    def run_model(self):
        """
        Executes the complete workflow of the model, including data loading, training,
        evaluation, and report generation.
        """
        self._load_data()
        self._train()
        self._evaluate_model()
        return self._generate_report()
    
    def _load_data(self):
        """
        Loads the data required for training and evaluation.

        This method must be implemented by subclasses.
        """
        raise NotImplementedError("Must implement data loading")
    
    def _train(self):
        """
        Trains the model using the loaded data.

        This method must be implemented by subclasses.
        """
        raise NotImplementedError("Must implement model training")
    
    def _evaluate_model(self):
        """
        Evaluates the model's performance on the test data.

        This method must be implemented by subclasses.
        """
        raise NotImplementedError("Must implement model evaluation")
    
    def _generate_report(self):
        """
        Generates a report summarizing the model's performance and diagnostics.

        This method must be implemented by subclasses.
        """
        raise NotImplementedError("Must implement report generation")
    

class PredictionModel:
    """
    A base class for implementing prediction models.

    This class provides a framework for building and evaluating prediction models.
    It defines common attributes and methods that can be extended by specific
    model implementations.
    """
    def __init__(self):
        """
        Initializes the PredictionModel with placeholders for metrics.
        """
        self.kind = "prediction" 
        self.metrics = None

    def run_model(self):
        """
        Executes the complete workflow of the model, including data loading, training,
        evaluation, and report generation.
        """
        self._load_data()
        pred_df, coef_df, diagnostics = self._train()
        metrics = self._evaluate(pred_df)
        
        # Save model and generate report
        self.save()
        self.generate_report(pred_df, coef_df, metrics, diagnostics)
        return True
    
    def _load_data(self):
        """
        Loads the data required for training and evaluation.

        This method must be implemented by subclasses.
        """
        raise NotImplementedError("Must implement data loading")
    
    def _train(self):
        """
        Trains the model using the loaded data.

        This method must be implemented by subclasses.
        """
        raise NotImplementedError("Must implement model training")
    
    def _evaluate(self):
        """
        Evaluates the model's performance on the test data.

        This method must be implemented by subclasses.
        """
        raise NotImplementedError("Must implement model evaluation")
    
    def _predict(self):
        """
        Generates prediction for unseen data

        This method must be implemented by subclasses.
        """
        raise NotImplementedError("Must implement prediction generation")
    
    def _generate_report(self):
        """
        Generates prediction for unseen data

        This method must be implemented by subclasses.
        """
        raise NotImplementedError("Must implement prediction generation")
    
    