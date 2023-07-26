import mlflow

class Logger():
    def __init__(self, config, experiment_name = None, run_name = None):
        self.config = config
        self.experiment_name = experiment_name
        self.active_session = False
        mlflow.set_tracking_uri(self.config["mlflow_params"]["tracking_uri"])

    def start(self, description = None, run_name=None):
        self.active_session = True
        mlflow.set_experiment(experiment_name=self.experiment_name)
        mlflow.start_run(description = description, run_name=run_name)
        self.log_initial()

    def end(self):
        self.active_session = False
        mlflow.end_run()

    def log_initial(self):
        mlflow.log_params(self.config["experiment_params"])

    def log_metrics(self, dic, epoch):
        mlflow.log_metrics(dic, step=epoch)
        
    def log_figure(self, fig, fig_name):
        mlflow.log_figure(fig, fig_name)
