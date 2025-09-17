import logging
import sys
import os
import optuna
import pandas
import logging
logger=logging.getLogger(__name__)

if len(sys.argv) > 1:
    task_name = sys.argv[1]
else:
    task_name = "ambotv1_field"


print(f"The task name is :{task_name} "
      "in checking optuna hyperparam dataset")
study_name = task_name +"_rl_hyperparam"
storage_name = "sqlite:///{}.db".format(study_name)
print(f"storage: {storage_name}")
study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True)
df = study.trials_dataframe(attrs=("number", "datetime_start", "datetime_complete","params", "value", "state"))
result_file_log = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    "../../logs/study_results.log"
)
result_file_csv = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    "../../logs/study_results.csv"
)
logging.basicConfig(filename=result_file_log, level=logging.INFO, filemode='w')

#logger.info(df[['number', 'datetime_start', 'datetime_complete', 'value', 'state']].to_string())
df.to_csv(result_file_csv, sep='\t', index=False)
logger.info(df.to_string())
logger.info(f"Best value: {study.best_value}")
print(f"Best params\n: {study.best_params}")
print(f"Best params\n: {study.best_params}")
print("Best value\n: {}".format(study.best_value))
#print("Best Trial: ", study.best_trial)

# Plot the importance of param
#from plotly.io import show, write_image
#fig = optuna.visualization.plot_param_importances(study)
#write_image(fig,"./param_importance.png","png")
