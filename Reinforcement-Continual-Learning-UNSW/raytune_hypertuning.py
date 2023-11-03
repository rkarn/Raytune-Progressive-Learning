import tensorflow.keras as keras
from ray.tune import track


class TuneReporterCallback(keras.callbacks.Callback):
    """Tune Callback for Keras.
    
    The callback is invoked every epoch.
    """

    def __init__(self, logs={}):
        self.iteration = 0
        super(TuneReporterCallback, self).__init__()

    def on_epoch_end(self, batch, logs={}):
        self.iteration += 1
        track.log(keras_info=logs, mean_accuracy=logs.get("accuracy"), mean_loss=logs.get("loss"))
        
def tune_UNSW(config):  
    model = create_model(learning_rate=config["lr"], dense_1=config["dense_1"], dense_2=config["dense_2"])  # TODO: Change me.
    checkpoint_callback = ModelCheckpoint(
        "model.h5", monitor='loss', save_best_only=True, save_freq=2)

    # Enable Tune to make intermediate decisions by using a Tune Callback hook. This is Keras specific.
    callbacks = [checkpoint_callback, TuneReporterCallback()]
    
    # Train the model
    hist = model.fit(
        X_train, Y_train, 
        validation_data=(X_test, Y_test),
        verbose=1, 
        batch_size=100, 
        epochs=5, 
        callbacks=callbacks)
    for key in hist.history:
        print(key)

# Random and uniform sampling for hypertune
def random_search():
    import numpy as np; np.random.seed(5)  
    hyperparameter_space = {
        "lr": tune.loguniform(0.001, 0.1),  
        "dense_1": tune.uniform(50, 150),
        "dense_2": tune.uniform(20, 100),
    }  
    num_samples = 10  
    ####################################################################################################
    ################ This is just a validation function for tutorial purposes only. ####################
    HP_KEYS = ["lr", "dense_1", "dense_2"]
    assert all(key in hyperparameter_space for key in HP_KEYS), (
        "The hyperparameter space is not fully designated. It must include all of {}".format(HP_KEYS))
    ######################################################################################################

    ray.shutdown()  # Restart Ray defensively in case the ray connection is lost. 
    ray.init(log_to_driver=False)
    # We clean out the logs before running for a clean visualization later.
    ! rm -rf ~/ray_results/tune_UNSW

    analysis = tune.run(
        tune_UNSW, 
        verbose=1, 
        config=hyperparameter_space,
        num_samples=num_samples)

    assert len(analysis.trials) > 2, "Did you set the correct number of samples?"

    # Obtain the directory where the best model is saved.
    print("You can use any of the following columns to get the best model: \n{}.".format(
        [k for k in analysis.dataframe() if k.startswith("keras_info")]))
    print("=" * 10)
    logdir = analysis.get_best_logdir("keras_info/val_acc", mode="max")
    print('Best model:',analysis.get_best_trial(metric='keras_info/val_acc', mode='max'), 
          analysis.get_best_config(metric='keras_info/val_acc', mode='max'))
    # We saved the model as `model.h5` in the logdir of the trial.
    from tensorflow.keras.models import load_model
    tuned_model = load_model(logdir + "/model.h5")
    tuned_model.summary()

    tuned_loss, tuned_accuracy = tuned_model.evaluate(X_test, Y_test, verbose=0)
    print("Loss is {:0.4f}".format(tuned_loss))
    print("Tuned accuracy is {:0.4f}".format(tuned_accuracy))
    print("The original un-tuned model had an accuracy of {:0.4f}".format(original_accuracy))

#PBT population based sampling 
def mutation_pbtsearch():
    from ray.tune.schedulers import PopulationBasedTraining
    from ray.tune.utils import validate_save_restore
    scheduler = PopulationBasedTraining(
        time_attr="training_iteration",
        metric="mean_accuracy",
        mode="max",
        perturbation_interval=5,
        hyperparam_mutations={
            # distribution for resampling
            "lr": lambda: np.random.uniform(0.0001, 1),
            # allow perturbations within this set of categorical values
            "dense_1": [40, 60, 100], "dense_2": [30, 50, 70], 
        }
    )

    old_dirs = os.listdir('/root/ray_results/')

    ray.shutdown()  # Restart Ray defensively in case the ray connection is lost. 
    ray.init(log_to_driver=False)


    analysis = tune.run(
        tune_UNSW,
        name="PBT_UNSW",
        scheduler=scheduler,
        reuse_actors=True,
        verbose=1,
        stop={
            "training_iteration": 20,
        },
        num_samples=10,

        # PBT starts by training many neural networks in parallel with random hyperparameters. 
        config={
            "lr": tune.uniform(0.001, 1),
            "dense_1": tune.uniform(50, 150), "dense_2": tune.uniform(20, 100),
        })
    print("You can use any of the following columns to get the best model: \n{}.".format(
        [k for k in analysis.dataframe() if k.startswith("keras_info")]))
    print("=" * 10)
    logdir = analysis.get_best_logdir("keras_info/val_acc", mode="max")
    print('Best model:',analysis.get_best_trial(metric='keras_info/val_acc', mode='max'), 
          analysis.get_best_config(metric='keras_info/val_acc', mode='max'))
    # We saved the model as `model.h5` in the logdir of the trial.
    from tensorflow.keras.models import load_model
    tuned_model = load_model(logdir + "/model.h5")
    tuned_model.summary()

    tuned_loss, tuned_accuracy = tuned_model.evaluate(X_test, Y_test, verbose=0)
    print("Loss is {:0.4f}".format(tuned_loss))
    print("Tuned accuracy is {:0.4f}".format(tuned_accuracy))


#ASHA Schedular
def ASHA_search():
    from ray.tune.schedulers import ASHAScheduler
    ray.shutdown()  # Restart Ray defensively in case the ray connection is lost. 
    ray.init(log_to_driver=False)
    custom_scheduler = ASHAScheduler(
        metric='episode_reward_mean',
        mode="max",
        reduction_factor = 2,
        grace_period=1)# TODO: Add a ASHA as custom scheduler here
    hyperparameter_space={
            "lr": tune.uniform(0.001, 1),
            "dense_1": tune.uniform(50, 150), "dense_2": tune.uniform(20, 100),
        }
    analysis = tune.run(
        tune_UNSW, 
        scheduler=custom_scheduler, 
        config=hyperparameter_space, 
        verbose=1,
        num_samples=10,
        #resources_per_trial={"cpu":4},
        name="ASHA_UNSW"  # This is used to specify the logging directory.
    )
    print("You can use any of the following columns to get the best model: \n{}.".format(
        [k for k in analysis.dataframe() if k.startswith("keras_info")]))
    print("=" * 10)
    logdir = analysis.get_best_logdir("keras_info/val_acc", mode="max")
    print('Best model:',analysis.get_best_trial(metric='keras_info/val_acc', mode='max'), 
          analysis.get_best_config(metric='keras_info/val_acc', mode='max'))
    # We saved the model as `model.h5` in the logdir of the trial.
    from tensorflow.keras.models import load_model
    tuned_model = load_model(logdir + "/model.h5")
    tuned_model.summary()

    tuned_loss, tuned_accuracy = tuned_model.evaluate(X_test, Y_test, verbose=0)
    print("Loss is {:0.4f}".format(tuned_loss))
    print("Tuned accuracy is {:0.4f}".format(tuned_accuracy))



#HyperOpt Search 
def hyperopt_search():
    from hyperopt import hp
    from ray.tune.suggest.hyperopt import HyperOptSearch

    # This is a HyperOpt specific hyperparameter space configuration.
    space = {
            "lr": hp.loguniform("lr", -10, -1),
            "dense_1": hp.loguniform("dense_1", 0.3, 3), "dense_2": hp.loguniform("dense_2", 0.2, 2),
        }
    # Create a HyperOptSearch object by passing in a HyperOpt specific search space.
    # Also enforce that only 1 trials can run concurrently.
    hyperopt_search = HyperOptSearch(space, max_concurrent=1, metric="mean_loss", mode="min") 

    # We Remove the dir so that we can visualize tensorboard correctly
    ! rm -rf ~/ray_results/search_algorithm 
    analysis = tune.run(
        tune_UNSW, 
        num_samples=10,  
        search_alg=hyperopt_search,
        verbose=1,
        name="UNSW_search_algorithm",
    )
    print("You can use any of the following columns to get the best model: \n{}.".format(
        [k for k in analysis.dataframe() if k.startswith("keras_info")]))
    print("=" * 10)
    logdir = analysis.get_best_logdir("keras_info/val_acc", mode="max")
    print('Best model:',analysis.get_best_trial(metric='keras_info/val_acc', mode='max'), 
          analysis.get_best_config(metric='keras_info/val_acc', mode='max'))
    # We saved the model as `model.h5` in the logdir of the trial.
    from tensorflow.keras.models import load_model
    tuned_model = load_model(logdir + "/model.h5")
    tuned_model.summary()

    tuned_loss, tuned_accuracy = tuned_model.evaluate(X_test, Y_test, verbose=0)
    print("Loss is {:0.4f}".format(tuned_loss))
    print("Tuned accuracy is {:0.4f}".format(tuned_accuracy))

#Bayesian Search
#https://docs.ray.io/en/latest/tune-searchalg.html#bohb
#https://github.com/ray-project/ray/blob/master/python/ray/tune/examples/bohb_example.py
#https://github.com/ray-project/ray/blob/master/python/ray/tune/suggest/bohb.py
def Bayesian_search():
    from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
    from ray.tune.suggest.bohb import TuneBOHB
    import ConfigSpace as CS
    config_space = CS.ConfigurationSpace()
    config_space.add_hyperparameter(
        CS.UniformFloatHyperparameter("lr", lower=0.001, upper=0.1))
    config_space.add_hyperparameter(
        CS.UniformFloatHyperparameter("dense_1", lower=50, upper=150))
    config_space.add_hyperparameter(
        CS.UniformFloatHyperparameter("dense_2", lower=20, upper=100))
    bohb_hyperband = HyperBandForBOHB(
        time_attr="training_iteration",
        max_t=3,
        reduction_factor=4,
        metric='mean_accuracy', 
        mode='min')
    bohb_search = TuneBOHB(
        config_space, max_concurrent=2, metric='mean_loss', 
        mode='min')

    tune.run(tune_UNSW,
        name="Bayseian_UNSW",
        scheduler=bohb_hyperband,
        search_alg=bohb_search,
        num_samples=10,
        stop={"training_iteration": 50})
    from ray.tune import Analysis as analysis
    analysis = ray.tune.Analysis('/root/ray_results/Bayseian_UNSW') 
    print("You can use any of the following columns to get the best model: \n{}.".format(
        [k for k in analysis.dataframe() if k.startswith("keras_info")]))
    print("=" * 10)
    logdir = analysis.get_best_logdir("keras_info/val_acc", mode="max")
    print('Best model:', analysis.get_best_config(metric='keras_info/val_acc', mode='max'))
    # We saved the model as `model.h5` in the logdir of the trial.
    from tensorflow.keras.models import load_model
    tuned_model = load_model(logdir + "/model.h5")
    tuned_model.summary()

    tuned_loss, tuned_accuracy = tuned_model.evaluate(X_test, Y_test, verbose=0)
    print("Loss is {:0.4f}".format(tuned_loss))
    print("Tuned accuracy is {:0.4f}".format(tuned_accuracy))