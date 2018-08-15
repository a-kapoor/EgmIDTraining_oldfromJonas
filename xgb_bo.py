from sklearn.model_selection import train_test_split
from sklearn import metrics
from numpy import argmax
from bayes_opt import BayesianOptimization
import xgboost as xgb
from scipy.special import logit
import pandas as pd

# The space of hyperparameters for the Bayesian optimization
hyperparams_ranges = {'min_child_weight': (1, 20),
                    'colsample_bytree': (0.1, 1),
                    'max_depth': (2, 15),
                    'subsample': (0.5, 1),
                    'gamma': (0, 10),
                    'reg_alpha': (0, 10),
                    'reg_lambda': (0, 10)}

# The default xgboost parameters
xgb_default = {'min_child_weight': 1,
               'colsample_bytree': 1,
               'max_depth': 6,
               'subsample': 1,
               'gamma': 0,
               'reg_alpha': 0,
               'reg_lambda': 1}

def format_params(params):
    """ Casts the hyperparameters to the required type and range.
    """
    p = dict(params)
    p['min_child_weight'] = p["min_child_weight"]
    p['colsample_bytree'] = max(min(p["colsample_bytree"], 1), 0)
    p['max_depth']        = int(p["max_depth"])
    p['subsample']        = max(min(p["subsample"], 1), 0)
    p['gamma']            = max(p["gamma"], 0)
    p['reg_alpha']        = max(p["reg_alpha"], 0)
    p['reg_lambda']       = max(p["reg_lambda"], 0)
    return p

def merge_two_dicts(x, y):
    """ Merge two dictionaries.

    Writing such a function is necessary in Python 2.

    In Python 3, one can just do:
        d_merged = {**d1, **d2}.
    """
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

def overtraining_guard(best_test_auc, verbose=True):

    def callback(env):
        train_auc = env.evaluation_result_list[0][1]
        test_auc = env.evaluation_result_list[1][1]

        if train_auc < best_test_auc:
            return

        print(train_auc - test_auc, 1 - best_test_auc)
        if train_auc - test_auc > 1 - best_test_auc:
            print("We probably have an overtraining problem! Stop boosting.")
            raise xgb.core.EarlyStopException(env.iteration)

    return callback

class XgbBoTrainer:
    """Trains a xgboost classifier with Bayesian-optimized hyperparameters.

    Public attributes:

    Private attributes:
        _random_state (int): seed for random number generation
    """

    def __init__(self, data, X_cols, y_col):
        """The __init__ method for XgbBoTrainer class.

        Args:
            data (pandas.DataFrame): The  data frame containing the features
                                     and target.
            X_cols (:obj:`list` of :obj:`str`) : Names of the feature columns.
            y_col (str) : Name of the colum containing the target of the binary
                          classification. This column has to contain zeros and
                          ones.
        """
        self._random_state      = 2018
        self._num_rounds_max    = 3000
        self._early_stop_rounds = 10
        self._nfold             = 3
        self._init_points       = 5
        self._n_iter            = 1
        self._verbose           = 0

        test_size             = 0.25
        max_n_per_class       = None
        # max_n_per_class       = 200000
        nthread               = 16

        self.params_base = {
            'silent'      : 1-self._verbose,
            'eval_metric' : 'auc',
            'verbose_eval': self._verbose,
            'seed'        : self._random_state,
            'nthread'     : nthread,
            'objective'   : 'binary:logitraw',
            }

        # Entries from the class with more entries are discarded. This is because
        # classifier performance is usually bottlenecked by the size of the
        # dataset for the class with fewer entries. Having one class with extra
        # statistics usually just adds computing time.
        n_per_class = min(min(data[y_col].value_counts()), max_n_per_class)

        # The number of entries per class can be limited by a parameter in case
        # the dataset is just too large for this algorithm to run in a
        # reasonable time.
        if not max_n_per_class is None:
            n_per_class = min(n_per_class, max_n_per_class)

        data = pd.concat([data[data[y_col] == 0].head(n_per_class),
                          data[data[y_col] == 1].head(n_per_class)])

        # Split in testing and training subsamples
        self.X_train, self.X_test, self.y_train, self.y_test = \
                train_test_split(data[X_cols],
                                 data[y_col],
                                 random_state=self._random_state,
                                 test_size=test_size)

        # Save data in xgboosts DMatrix format so the encoding doesn't have to
        # be repeated at every step of the Bayesian optimization.
        self._xgtrain = xgb.DMatrix(self.X_train, label=self.y_train)
        self._xgtest  = xgb.DMatrix(self.X_test, label=self.y_test)

        # Set up the Bayesian optimization
        self._bo = BayesianOptimization(self.evaluate_xgb,
                                        hyperparams_ranges,
                                        random_state=self._random_state)

        # This list will memorize the number of rounds that each step in the
        # Bayesian optimization was trained for before early stopping gets
        # triggered. This way, we can train our final classifier with the
        # correct n_estimators matching to the optimal hyperparameters.
        self._early_stops = []

        # This dictionary will hold the xgboost models created when running
        # this training class.
        self.models = {}

    def evaluate_xgb(self, **hyperparameters):

        params = format_params(merge_two_dicts(self.params_base,
                                               hyperparameters))

        best_test_auc = self._bo.res["max"]["max_val"]
        if best_test_auc is None: best_test_auc = 0.0

        cv_result = xgb.cv(params, self._xgtrain,
                           num_boost_round=self._num_rounds_max,
                           nfold=self._nfold,
                           seed=self._random_state,
                           callbacks=[
                               xgb.callback.early_stop(self._early_stop_rounds,   verbose=False),
                               overtraining_guard(best_test_auc, verbose=True),
                               ])

        self._early_stops.append(len(cv_result))

        return cv_result['test-auc-mean'].values[-1]

    def run(self):

        # Explore the default xgboost hyperparameters
        self._bo.explore({k:[v] for k, v in xgb_default.items()}, eager=True)

        # Do the Bayesian optimization
        self._bo.maximize(init_points=self._init_points, n_iter=self._n_iter, acq='ei')

        # Set up the parameters for the default training
        params_default = merge_two_dicts(self.params_base, xgb_default)
        params_default["n_estimators"] = self._early_stops[0]

        # Set up the parameters for the Bayesian-optimized training
        params_bo = merge_two_dicts(self.params_base,
                                    format_params(self._bo.res["max"]["max_params"]))
        params_bo["n_estimators"] = self._early_stops[argmax(self._bo.res["all"]["values"])]

        # Fit default model to test sample
        self.models["default"] = xgb.XGBClassifier(**params_default)
        self.models["default"].fit(self.X_train, self.y_train)

        # Fit Bayesian-optimized model to test sample
        self.models["bo"] = xgb.XGBClassifier(**params_bo)
        self.models["bo"].fit(self.X_train,self.y_train)

        return self._bo.res

    def get_score(self, model_name):
        return self.models[model_name]._Booster.predict(self._xgtest)

    def get_roc(self, model_name):
        fpr, tpr, thresholds = metrics.roc_curve(self.y_test, self.get_score(model_name))

        return fpr, tpr, thresholds
