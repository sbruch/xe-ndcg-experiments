import argparse
import lib
import time
import timeit

import lightgbm as gbm
from scipy import stats


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--trials', type=int, action='store',
                      default=100, help='number of trials')
  parser.add_argument('--threads', type=int, action='store',
                      default=10, help='number of threads')
  parser.add_argument('--early_stopping_rounds', type=int, action='store',
                      default=50, help='early stopping rounds (0 to disable)')
  parser.add_argument('--label_noise', type=float, action='store',
                      default=0., help='fraction of labels to perturb')
  parser.add_argument('--negs_from_others', type=float, action='store', default=0.,
                      help=('fraction by which to augment lists using negative '
                            'documents sampled from other queries'))
  parser.add_argument('--click_impressions', type=int, action='store', default=0.,
                      help='number of impressions when generating clicks')
  parser.add_argument('--click_probs', type=float, nargs='+',
                      help='probability of a click given relevance')
  parser.add_argument('--paths', nargs='+', help='paths to LibSVM-formatted data')

  args = parser.parse_args()

  # Load the data
  start = timeit.default_timer()
  print('loading data...')
  collection = lib.Collection(args.paths)
  print('time to load data: ', timeit.default_timer() - start)

  start = timeit.default_timer()
  metrics = {}

  # Prepare transformations.
  transformations = []
  if args.label_noise > 0:
    transformations.append(
        lib.PerturbLabels(args.label_noise, [.5, .2, .15, .1, .05]))
  if args.negs_from_others > 0:
    transformations.append(
        lib.AugmentListByExternalNegativeSamples(args.negs_from_others))
  if args.click_impressions > 0:
    transformations.append(lib.GenerateClicks(args.click_impressions, args.click_probs))

  # Prepare split configurations.
  configs = [
      # Training
      lib.SplitConfig(0.6, int(0.6*collection.num_queries), transformations),
      # Validation
      lib.SplitConfig(0.2, int(0.2*collection.num_queries), transformations),
      # Test (no transformations)
      lib.SplitConfig(0.2, int(0.2*collection.num_queries)),
  ]

  feval = lib.NDCG([5, 10]).eval

  for trial in range(args.trials):
    splits = collection.generate_splits(configs, params={'max_bin': 255})

    print('Trial {}'.format(trial), flush=True)
    for ranker in ['lambdarank', 'rank_xendcg']:
      # Train on the training and validation sets.
      model = gbm.train(
          params={
              'objective': ranker,
              'min_child_samples': 50,
              'min_child_weight': 0,
              'learning_rate': 0.02,
              'num_leaves': 400,
              'boosting_type': 'gbdt',
              'metric': 'ndcg',
              'ndcg_eval_at': 5,
              'first_metric_only': True,
              'lambdarank_norm': False,
              'sigmoid': 1,
              'tree_learner': 'serial',
              'verbose': 0,
              'objective_seed': int(time.time()),
              'force_row_wise': True,
              'num_threads': args.threads,
          },
          train_set=splits[0],
          num_boost_round=500,
          early_stopping_rounds=args.early_stopping_rounds,
          valid_sets=[splits[1]],
          verbose_eval=50,
          keep_training_booster=True)

      # Compute metrics on the training set.
      eval_results = feval(model.predict(splits[0].get_data()), splits[0])
      for metric, value, _ in eval_results:
        print('Eval on train set: {} {}:{:.5f}'.format(ranker, metric, value), flush=True)

      # Compute metrics on the test set.
      eval_results = feval(model.predict(splits[2].data), splits[2])
      for metric, value, _ in eval_results:
        if metric not in metrics:
          metrics[metric] = {}
        if ranker not in metrics[metric]:
          metrics[metric][ranker] = []
        metrics[metric][ranker].append(value)

        print('Eval on test set: {} {}:{:.5f}'.format(ranker, metric, value), flush=True)

  # Report the collected metrics and compute statistical significance.
  observed = []
  for metric in metrics:
    for ranker1, data1 in metrics[metric].items():
      for ranker2, data2 in metrics[metric].items():
        if ranker1 == ranker2:
          continue
        if ((metric, ranker1, ranker2) in observed or
            (metric, ranker2, ranker1) in observed):
          continue
        observed.append((metric, ranker1, ranker2))

        l = min(len(data1), len(data2))
        tstat, pval = stats.ttest_rel(data1[:l], data2[:l])
        print('-------')
        print('mean {} for {}: {:.5f}'.format(metric, ranker1, sum(data1[:l])/l))
        print('mean {} for {}: {:.5f}'.format(metric, ranker2, sum(data2[:l])/l))
        print('p-value: {}'.format(pval))

  print('running time: ', timeit.default_timer() - start)
