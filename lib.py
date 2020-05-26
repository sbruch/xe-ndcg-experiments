import math
import numpy as np
import random

import lightgbm as gbm


class SplitConfig(object):
  def __init__(self, population_pct, sample_size, transformations=None):
    """Creates a split configuration.

    Args:
        population_pct: (float) The percentage of the original dataset
          to use as the population.
        sample_size: (int) The number of queries to sample from the population
          to form the split.
        transformations: list of `Transformation` objects to apply to
          sampled queries.
    """
    self.population_pct = population_pct
    self.sample_size = sample_size
    self.transformations = transformations

    if self.transformations is None:
      self.transformations = []


class Collection(object):
  """Data structure that holds a collection of queries."""

  def __init__(self, paths):
    self.features = {}
    self.relevances = {}

    for path in paths:
      for line in open(path, "r"):
        items = line.split()
        rel = int(items[0])
        qid = int(items[1].split(":")[1])
        if qid not in self.features:
          self.features[qid] = []
          self.relevances[qid] = []

        self.features[qid].append(
            np.array([float(s.split(':')[1]) for s in items[2:]]))
        self.relevances[qid].append(rel)

    self.qids = [x for x, _ in self.features.items()]

  @property
  def num_queries(self):
    return len(self.qids)

  def generate_splits(self, configs, params=None):
    """Generates splits for training and evaluation.

    Args:
      configs: list of `SplitConfig` objects.
      params: (dict) Parameters to pass to LightGBM.Dataset.

    Returns:
      List of `lightgbm.Dataset` objects.
    """
    # Randomly shuffle the query IDs.
    random.shuffle(self.qids)

    # Gather query IDs for each split population.
    population_qids = []
    lower = 0
    for pct in [c.population_pct for c in configs]:
      upper = int(lower + pct * self.num_queries + 1)
      if upper >= self.num_queries:
        upper = self.num_queries
      population_qids.append(self.qids[lower:upper])
      lower = upper

    # Sample queries to form each split.
    split_qids = []
    for sample_size in [c.sample_size for c in configs]:
      split_qids.append(np.random.choice(
          population_qids[len(split_qids)], sample_size))

    # List of datasets to return
    datasets = []

    for qids in split_qids:
      # Create a deep copy of features and relevances.
      relevances = [np.copy(self.relevances[qid]) for qid in qids]
      features = [np.copy(self.features[qid]) for qid in qids]

      for transform in configs[len(datasets)].transformations:
        features, relevances = transform.apply(features, relevances)

      groups = [len(rels) for rels in relevances]
      relevances = np.concatenate(relevances)
      features = np.concatenate(features).reshape([len(relevances), -1])

      if len(datasets) == 0:
        dataset = gbm.Dataset(data=features, label=relevances,
                              group=groups, params=params,
                              silent=True)
      else:
        dataset = gbm.Dataset(data=features, label=relevances,
                              group=groups, reference=datasets[0],
                              silent=True)
      datasets.append(dataset)

    return datasets


class Transformation(object):
  def apply(self, features, relevances):
    """Applies a transformation.

    Args:
      features: A 3D ndarray.
      relevances: A 2D ndarray.

    Returns:
      A tuple consisting of new features and relevances.
    """
    raise NotImplementedError


class PerturbLabels(Transformation):
  def __init__(self, factor, dist):
    """Creates a `Transformation` to perturb labels.

    Args:
        factor: (float) Percentage of labels to perturb per query.
        dist: list of floats. The probabilities associated with each label.
    """
    self.factor = factor
    self.dist = dist

  def apply(self, features, relevances):
    for idx, rels in enumerate(relevances):
      labels = np.random.choice(len(self.dist), len(rels), p=self.dist)
      v = np.random.rand(len(rels))
      relevances[idx] = np.where(np.less(v, self.factor), labels, rels)
    return features, relevances


class AugmentListByExternalNegativeSamples(Transformation):
  def __init__(self, factor):
    """
    Creates a `Transformation` to augment lists by sampling negative
    examples from other queries.

    Args:
        factor: (float) Factor by which each list will be augmented.
    """
    self.factor = factor

  def apply(self, features, relevances):
    extra_features = []

    for idx in range(len(features)):
      size = int(self.factor * len(features[idx]))
      v = np.random.randint(0, len(features) - 1, size)
      indices = np.where(np.less(v, idx), v, v + 1)

      extras = []
      for r in indices:
        b = np.random.randint(0, len(features[r]))
        extras.append(np.copy(features[r][b]))

      extra_features.append(extras)

    for idx in range(len(features)):
      features[idx] = np.append(features[idx], extra_features[idx])
      relevances[idx] = np.append(
          relevances[idx], np.zeros(len(extra_features[idx])))

    return features, relevances


class GenerateClicks(Transformation):
  def __init__(self, impressions, click_prob):
    """
    Creates a `Transformation` to generate clicks using a random ranker.

    Args:
        impressions: (int) Number of impressions per query.
        click_prob: list of floats. Click probability given relevance.
    """
    self.impressions = impressions
    self.click_prob = click_prob

  def apply(self, features, relevances):
    _features = []
    _relevances = []

    for idx in range(len(features)):
      indices = np.arange(len(features[idx]))

      for _ in range(self.impressions):
        np.random.shuffle(indices)
        v = np.random.rand(len(indices))

        f = []
        clicked = False
        for i in indices:
          f.append(np.copy(features[idx][i]))
          if v[i] <= self.click_prob[relevances[idx][i]]:
            clicked = True
            break

        r = np.zeros(len(f))
        if clicked:
          r[-1] = 1

        _features.append(f)
        _relevances.append(r)

    return _features, _relevances


class NDCG(object):
  def __init__(self, cutoffs):
    self.cutoffs = cutoffs

  def eval(self, preds, data):
    """Computes NDCG at rank cutoff.

    Args:
        preds: list of floats.
        data: A `lightgbm.Dataset` object.
    """
    # Transform the relevance labels and predictions to the correct shape.
    relevances = []
    scores = []
    idx = 0
    for group in data.group:
      relevances.append(data.label[idx:idx + group])
      scores.append(preds[idx:idx + group])
      idx += group

    ndcg_at = {}
    count = 0

    for s, r in zip(scores, relevances):
      # Skip queries with no relevant documents.
      if sum(r) == 0:
        continue
      count += 1

      sorted_by_scores = [i for _,i in sorted(zip(s,r), key=lambda p: p[0], reverse=True)]

      gains_scores = [pow(2, i) - 1. for i in sorted_by_scores]
      gains_rels = sorted(gains_scores, reverse=True)
      discounts = [1./math.log(i+2, 2) for i, _ in enumerate(sorted_by_scores)]

      for cutoff in self.cutoffs:
        dcg = sum([g*d for g, d in zip(gains_scores[:cutoff], discounts[:cutoff])])
        max_dcg = sum([g*d for g, d in zip(gains_rels[:cutoff], discounts[:cutoff])])

        if cutoff not in ndcg_at:
          ndcg_at[cutoff] = 0.
        ndcg_at[cutoff] += dcg / max_dcg

    results = []
    for cutoff in self.cutoffs:
      results.append(('ndcg@{}'.format(cutoff), ndcg_at[cutoff]/count, True))

    return results
