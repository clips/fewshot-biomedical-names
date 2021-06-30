import json
import random
import time
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

from ranking_utils import RankingUtils
from encoder_base import BaseFNN

from reach import Reach
from scipy.stats import spearmanr



class Encoder(BaseFNN):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.train_data = None
        self.validation_data = None
        self.training_names = None
        self.training_clusters = None
        self.cluster_prototypes = None

        self.negative_samples_train = None
        self.negative_samples_validation = None

    def create_prototype(self, strings, embeddings):

        embs = []
        for s in strings:
            emb_index = embeddings.items[s]
            emb = embeddings.vectors[emb_index]
            embs.append(emb)
        pooled_embedding = np.average(np.array(embs), axis=0)

        return pooled_embedding

    def create_cluster_prototypes(self, provided_embeddings=None, total=False, pretrained=True):

        if provided_embeddings != None:
            embeddings = provided_embeddings
        else:
            if pretrained:
                embeddings = self.pretrained_name_embeddings
            else:
                embeddings = self.extract_online_dan_embeddings()

        clusters = self.clusters if total else self.training_clusters

        print('Creating cluster prototypes...')
        cluster_prototypes = {}
        for label, strings in clusters.items():
            strings = set(strings).intersection(self.training_names)
            cluster_prototypes[label] = self.create_prototype(strings, embeddings)
        items, vectors = zip(*cluster_prototypes.items())
        self.cluster_prototypes = Reach(vectors, items)

    def negative_sampling(self, train_name_embeddings, threshold=1000, amount_negative=1, verbose=True):

        clusters = self.training_clusters

        if verbose:
            print('Negative sampling...')

        self.negative_samples_train = {}
        for anchor, concept in tqdm(self.train_data, disable=not verbose):

            # threshold = min(threshold, len(clusters[concept]) - amount_negative - 1)
            threshold = min(threshold, len(self.training_names) - amount_negative - 1)

            # calculate distances
            reference_idx = train_name_embeddings.items[anchor]
            reference_vector = train_name_embeddings.norm_vectors[reference_idx]
            cosines = train_name_embeddings.norm_vectors.dot(reference_vector.T)

            exclude_names = clusters[concept]
            exclude_idxs = {train_name_embeddings.items[exclude_name] for exclude_name in exclude_names}

            cutoff = amount_negative + threshold
            top_cosines_idxs = np.argpartition(-cosines, cutoff)[:cutoff]
            top_cosines_idxs = [x for x in top_cosines_idxs if x not in exclude_idxs]
            top_cosines = [cosines[i] for i in top_cosines_idxs]

            weights = [1 / (np.clip(1 - x, 0.000000001, 1)) for x in top_cosines]
            total_weight = sum(weights)
            probs = [weight / total_weight for weight in weights]

            negative_names = set()
            while len(negative_names) < amount_negative:
                sampled_idx = np.random.choice(np.array(top_cosines_idxs), p=probs)
                sampled_negative_name = train_name_embeddings.indices[sampled_idx]
                negative_names.add(sampled_negative_name)

            self.negative_samples_train[anchor] = list(negative_names)

        self.negative_samples_validation = {}
        for anchor, concept in tqdm(self.validation_data, disable=not verbose):

            # threshold = min(threshold, len(clusters[concept]) - amount_negative - 1)
            threshold = min(threshold, len(self.training_names) - amount_negative - 1)

            # calculate distances
            reference_idx = self.pretrained_name_embeddings.items[anchor]
            reference_vector = self.pretrained_name_embeddings.norm_vectors[reference_idx]
            cosines = train_name_embeddings.norm_vectors.dot(reference_vector.T)

            exclude_names = clusters[concept]
            exclude_idxs = {train_name_embeddings.items[exclude_name] for exclude_name in exclude_names}

            cutoff = amount_negative + threshold
            top_cosines_idxs = np.argpartition(-cosines, cutoff)[:cutoff]
            top_cosines_idxs = [x for x in top_cosines_idxs if x not in exclude_idxs]
            top_cosines = [cosines[i] for i in top_cosines_idxs]

            weights = [1 / (np.clip(1 - x, 0.000000001, 1)) for x in top_cosines]
            total_weight = sum(weights)
            probs = [weight / total_weight for weight in weights]

            negative_names = set()
            while len(negative_names) < amount_negative:
                sampled_idx = np.random.choice(np.array(top_cosines_idxs), p=probs)
                sampled_negative_name = train_name_embeddings.indices[sampled_idx]
                negative_names.add(sampled_negative_name)

            self.negative_samples_validation[anchor] = list(negative_names)

    def batch_step_siamese(self, positive_samples_batch, normalize=True, train=True):

        if train:
            negative_samples_lookup = self.negative_samples_train
        else:
            negative_samples_lookup = self.negative_samples_validation

        clusters = self.training_clusters

        losses = {}

        # collect all vectors
        anchor_name_batch = []
        positive_name_batch = []
        negative_name_batch = []
        anchor_embeddings = self.pretrained_name_embeddings
        for (anchor, concept) in positive_samples_batch:
            # instead of sampling a parent name, sample all hyponym names of the same parent concept
            matching_hyponym_names = [x for x in clusters[concept] if x != anchor]
            for parent_name in matching_hyponym_names:
                # sample anchor
                anchor_name_idx = anchor_embeddings.items[anchor]
                if normalize:
                    anchor_vector = anchor_embeddings.norm_vectors[anchor_name_idx]
                else:
                    anchor_vector = anchor_embeddings.vectors[anchor_name_idx]
                anchor_name_batch.append(anchor_vector)
                # sample positive
                parent_name_idx = anchor_embeddings.items[parent_name]
                if normalize:
                    positive_vector = anchor_embeddings.norm_vectors[parent_name_idx]
                else:
                    positive_vector = anchor_embeddings.vectors[parent_name_idx]
                positive_name_batch.append(positive_vector)
                # sample a negative name, belonging to a different concept
                negative_names = negative_samples_lookup[anchor]
                for negative_name in negative_names:
                    negative_index = anchor_embeddings.items[negative_name]
                    if normalize:
                        negative_vector = anchor_embeddings.norm_vectors[negative_index]
                    else:
                        negative_vector = anchor_embeddings.vectors[negative_index]
                    negative_name_batch.append(negative_vector)

        # forward passes
        input_anchor_name_batch = torch.FloatTensor(np.array(anchor_name_batch)).to(self.device).reshape(-1, self.input_size)
        online_anchor_name_batch = self.model(input_anchor_name_batch)
        input_positive_name_batch = torch.FloatTensor(np.array(positive_name_batch)).to(self.device).reshape(-1, self.input_size)
        online_positive_name_batch = self.model(input_positive_name_batch)
        input_negative_name_batch = torch.FloatTensor(np.array(negative_name_batch)).to(self.device).reshape(-1, self.input_size)
        online_negative_name_batch = self.model(input_negative_name_batch).reshape(-1, self.amount_negative_names, self.input_size)

        if train:
            assert self.model.training

        positive_name_distance = self.positive_distance(online_anchor_name_batch, online_positive_name_batch)
        negative_name_distance = self.negative_distance(online_anchor_name_batch, online_negative_name_batch)
        triplet_name = self.triplet_loss(positive_name_distance, negative_name_distance)

        losses['positive_name_distance'] = positive_name_distance
        losses['negative_name_distance'] = negative_name_distance
        losses['triplet_name'] = triplet_name

        loss = triplet_name

        losses = {k: v.item() for k, v in losses.items()}

        return loss, losses

    def batch_step_grounding(self, batch, normalize=True, train=True):

        clusters = self.training_clusters

        losses = {}

        anchor_batch = []
        parent_batch = []
        anchor_embeddings = self.pretrained_name_embeddings
        prototype_embeddings = self.cluster_prototypes
        for (anchor, concept) in batch:
            matching_hyponym_names = [x for x in clusters[concept] if x != anchor]
            for _ in matching_hyponym_names:
                # anchor
                name_idx = anchor_embeddings.items[anchor]
                if normalize:
                    anchor_vector = anchor_embeddings.norm_vectors[name_idx]
                else:
                    anchor_vector = anchor_embeddings.vectors[name_idx]
                anchor_batch.append(anchor_vector)
                # parent prototype
                parent_idx = prototype_embeddings.items[concept]
                if normalize:
                    parent_vector = prototype_embeddings.norm_vectors[parent_idx]
                else:
                    parent_vector = prototype_embeddings.vectors[parent_idx]
                # SPECIAL MODIFICATION!!!
                parent_vector = np.average([parent_vector, anchor_vector], axis=0)
                # SPECIAL MODIFICATION!!!
                parent_batch.append(parent_vector)

        if train:
            assert self.model.training

        # pretrained name losses
        anchor_batch = torch.FloatTensor(np.array(anchor_batch)).to(self.device).reshape(-1, self.input_size)
        online_anchor_batch = self.model(anchor_batch)
        parent_prototype_batch = torch.FloatTensor(np.array(parent_batch)).to(self.device).reshape(-1, self.input_size)

        grounding_loss = self.pretrained_loss(online_anchor_batch, parent_prototype_batch)

        losses['grounding'] = grounding_loss

        loss = grounding_loss
        losses = {k: v.item() for k, v in losses.items()}

        return loss, losses

    def sample_training_data(self, few_shot, seed, resample=True):

        if not resample:
            return

        train_data = {}
        validation_data = {}
        for label, names in sorted(self.clusters.items()):
            random.seed(seed)
            selected = random.sample(names, few_shot*2)
            for name in selected[:few_shot]:
                train_data[name] = label
            for name in selected[few_shot:]:
                validation_data[name] = label
        self.train_data = list(train_data.items())
        self.validation_data = list(validation_data.items())
        print('Train data:', len(self.train_data), 'Validation data:', len(self.validation_data))
        self.training_names = {name for name, concept in self.train_data}

        train_clusters = defaultdict(set)
        for name, concept in self.train_data:
            train_clusters[concept].add(name)
        self.training_clusters = train_clusters

    @staticmethod
    def process_losses(losses):

        avg_losses = defaultdict(list)
        for loss_dict in losses:
            for loss_type, loss in loss_dict.items():
                avg_losses[loss_type].append(loss)
        avg_losses = {k: np.mean(v) for k, v in avg_losses.items()}

        return avg_losses

    def train(self, few_shot=5, resample=True, include_validation=True, stopping_criterion=True,
              amount_negative_names=1, reinitialize=False, normalize=True, verbose=False, num_epochs=0,
              seed=1993, outfile=''):

        self.sample_training_data(few_shot=few_shot, seed=seed, resample=resample)
        self.create_cluster_prototypes()

        # self.negative_sampling(train_name_embeddings, verbose=False)
        if reinitialize:
            self.reinitialize_model()
            self.loss_cache = defaultdict(dict)
            self.stopping_criterion_cache = {}

        self.amount_negative_names = amount_negative_names
        assert self.amount_negative_names

        self.stopping_criterion_cache = {}
        if stopping_criterion:
            self.num_epochs = 1000
        elif num_epochs:
            self.num_epochs = num_epochs

        torch.requires_grad = True
        # iterate over epochs
        start = time.time()
        for _ in tqdm(range(self.num_epochs), total=self.num_epochs, disable=not verbose):

            train_siamese_embeddings = self.extract_online_dan_embeddings(provided_names=self.training_names)
            self.negative_sampling(train_siamese_embeddings, verbose=False)

            # determine epoch ref
            epoch_ref = max(self.loss_cache) + 1 if self.loss_cache else 1
            if verbose:
                print('Started epoch {}'.format(epoch_ref))
                print('Training...')

            # iterate over shuffled batches
            train_losses_hyponym = []
            train_losses_grounding = []
            iteration = 0
            random.shuffle(self.train_data)
            for i in tqdm(range(0, len(self.train_data), self.batch_size), disable=not verbose):

                # set model back to train mode
                self.model.train()
                # clear gradients w.r.t. parameters
                self.optimizer.zero_grad()

                batch = self.train_data[i: i + self.batch_size]

                # hyponym siamese
                train_loss_hyponym, level_losses_hyponym = self.batch_step_siamese(batch, normalize=normalize, train=True)
                train_losses_hyponym.append(level_losses_hyponym)

                # grounding step
                train_loss_grounding, level_losses_grounding = self.batch_step_grounding(batch, normalize=normalize, train=True)
                train_losses_grounding.append(level_losses_grounding)

                # combine multi-task losses
                train_loss = torch.sum(torch.stack([
                                                    train_loss_hyponym * self.loss_weights['siamese'],
                                                    train_loss_grounding * self.loss_weights['grounding']
                                                    ]))
                # backpropagate
                train_loss.backward()
                self.optimizer.step()

                iteration += 1

            # update training losses
            avg_train_losses_hyponym = self.process_losses(train_losses_hyponym)
            avg_train_losses_grounding = self.process_losses(train_losses_grounding)
            if verbose:
                print('Iteration: {}. Average training losses:'.format(iteration))
                print(avg_train_losses_hyponym)
                print(avg_train_losses_grounding)

            # save in cache
            self.loss_cache[epoch_ref]['train'] = (avg_train_losses_hyponym,
                                                   avg_train_losses_grounding
                                                   )

            # optionally calculate validation loss
            if include_validation:

                # iterate over all validation data
                if verbose:
                    print('Validating...')
                validation_losses = []
                validation_losses_hyponym = []
                validation_losses_grounding = []
                for i in tqdm(range(0, len(self.validation_data), self.batch_size), disable=True):

                    batch = self.validation_data[i: i + self.batch_size]

                    # hyponym siamese
                    validation_loss_hyponym, level_losses_hyponym = self.batch_step_siamese(batch, normalize=normalize,
                                                                                       train=False)
                    validation_losses_hyponym.append(level_losses_hyponym)

                    # regularization step
                    validation_loss_grounding, level_losses_grounding = self.batch_step_grounding(batch,
                                                                                                normalize=normalize,
                                                                                                train=False)
                    validation_losses_grounding.append(level_losses_grounding)

                    validation_loss = torch.sum(torch.stack([
                        validation_loss_hyponym * self.loss_weights['siamese'],
                        validation_loss_grounding * self.loss_weights['grounding']
                    ]))
                    validation_losses.append(validation_loss.item())

                validation_loss = np.mean(validation_losses)
                avg_validation_losses_hyponym = self.process_losses(validation_losses_hyponym)
                avg_validation_losses_grounding = self.process_losses(validation_losses_grounding)
                if verbose:
                    print('Iteration: {}. Average validation losses:'.format(iteration))
                    print(avg_validation_losses_hyponym)
                    print(avg_validation_losses_grounding)

                # save in cache
                self.loss_cache[epoch_ref]['validation'] = (avg_validation_losses_hyponym,
                                                            avg_validation_losses_grounding)

                # optionally calculate stopping criterion
                if stopping_criterion:
                    if verbose:
                        print('Calculating stopping criterion on validation data...')
                    stopping_value = validation_loss
                    # print(stopping_value)
                    self.stopping_criterion_cache[epoch_ref] = stopping_value
                    stop, best_checkpoint = self.stop_training(epoch_ref)
                    if stop:
                        self.best_checkpoint = best_checkpoint
                        if outfile:
                            data = {'losses': self.loss_cache,
                                    'stopping_criterion': self.stopping_criterion_cache,
                                    'best_checkpoint': best_checkpoint}
                            with open('{}.json'.format(outfile), 'w') as f:
                                json.dump(data, f)
                        return

            # save intermediate results
            if outfile:
                data = {'losses': self.loss_cache,
                        'stopping_criterion': self.stopping_criterion_cache}
                with open('{}.json'.format(outfile), 'w') as f:
                    json.dump(data, f)
                self.save_model('{}_{}.cpt'.format(outfile, epoch_ref))

            if verbose:
                print('-------------------------------------------------------------------------------------------------')
                print('-------------------------------------------------------------------------------------------------')

        print('Finished training!')
        print('Ran {} epochs. Final average training losses: {}.'.format(
            max(self.loss_cache), self.loss_cache[max(self.loss_cache.keys())]
        ))
        end = time.time()
        print('Training time: {} seconds'.format(round(end-start, 2)))

    def stop_training(self, epoch_ref):
        # returns True if stopping criterion has been fulfilled

        lookback_batch = 10
        if epoch_ref <= lookback_batch:
            return False, None

        sorted_values = sorted(self.stopping_criterion_cache.items())
        lookback = sorted_values[epoch_ref-lookback_batch:epoch_ref+1]
        if lookback[-1][1] >= lookback[0][1]:
            stop = True
            best_checkpoint = sorted_values[np.argmin([x for _, x in sorted_values])][0]
        else:
            stop = False
            best_checkpoint = None

        return stop, best_checkpoint

    def synonym_retrieval(self, normalize=True, pretrained=False):

        rank_util = RankingUtils()

        # first encode
        train_names = sorted({name for name, concept in self.train_data})
        train_vectors = [self.pretrained_name_embeddings[x] for x in train_names]
        train_embeddings = Reach(train_vectors, train_names)
        if not pretrained:
            train_embeddings = self.extract_online_dan_embeddings(provided_names=set(train_embeddings.items.keys()), normalize=normalize)
        validation_names = sorted({name for name, concept in self.validation_data})
        validation_vectors = [self.pretrained_name_embeddings[x] for x in validation_names]
        validation_embeddings = Reach(validation_vectors, validation_names)
        if not pretrained:
            validation_embeddings = self.extract_online_dan_embeddings(provided_names=set(validation_embeddings.items.keys()), normalize=normalize)

        # then rank training data for each validation_item
        complete_ranking = []
        for reference, concept in self.validation_data:

            # calculate distances
            reference_idx = validation_embeddings.items[reference]
            reference_vector = validation_embeddings.norm_vectors[reference_idx]
            scores = train_embeddings.norm_vectors.dot(reference_vector.T)

            # rank
            synonym_names = self.training_clusters[concept]
            synonym_idxs = [train_embeddings.items[synonym_name] for synonym_name in synonym_names]
            ranking = np.argsort(-scores)
            ranks = [np.where(ranking == synonym_idx)[0][0] for synonym_idx in synonym_idxs]
            ranks, synonyms = zip(*sorted(zip(ranks, synonym_names)))
            complete_ranking.append((reference, synonyms, ranks))

        ranking = [x[-1] for x in complete_ranking]
        print(rank_util.ranking_accuracy(ranking))
        print(rank_util.mrr(ranking))
        print(rank_util.mean_average_precision(ranking))

        return complete_ranking

    def synonym_retrieval_train(self, normalize=True, pretrained=False):

        # first encode
        train_names = sorted({name for name, concept in self.train_data})
        train_vectors = [self.pretrained_name_embeddings[x] for x in train_names]
        train_embeddings = Reach(train_vectors, train_names)
        if not pretrained:
            train_embeddings = self.extract_online_dan_embeddings(provided_names=set(train_embeddings.items.keys()), normalize=normalize)

        # then rank training data for each validation_item
        complete_ranking = []
        for reference, concept in self.train_data:

            # calculate distances
            reference_idx = train_embeddings.items[reference]
            reference_vector = train_embeddings.norm_vectors[reference_idx]
            scores = train_embeddings.norm_vectors.dot(reference_vector.T)

            # rank
            synonym_names = [x for x in self.training_clusters[concept] if x != reference]
            synonym_idxs = [train_embeddings.items[synonym_name] for synonym_name in synonym_names]
            mask = [1 if x == reference_idx else 0 for x in range(len(train_embeddings.items))]
            scores = np.ma.array(scores, mask=mask)
            ranking = np.argsort(-scores)
            ranks = [np.where(ranking == synonym_idx)[0][0] for synonym_idx in synonym_idxs]
            ranks, synonyms = zip(*sorted(zip(ranks, synonym_names)))
            complete_ranking.append((reference, synonyms, ranks))

        rank_util = RankingUtils()
        ranking = [x[-1] for x in complete_ranking]
        print(rank_util.ranking_accuracy(ranking))
        print(rank_util.mrr(ranking))
        print(rank_util.mean_average_precision(ranking))

        return complete_ranking

    def load_benchmarks(self, data_infile='data/benchmarks.json'):
        with open(data_infile, 'r') as f:
            benchmarks = json.load(f)

        return benchmarks

    def correlation_benchmarks(self, baseline=False, normalize=True):

        self.model.eval()
        self.vectorize.allow_construct_oov()

        corrs = []
        benchmarks = self.load_benchmarks()
        for benchmark, data in benchmarks.items():
            print(benchmark)
            source_names = data['source']
            target_names = data['target']
            sims = data['sims']

            # calculate cosines
            source_vectors = []
            target_vectors = []
            for source, target in zip(source_names, target_names):
                source_vector = np.average(self.vectorize.vectorize_string(source, norm=False), axis=0)
                target_vector = np.average(self.vectorize.vectorize_string(target, norm=False), axis=0)
                if normalize:
                    source_vector = Reach.normalize(source_vector)
                    target_vector = Reach.normalize(target_vector)
                source_vectors.append(source_vector)
                target_vectors.append(target_vector)
            source_vectors = np.array(source_vectors)
            target_vectors = np.array(target_vectors)

            if baseline:
                source_vectors = Reach.normalize(source_vectors)
                target_vectors = Reach.normalize(target_vectors)
                cosines = [x.dot(y.T) for x, y in zip(source_vectors, target_vectors)]
            else:
                source_vectors = torch.FloatTensor(source_vectors).to(self.device).reshape(-1, self.input_size)
                target_vectors = torch.FloatTensor(target_vectors).to(self.device).reshape(-1, self.input_size)
                source_out = self.model(source_vectors)
                target_out = self.model(target_vectors)
                # take the dot product of the outputted reference and synonym embedding
                ref = source_out / source_out.norm(dim=1).reshape(-1, 1)
                syn = target_out / target_out.norm(dim=1).reshape(-1, 1)
                dot_products = torch.stack([torch.mm(x.reshape(1, -1), y.reshape(1, -1).t()) for x, y in zip(ref, syn)], dim=0)
                cosines = dot_products.reshape(-1).detach().cpu().numpy()

            corr = spearmanr(cosines, sims)
            print(corr)
            corrs.append(corr)

        return corrs
