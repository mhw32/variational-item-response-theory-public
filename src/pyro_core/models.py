import os
import numpy as np

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

import pyro
import pyro.poutine as poutine
from pyro.optim import Adam
import pyro.distributions as dist
from pyro.nn import AutoRegressiveNN
from pyro.distributions import TransformedDistribution
from pyro.distributions.transforms import affine_autoregressive

from src.utils import product_of_experts, multivariate_product_of_experts
from src.torch_core.models import (
    AbilityInferenceNetwork,
    ConditionalAbilityInferenceNetwork,
    ItemInferenceNetwork,
)


def irt_model_1pl(
        ability_dim, 
        num_person, 
        num_item, 
        device, 
        response = None, 
        mask = None, 
        annealing_factor = 1,
        nonlinear = False,
    ):
    ability_prior = dist.Normal(
        torch.zeros((num_person, ability_dim), device=device), 
        torch.ones((num_person, ability_dim), device=device),
    )
    with poutine.scale(scale=annealing_factor):
        ability = pyro.sample("ability", ability_prior)

    item_feat_prior = dist.Normal(
        torch.zeros((num_item, 1), device=device), 
        torch.ones((num_item, 1), device=device),
    )
    item_feat = pyro.sample("item_feat", item_feat_prior)
    difficulty = item_feat

    logit = (torch.sum(ability, dim=1, keepdim=True) + difficulty.T).unsqueeze(2)
    
    if nonlinear:
        logit = torch.pow(logit, 2)

    response_mu = torch.sigmoid(logit)

    if mask is not None:
        response_dist = dist.Bernoulli(response_mu).mask(mask)
    else:
        response_dist = dist.Bernoulli(response_mu)

    if response is not None:
        pyro.sample("response", response_dist, obs=response)
    else:
        response = pyro.sample("response", response_dist)
        return response, ability, item_feat


def irt_model_2pl(
        ability_dim, 
        num_person, 
        num_item, 
        device, 
        response = None, 
        mask = None, 
        annealing_factor = 1,
        nonlinear = False,
    ):
    ability_prior = dist.Normal(
        torch.zeros((num_person, ability_dim), device=device), 
        torch.ones((num_person, ability_dim), device=device),
    )
    with poutine.scale(scale=annealing_factor):
        ability = pyro.sample("ability", ability_prior)

    item_feat_prior = dist.Normal(
        torch.zeros((num_item, ability_dim + 1), device=device),
        torch.ones((num_item, ability_dim + 1), device=device),
    )
    with poutine.scale(scale=annealing_factor):
        item_feat = pyro.sample("item_feat", item_feat_prior)

    discrimination, difficulty = item_feat[:, :ability_dim], item_feat[:, ability_dim:]

    logit = (torch.mm(ability, -discrimination.T) + difficulty.T).unsqueeze(2)

    if nonlinear:
        logit = torch.pow(logit, 2)

    response_mu = torch.sigmoid(logit)

    if mask is not None:
        response_dist = dist.Bernoulli(response_mu).mask(mask)
    else:
        response_dist = dist.Bernoulli(response_mu)

    if response is not None:
        pyro.sample("response", response_dist, obs=response)
    else:
        response = pyro.sample("response", response_dist)
        return response, ability, item_feat


def irt_model_3pl(
        ability_dim, 
        num_person, 
        num_item, 
        device, 
        response = None, 
        mask = None, 
        annealing_factor = 1,
        nonlinear = False,
    ):
    ability_prior = dist.Normal(
        torch.zeros((num_person, ability_dim), device=device), 
        torch.ones((num_person, ability_dim), device=device),
    )
    with poutine.scale(scale=annealing_factor):
        ability = pyro.sample("ability", ability_prior)

    item_feat_prior = dist.Normal(
        torch.zeros((num_item, ability_dim + 2), device=device),
        torch.ones((num_item, ability_dim + 2), device=device),
    )
    with poutine.scale(scale=annealing_factor):
        item_feat = pyro.sample("item_feat", item_feat_prior)

    discrimination = item_feat[:, :ability_dim]
    difficulty = item_feat[:, ability_dim:ability_dim+1]
    guess_logit = item_feat[:, ability_dim+1:ability_dim+2]
    guess = torch.sigmoid(guess_logit)

    logit = (torch.mm(ability, -discrimination.T) + difficulty.T).unsqueeze(2)

    if nonlinear:
        logit = torch.pow(logit, 2)

    guess = guess.unsqueeze(0)  
    response_mu = guess + (1. - guess) * torch.sigmoid(logit)

    if mask is not None:
        response_dist = dist.Bernoulli(response_mu).mask(mask)
    else:
        response_dist = dist.Bernoulli(response_mu)

    if response is not None:
        pyro.sample("response", response_dist, obs=response)
    else:
        response = pyro.sample("response", response_dist)
        return response, ability, item_feat


def irt_model_3pl_hierarchical(
        ability_dim, 
        num_person, 
        num_item, 
        device, 
        response = None, 
        mask = None, 
        annealing_factor = 1,
        num_state = 5,
    ):
    """
    Add a global latent variable over items that allows us 
    to share statistics over the independent items. We hope
    this to reduce the amount of data we need to reason about
    difficulty and discrimination.

    We use a switching state generative model. The global latent
    variable is categorical and switches between different 
    item parameters.

    This is not to be used in generating data. It is only to 
    be used for generative modelling.
    """
    ability_prior = dist.Normal(
        torch.zeros((num_person, ability_dim), device=device), 
        torch.ones((num_person, ability_dim), device=device),
    )
    with poutine.scale(scale=annealing_factor):
        ability = pyro.sample("ability", ability_prior)

    global_item_feat_prior = dist.Normal(
        torch.zeros((1, ability_dim + 2), device=device),
        torch.ones((1, ability_dim + 2), device=device),
    )
    with poutine.scale(scale=annealing_factor):
        global_item_feat = pyro.sample("global_item_feat", global_item_feat_prior)
        global_item_feat = global_item_feat.repeat(num_item, 1)

    item_feat_prior = dist.Normal(
        global_item_feat,
        torch.ones((num_item, ability_dim + 2), device=device),
    )
    with poutine.scale(scale=annealing_factor):
        item_feat = pyro.sample("item_feat", item_feat_prior)

    discrimination = item_feat[:, :ability_dim]
    difficulty = item_feat[:, ability_dim:ability_dim+1]
    guess_logit = item_feat[:, ability_dim+1:ability_dim+2]
    guess = torch.sigmoid(guess_logit)

    logit = (torch.mm(ability, -discrimination.T) + difficulty.T).unsqueeze(2)
    guess = guess.unsqueeze(0)  
    response_mu = guess + (1. - guess) * torch.sigmoid(logit)

    if mask is not None:
        response_dist = dist.Bernoulli(response_mu).mask(mask)
    else:
        response_dist = dist.Bernoulli(response_mu)

    if response is not None:
        pyro.sample("response", response_dist, obs=response)
    else:
        response = pyro.sample("response", response_dist)
        return response, ability, item_feat


class VIBO_1PL(nn.Module):

    def __init__(
            self, 
            latent_dim, 
            num_item, 
            hidden_dim = 16, 
            ability_merge = 'mean',
            conditional_posterior = False,
            generative_model = 'irt',
            num_iafs = 0, 
            iaf_dim = 32,
        ):
        super().__init__()
        
        self.latent_dim            = latent_dim
        self.ability_dim           = latent_dim
        self.response_dim          = 1
        self.hidden_dim            = hidden_dim
        self.num_item              = num_item
        self.ability_merge         = ability_merge
        self.conditional_posterior = conditional_posterior
        self.generative_model      = generative_model
        self.num_iafs              = num_iafs
        self.iaf_dim               = iaf_dim

        self._set_item_feat_dim()
        self._set_irt_num()

        if self.num_iafs > 0:
            self.iafs = [
                affine_autoregressive(self.latent_dim, hidden_dims=[self.iaf_dim])
                for _ in range(self.num_iafs)
            ]
            self.iafs_modules = nn.ModuleList(self.iafs)

        if self.conditional_posterior:
            self.ability_encoder = ConditionalAbilityInferenceNetwork(
                self.ability_dim, 
                self.response_dim, 
                self.item_feat_dim, 
                self.hidden_dim, 
                ability_merge = self.ability_merge,
            )
        else:
            self.ability_encoder = AbilityInferenceNetwork(
                self.ability_dim, 
                self.response_dim, 
                self.hidden_dim, 
                ability_merge = self.ability_merge,
            )

        self.item_encoder = ItemInferenceNetwork(self.num_item, self.item_feat_dim) 

        if self.generative_model == 'link':
            self.decoder = LinkedIRT(
                irt_model = f'{self.irt_num}pl',
                hidden_dim = self.hidden_dim,
            )
        elif self.generative_model == 'deep':
            self.decoder = DeepIRT(
                self.ability_dim,
                irt_model = f'{self.irt_num}pl',
                hidden_dim = self.hidden_dim,
            )
        elif self.generative_model == 'residual':
            self.decoder = ResidualIRT(
                self.ability_dim,
                irt_model = f'{self.irt_num}pl',
                hidden_dim = self.hidden_dim,
            )

        self.apply(self.weights_init)

    def _set_item_feat_dim(self):
        self.item_feat_dim = 1

    def _set_irt_num(self):
        self.irt_num = 1
   
    def model(self, response, mask, annealing_factor=1):
        if self.generative_model == 'irt':
            irt_model_fn = globals()[f'irt_model_{self.irt_num}pl']
            return irt_model_fn(
                self.ability_dim, 
                response.size(0), 
                self.num_item, 
                response.device,
                response = response, 
                mask = mask, 
                annealing_factor = annealing_factor,
            )
        else:
            pyro.module("decoder", self.decoder)
            
            ability_prior = dist.Normal(
                torch.zeros((num_person, ability_dim), device=device), 
                torch.ones((num_person, ability_dim), device=device),
            )
            with poutine.scale(scale=annealing_factor):
                ability = pyro.sample("ability", ability_prior)

            item_feat_prior = dist.Normal(
                torch.zeros((num_item, self.item_feat_dim), device=device), 
                torch.ones((num_item, self.item_feat_dim), device=device),
            )
            item_feat = pyro.sample("item_feat", item_feat_prior)

            response_mu = self.decoder(ability, item_feat)

            if mask is not None:
                response_dist = dist.Bernoulli(response_mu).mask(mask)
            else:
                response_dist = dist.Bernoulli(response_mu)

            if response is not None:
                pyro.sample("response", response_dist, obs=response)
            else:
                response = pyro.sample("response", response_dist)
                return response, ability, item_feat

    def guide(self, response, mask, annealing_factor=1):
        pyro.module("item_encoder", self.item_encoder)
        pyro.module("ability_encoder", self.ability_encoder)
        device = response.device

        item_domain = torch.arange(self.num_item).unsqueeze(1).to(device)
        item_feat_mu, item_feat_logvar = self.item_encoder(item_domain)
        item_feat_scale = torch.exp(0.5 * item_feat_logvar)

        with poutine.scale(scale=annealing_factor):
            item_feat = pyro.sample(
                "item_feat", 
                dist.Normal(item_feat_mu, item_feat_scale),
            )

        if self.conditional_posterior:
            ability_mu, ability_logvar = self.ability_encoder(response, mask, item_feat)
        else:
            ability_mu, ability_logvar = self.ability_encoder(response, mask)

        ability_scale = torch.exp(0.5 * ability_logvar)
        ability_dist = dist.Normal(ability_mu, ability_scale) 

        if self.num_iafs > 0:
            ability_dist = TransformedDistribution(ability_dist, self.iafs)
        
        with poutine.scale(scale=annealing_factor):
            ability = pyro.sample("ability", ability_dist)

        return ability_mu, ability_logvar, item_feat_mu, item_feat_logvar

    def generate(self, ability, item_feat):
        difficulty = item_feat
        logit = (torch.sum(ability, dim=1, keepdim=True) + difficulty.T).unsqueeze(2)
        response_mu = torch.sigmoid(logit)
        response_dist = dist.Bernoulli(response_mu)
        response = pyro.sample("response", response_dist)
        return response

    @staticmethod
    def weights_init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            init.xavier_normal_(m.weight.data, gain=init.calculate_gain('relu'))
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm1d):
            pass


class VIBO_2PL(VIBO_1PL):

    def _set_item_feat_dim(self):
        self.item_feat_dim = self.latent_dim + 1
    
    def _set_irt_num(self):
        self.irt_num = 2

    def generate(self, ability, item_feat):
        ability_dim = ability.size(1)
        discrimination, difficulty = item_feat[:, :ability_dim], item_feat[:, ability_dim:]
        logit = (torch.mm(ability, -discrimination.T) + difficulty.T).unsqueeze(2)
        response_mu = torch.sigmoid(logit)
        response_dist = dist.Bernoulli(response_mu)
        response = pyro.sample("response", response_dist)
        return response


class VIBO_3PL(VIBO_2PL):

    def _set_item_feat_dim(self):
        self.item_feat_dim = self.latent_dim + 2

    def _set_irt_num(self):
        self.irt_num = 3

    def generate(self, ability, item_feat):
        ability_dim = ability.size(1)
        discrimination = item_feat[:, :ability_dim]
        difficulty = item_feat[:, ability_dim:ability_dim+1]
        guess_logit = item_feat[:, ability_dim+1:ability_dim+2]
        guess = torch.sigmoid(guess_logit)

        logit = (torch.mm(ability, -discrimination.T) + difficulty.T).unsqueeze(2)
        guess = guess.unsqueeze(0)
        response_mu = guess + (1. - guess) * torch.sigmoid(logit)
        response_dist = dist.Bernoulli(response_mu)
        response = pyro.sample("response", response_dist)
        return response
