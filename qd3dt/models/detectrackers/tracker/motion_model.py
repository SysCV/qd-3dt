from typing import Tuple
import torch
import torch.nn as nn


def init_module(layer):
    '''
    Initial modules weights and biases
    '''
    for m in layer.modules():
        if isinstance(m, nn.Conv2d) or \
            isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            if m.bias is not None:
                m.bias.data.zero_()

        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d) or \
            isinstance(m, nn.GroupNorm):
            m.weight.data.uniform_()
            if m.bias is not None:
                m.bias.data.zero_()


def init_lstm_module(layer):
    '''
    Initial LSTM weights and biases
    '''
    for name, param in layer.named_parameters():

        if 'weight_ih' in name:
            torch.nn.init.xavier_uniform_(param.data)
        elif 'weight_hh' in name:
            torch.nn.init.orthogonal_(param.data)
        elif 'bias' in name:
            param.data.fill_(0)  # initializing the lstm bias with zeros


class LSTM(nn.Module):
    '''
    Estimating object location in world coordinates
    Prediction LSTM:
        Input: 5 frames velocity
        Output: Next frame location
    Updating LSTM:
        Input: predicted location and observed location
        Output: Refined location
    '''

    def __init__(self,
                 batch_size: int,
                 feature_dim: int,
                 hidden_size: int,
                 num_layers: int,
                 loc_dim: int,
                 dropout: float = 0.0):
        super(LSTM, self).__init__()
        self.batch_size = batch_size
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.loc_dim = loc_dim

        self.loc2feat = nn.Linear(
            loc_dim,
            feature_dim,
        )

        self.pred2vel = nn.Linear(
            hidden_size,
            loc_dim,
            bias=False,
        )

        self.vel2feat = nn.Linear(
            loc_dim,
            feature_dim,
        )

        self.pred_lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_size,
            dropout=dropout,
            num_layers=num_layers,
        )

        self.refine_lstm = nn.LSTM(
            input_size=2 * feature_dim,
            hidden_size=hidden_size,
            dropout=dropout,
            num_layers=num_layers,
        )

        self._init_param()

    def init_hidden(self, device):
        # Before we've done anything, we dont have any hidden state.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(self.num_layers, self.batch_size,
                            self.hidden_size).to(device),
                torch.zeros(self.num_layers, self.batch_size,
                            self.hidden_size).to(device))

    def predict(self, velocity, location, hc_0):
        '''
        Predict location at t+1 using updated location at t
        Input:
            velocity: (num_seq, num_batch, loc_dim), location from previous update
            location: (num_batch, loc_dim), location from previous update
            hc_0: (num_layers, num_batch, hidden_size), tuple of hidden and cell
        Middle:
            embed: (num_seq, num_batch x feature_dim), location feature
            out: (num_seq x num_batch x hidden_size), lstm output
            merge_feat: (num_batch x hidden_size), the predicted residual
        Output:
            hc_n: (num_layers, num_batch, hidden_size), tuple of updated hidden, cell
            output_pred: (num_batch x loc_dim), predicted location
        '''
        num_seq, num_batch, _ = velocity.shape

        # Embed feature to hidden_size
        embed = self.vel2feat(velocity).view(num_seq, num_batch,
                                             self.feature_dim)

        out, (h_n, c_n) = self.pred_lstm(embed, hc_0)

        # Merge embed feature with output
        # merge_feat = h_n + embed
        merge_feat = out[-1]

        output_pred = self.pred2vel(merge_feat).view(num_batch,
                                                     self.loc_dim) + location

        return output_pred, (h_n, c_n)

    def refine(self, location: torch.Tensor, observation: torch.Tensor,
               prev_location: torch.Tensor, confidence: torch.Tensor, hc_0):
        '''
        Refine predicted location using single frame estimation at t+1
        Input:
            location: (num_batch x 3), location from prediction
            observation: (num_batch x 3), location from single frame estimation
            confidence: (num_batch X 1), depth estimation confidence
            hc_0: (num_layers, num_batch, hidden_size), tuple of hidden and cell
        Middle:
            loc_embed: (1, num_batch x feature_dim), predicted location feature
            obs_embed: (1, num_batch x feature_dim), single frame location feature
            embed: (1, num_batch x 2*feature_dim), location feature
            out: (1 x num_batch x hidden_size), lstm output
            merge_feat: same as out
        Output:
            hc_n: (num_layers, num_batch, hidden_size), tuple of updated hidden, cell
            output_pred: (num_batch x loc_dim), predicted location
        '''
        num_batch = location.shape[0]

        # Embed feature to hidden_size
        loc_embed = self.loc2feat(location).view(num_batch, self.feature_dim)
        obs_embed = self.loc2feat(observation).view(num_batch,
                                                    self.feature_dim)
        embed = torch.cat([loc_embed, obs_embed],
                          dim=1).view(1, num_batch, 2 * self.feature_dim)

        out, (h_n, c_n) = self.refine_lstm(embed, hc_0)

        # Merge embed feature with output
        # merge_feat = h_n + embed
        merge_feat = out

        output_pred = self.pred2vel(merge_feat).view(
            num_batch, self.loc_dim) + observation

        return output_pred, (h_n, c_n)

    def _init_param(self):
        init_module(self.loc2feat)
        init_module(self.vel2feat)
        init_module(self.pred2vel)
        init_lstm_module(self.pred_lstm)
        init_lstm_module(self.refine_lstm)


class ConfLSTM(LSTM):
    '''
    Estimating object location in world coordinates
    Prediction LSTM:
        Input: 5 frames velocity
        Output: Next frame location
    Updating LSTM:
        Input: predicted location and observed location
        Output: Refined location
    '''

    def __init__(self,
                 batch_size: int,
                 feature_dim: int,
                 hidden_size: int,
                 num_layers: int,
                 loc_dim: int,
                 dropout: float = 0.0):
        super(ConfLSTM, self).__init__(batch_size, feature_dim, hidden_size,
                                       num_layers, loc_dim, dropout)
        self.refine_lstm = nn.LSTM(
            input_size=3 * feature_dim,
            hidden_size=hidden_size,
            dropout=dropout,
            num_layers=num_layers,
        )

        self._init_param()

    def refine(self, location, observation, confidence, hc_0):
        '''
        Refine predicted location using single frame estimation at t+1
        Input:
            location: (num_batch x 3), location from prediction
            observation: (num_batch x 3), location from single frame estimation
            confidence: (num_batch X 1), depth estimation confidence
            hc_0: (num_layers, num_batch, hidden_size), tuple of hidden and cell
        Middle:
            loc_embed: (1, num_batch x feature_dim), predicted location feature
            obs_embed: (1, num_batch x feature_dim), single frame location feature
            embed: (1, num_batch x 2*feature_dim), location feature
            out: (1 x num_batch x hidden_size), lstm output
            merge_feat: same as out
        Output:
            hc_n: (num_layers, num_batch, hidden_size), tuple of updated hidden, cell
            output_pred: (num_batch x loc_dim), predicted location
        '''
        num_batch = location.shape[0]

        # Embed feature to hidden_size
        loc_embed = self.loc2feat(location).view(num_batch, self.feature_dim)
        obs_embed = self.loc2feat(observation).view(num_batch,
                                                    self.feature_dim)
        embed = torch.cat([
            loc_embed, obs_embed,
            (1.0 - confidence) * loc_embed + confidence * obs_embed
        ],
                          dim=1).view(1, num_batch, 3 * self.feature_dim)

        out, (h_n, c_n) = self.refine_lstm(embed, hc_0)

        # Merge embed feature with output
        # merge_feat = h_n + embed
        merge_feat = out

        output_pred = self.pred2vel(merge_feat).view(
            num_batch, self.loc_dim) + observation

        return output_pred, (h_n, c_n)


class LocLSTM(LSTM):
    '''
    Estimating object location in world coordinates
    Prediction LSTM:
        Input: 5 frames velocity
        Output: Next frame location
    Updating LSTM:
        Input: predicted location and observed location
        Output: Refined location
    '''

    def __init__(self,
                 batch_size: int,
                 feature_dim: int,
                 hidden_size: int,
                 num_layers: int,
                 loc_dim: int,
                 dropout: float = 0.0):
        super(LocLSTM, self).__init__(batch_size, feature_dim, hidden_size,
                                      num_layers, loc_dim, dropout)

        self.pred2atten = nn.Linear(
            hidden_size,
            loc_dim,
            bias=False,
        )

        init_module(self.pred2atten)

    def predict(self, vel_history, location, hc_0):
        '''
        Predict location at t+1 using updated location at t
        Input:
            vel_history: (num_seq, num_batch, loc_dim), velocity from previous num_seq updates
            location: (num_batch, loc_dim), location from previous update
            hc_0: (num_layers, num_batch, hidden_size), tuple of hidden and cell
        Middle:
            embed: (num_seq, num_batch x feature_dim), location feature
            out: (num_seq x num_batch x hidden_size), lstm output
            merge_feat: (num_batch x hidden_size), the predicted residual
        Output:
            hc_n: (num_layers, num_batch, hidden_size), tuple of updated hidden, cell
            output_pred: (num_batch x loc_dim), predicted location
        '''
        num_seq, num_batch, _ = vel_history.shape

        # Embed feature to hidden_size
        embed = self.vel2feat(vel_history).view(num_seq, num_batch,
                                                self.feature_dim)

        out, (h_n, c_n) = self.pred_lstm(embed, hc_0)

        attention_logit = self.pred2atten(out).view(num_seq, num_batch,
                                                    self.loc_dim)
        attention = torch.softmax(attention_logit, dim=0)

        output_pred = torch.sum(attention * vel_history, dim=0) + location

        return output_pred, (h_n, c_n)


class VeloLSTM(LSTM):
    '''
    Estimating object location in world coordinates
    Prediction LSTM:
        Input: 5 frames velocity
        Output: Next frame location
    Updating LSTM:
        Input: predicted location and observed location
        Output: Refined location
    '''

    def __init__(self,
                 batch_size: int,
                 feature_dim: int,
                 hidden_size: int,
                 num_layers: int,
                 loc_dim: int,
                 dropout: float = 0.0):
        super(VeloLSTM, self).__init__(batch_size, feature_dim, hidden_size,
                                       num_layers, loc_dim, dropout)
        self.refine_lstm = nn.LSTM(
            input_size=3 * feature_dim,
            hidden_size=hidden_size,
            dropout=dropout,
            num_layers=num_layers,
        )

        self._init_param()

        self.pred2atten = nn.Linear(
            hidden_size,
            loc_dim,
            bias=False,
        )
        self.conf2atten = nn.Linear(
            hidden_size,
            loc_dim,
            bias=False,
        )
        self.conf2feat = nn.Linear(
            1,
            feature_dim,
            bias=False,
        )
        init_module(self.pred2atten)
        init_module(self.conf2feat)

    def refine(
        self, location: torch.Tensor, observation: torch.Tensor,
        prev_location: torch.Tensor, confidence: torch.Tensor,
        hc_0: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        '''
        Refine predicted location using single frame estimation at t+1
        Input:
            location: (num_batch x loc_dim), location from prediction
            observation: (num_batch x loc_dim), location from single frame estimation
            prev_location: (num_batch x loc_dim), refined location
            confidence: (num_batch X 1), depth estimation confidence
            hc_0: (num_layers, num_batch, hidden_size), tuple of hidden and cell
        Middle:
            loc_embed: (1, num_batch x feature_dim), predicted location feature
            obs_embed: (1, num_batch x feature_dim), single frame location feature
            conf_embed: (1, num_batch x feature_dim), depth estimation confidence feature
            embed: (1, num_batch x 2*feature_dim), location feature
            out: (1 x num_batch x hidden_size), lstm output
        Output:
            hc_n: (num_layers, num_batch, hidden_size), tuple of updated hidden, cell
            output_pred: (num_batch x loc_dim), predicted location
        '''
        num_batch = location.shape[0]

        pred_vel = location - prev_location
        obsv_vel = observation - prev_location

        # Embed feature to hidden_size
        loc_embed = self.vel2feat(pred_vel).view(num_batch, self.feature_dim)
        obs_embed = self.vel2feat(obsv_vel).view(num_batch, self.feature_dim)
        conf_embed = self.conf2feat(confidence).view(num_batch,
                                                     self.feature_dim)
        embed = torch.cat([
            loc_embed,
            obs_embed,
            conf_embed,
        ], dim=1).view(1, num_batch, 3 * self.feature_dim)

        out, (h_n, c_n) = self.refine_lstm(embed, hc_0)

        delta_vel_atten = torch.sigmoid(self.conf2atten(out)).view(
            num_batch, self.loc_dim)

        output_pred = delta_vel_atten * obsv_vel + (
            1.0 - delta_vel_atten) * pred_vel + prev_location

        return output_pred, (h_n, c_n)

    def predict(
        self, vel_history: torch.Tensor, location: torch.Tensor,
        hc_0: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        '''
        Predict location at t+1 using updated location at t
        Input:
            vel_history: (num_seq, num_batch, loc_dim), velocity from previous num_seq updates
            location: (num_batch, loc_dim), location from previous update
            hc_0: (num_layers, num_batch, hidden_size), tuple of hidden and cell
        Middle:
            embed: (num_seq, num_batch x feature_dim), location feature
            out: (num_seq x num_batch x hidden_size), lstm output
            attention_logit: (num_seq x num_batch x loc_dim), the predicted residual
        Output:
            hc_n: (num_layers, num_batch, hidden_size), tuple of updated hidden, cell
            output_pred: (num_batch x loc_dim), predicted location
        '''
        num_seq, num_batch, _ = vel_history.shape

        # Embed feature to hidden_size
        embed = self.vel2feat(vel_history).view(num_seq, num_batch,
                                                self.feature_dim)

        out, (h_n, c_n) = self.pred_lstm(embed, hc_0)

        attention_logit = self.pred2atten(out).view(num_seq, num_batch,
                                                    self.loc_dim)
        attention = torch.softmax(attention_logit, dim=0)

        output_pred = torch.sum(attention * vel_history, dim=0) + location

        return output_pred, (h_n, c_n)


class LocConfLSTM(LSTM):
    '''
    Estimating object location in world coordinates
    Prediction LSTM:
        Input: 5 frames velocity
        Output: Next frame location
    Updating LSTM:
        Input: predicted location and observed location
        Output: Refined location
    '''

    def __init__(self,
                 batch_size: int,
                 feature_dim: int,
                 hidden_size: int,
                 num_layers: int,
                 loc_dim: int,
                 dropout: float = 0.0):
        super(LocConfLSTM, self).__init__(batch_size, feature_dim, hidden_size,
                                          num_layers, loc_dim, dropout)
        self.refine_lstm = nn.LSTM(
            input_size=3 * feature_dim,
            hidden_size=hidden_size,
            dropout=dropout,
            num_layers=num_layers,
        )

        self._init_param()

        self.pred2atten = nn.Linear(
            hidden_size,
            loc_dim,
            bias=False,
        )
        self.conf2feat = nn.Linear(
            1,
            feature_dim,
            bias=False,
        )
        init_module(self.pred2atten)
        init_module(self.conf2feat)

    def refine(self, location, observation, confidence, hc_0):
        '''
        Refine predicted location using single frame estimation at t+1
        Input:
            location: (num_batch x 3), location from prediction
            observation: (num_batch x 3), location from single frame estimation
            confidence: (num_batch X 1), depth estimation confidence
            hc_0: (num_layers, num_batch, hidden_size), tuple of hidden and cell
        Middle:
            loc_embed: (1, num_batch x feature_dim), predicted location feature
            obs_embed: (1, num_batch x feature_dim), single frame location feature
            embed: (1, num_batch x 2*feature_dim), location feature
            out: (1 x num_batch x hidden_size), lstm output
            merge_feat: same as out
        Output:
            hc_n: (num_layers, num_batch, hidden_size), tuple of updated hidden, cell
            output_pred: (num_batch x loc_dim), predicted location
        '''
        num_batch = location.shape[0]

        # Embed feature to hidden_size
        loc_embed = self.loc2feat(location).view(num_batch, self.feature_dim)
        obs_embed = self.loc2feat(observation).view(num_batch,
                                                    self.feature_dim)
        conf_embed = self.conf2feat(confidence).view(num_batch,
                                                     self.feature_dim)
        embed = torch.cat([
            loc_embed,
            obs_embed,
            conf_embed,
        ], dim=1).view(1, num_batch, 3 * self.feature_dim)

        out, (h_n, c_n) = self.refine_lstm(embed, hc_0)

        delta_vel = self.pred2vel(out).view(num_batch, self.loc_dim)

        output_pred = delta_vel + observation

        return output_pred, (h_n, c_n)

    def predict(self, vel_history, location, hc_0):
        '''
        Predict location at t+1 using updated location at t
        Input:
            vel_history: (num_seq, num_batch, loc_dim), velocity from previous num_seq updates
            location: (num_batch, loc_dim), location from previous update
            hc_0: (num_layers, num_batch, hidden_size), tuple of hidden and cell
        Middle:
            embed: (num_seq, num_batch x feature_dim), location feature
            out: (num_seq x num_batch x hidden_size), lstm output
            merge_feat: (num_batch x hidden_size), the predicted residual
        Output:
            hc_n: (num_layers, num_batch, hidden_size), tuple of updated hidden, cell
            output_pred: (num_batch x loc_dim), predicted location
        '''
        num_seq, num_batch, _ = vel_history.shape

        # Embed feature to hidden_size
        embed = self.vel2feat(vel_history).view(num_seq, num_batch,
                                                self.feature_dim)

        out, (h_n, c_n) = self.pred_lstm(embed, hc_0)

        attention_logit = self.pred2atten(out).view(num_seq, num_batch,
                                                    self.loc_dim)
        attention = torch.softmax(attention_logit, dim=0)

        output_pred = torch.sum(attention * vel_history, dim=0) + location

        return output_pred, (h_n, c_n)


class LocTrajLSTM(LSTM):
    '''
    Estimating object location in world coordinates
    Prediction LSTM:
        Input: 5 frames velocity
        Output: Next frame location
    Updating LSTM:
        Input: predicted location and observed location
        Output: Refined location
    '''

    def __init__(self,
                 batch_size: int,
                 feature_dim: int,
                 hidden_size: int,
                 num_layers: int,
                 loc_dim: int,
                 dropout: float = 0.0):
        super(LocTrajLSTM, self).__init__(batch_size, feature_dim, hidden_size,
                                          num_layers, loc_dim, dropout)
        self.refine_lstm = nn.LSTM(
            input_size=3 * feature_dim,
            hidden_size=hidden_size,
            dropout=dropout,
            num_layers=num_layers,
        )

        self.pred_lstm = nn.LSTM(
            input_size=3 * feature_dim,
            hidden_size=hidden_size,
            dropout=dropout,
            num_layers=num_layers,
        )

        self.pred2atten = nn.Linear(
            hidden_size,
            loc_dim,
            bias=False,
        )

        init_module(self.pred2atten)
        self._init_param()

    def refine(self, location, observation, confidence, hc_0):
        '''
        Refine predicted location using single frame estimation at t+1
        Input:
            location: (num_batch x 3), location from prediction
            observation: (num_batch x 3), location from single frame estimation
            confidence: (num_batch X 1), depth estimation confidence
            hc_0: (num_layers, num_batch, hidden_size), tuple of hidden and cell
        Middle:
            loc_embed: (1, num_batch x feature_dim), predicted location feature
            obs_embed: (1, num_batch x feature_dim), single frame location feature
            embed: (1, num_batch x 2*feature_dim), location feature
            out: (1 x num_batch x hidden_size), lstm output
            merge_feat: same as out
        Output:
            hc_n: (num_layers, num_batch, hidden_size), tuple of updated hidden, cell
            output_pred: (num_batch x loc_dim), predicted location
        '''
        num_batch = location.shape[0]

        # Embed feature to hidden_size
        loc_embed = self.loc2feat(location).view(num_batch, self.feature_dim)
        obs_embed = self.loc2feat(observation).view(num_batch,
                                                    self.feature_dim)
        embed = torch.cat([
            loc_embed, obs_embed,
            (1.0 - confidence) * loc_embed + confidence * obs_embed
        ],
                          dim=1).view(1, num_batch, 3 * self.feature_dim)

        out, (h_n, c_n) = self.refine_lstm(embed, hc_0)

        # Merge embed feature with output
        # merge_feat = h_n + embed
        merge_feat = out

        output_pred = self.pred2vel(merge_feat).view(
            num_batch, self.loc_dim) + observation

        return output_pred, (h_n, c_n)

    def predict(self, vel_history, refine_loc, pred_loc, hc_0):
        '''
        Predict location at t+1 using updated location at t
        Input:
            vel_history: (num_seq, num_batch, loc_dim), location from previous update
            location: (num_batch, loc_dim), location from previous update
            hc_0: (num_layers, num_batch, hidden_size), tuple of hidden and cell
        Middle:
            embed: (num_seq, num_batch x feature_dim), location feature
            out: (num_seq x num_batch x hidden_size), lstm output
            merge_feat: (num_batch x hidden_size), the predicted residual
        Output:
            hc_n: (num_layers, num_batch, hidden_size), tuple of updated hidden, cell
            output_pred: (num_batch x loc_dim), predicted location
        '''
        num_seq, num_batch, _ = vel_history.shape

        # Embed feature to hidden_size
        embed = self.vel2feat(vel_history).view(num_seq, num_batch,
                                                self.feature_dim)
        pred_embed = self.loc2feat(pred_loc).view(num_seq, num_batch,
                                                  self.feature_dim)
        refine_embed = self.loc2feat(refine_loc).view(num_seq, num_batch,
                                                      self.feature_dim)
        embed = torch.cat([
            embed,
            pred_embed,
            refine_embed,
        ], dim=2).view(num_seq, num_batch, 3 * self.feature_dim)

        out, (h_n, c_n) = self.pred_lstm(embed, hc_0)

        attention_logit = self.pred2atten(out).view(num_seq, num_batch,
                                                    self.loc_dim)
        attention = torch.softmax(attention_logit, dim=0)

        output_pred = torch.sum(
            attention * vel_history, dim=0) + refine_loc[-1]

        return output_pred, (h_n, c_n)


LSTM_MODEL_ZOO = {
    'LSTM': LSTM,
    'ConfLSTM': ConfLSTM,
    'LocLSTM': LocLSTM,
    'VeloLSTM': VeloLSTM,
    'LocConfLSTM': LocConfLSTM,
    'LocTrajLSTM': LocTrajLSTM
}


def get_lstm_model(lstm_model_name: str) -> nn.Module:
    lstm_model = LSTM_MODEL_ZOO.get(lstm_model_name, None)
    if lstm_model is None:
        raise NotImplementedError

    return lstm_model