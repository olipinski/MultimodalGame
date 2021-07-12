import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as _Variable
import torch.optim as optim
from torch.nn.parameter import Parameter
import math
import numpy as np
import logging
import functools

from torchinfo import summary


FORMAT = '[%(asctime)s %(levelname)s] %(message)s'
logging.basicConfig(format=FORMAT)
debuglogger = logging.getLogger('main_logger')
debuglogger.setLevel('INFO')


def reset_parameters_util_x(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data, 1)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.GRUCell):
            for mm in m.parameters():
                if mm.data.ndimension() == 2:
                    nn.init.xavier_normal_(mm.data, 1)
                elif mm.data.ndimension() == 1:  # Bias
                    mm.data.zero_()


def reset_parameters_util_h(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.nn.init.kaiming_normal_(m.weight.data, 1)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.GRUCell):
            for mm in m.parameters():
                if mm.data.ndimension() == 2:
                    nn.init.nn.init.kaiming_normal_(mm.data, 1)
                elif mm.data.ndimension() == 1:  # Bias
                    mm.data.zero_()


class CapsConvLayer(nn.Module):
    def __init__(self, in_channels=3, out_channels=256, kernel_size=9):
        super(CapsConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=1
                              )

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        return x


class CapsPrimaryLayer(nn.Module):
    def __init__(self, num_capsules=8, in_channels=256, out_channels=32, kernel_size=9):
        super(CapsPrimaryLayer, self).__init__()
        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=2, padding=0)
            for _ in range(num_capsules)])

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor

    def forward(self, x):
        u = [capsule(x) for capsule in self.capsules]
        u = torch.stack(u, dim=1)
        u = u.view(x.size(0), 32 * 6 * 6, -1)
        return self.squash(u)


class CapsShapeLayer(nn.Module):
    def __init__(self, num_capsules=10, num_routes=32 * 6 * 6, in_channels=8, out_channels=16, cuda=False):
        super(CapsShapeLayer, self).__init__()
        self.in_channels = in_channels
        self.num_routes = num_routes
        self.num_capsules = num_capsules
        self.use_cuda = cuda

        self.W = nn.Parameter(torch.randn(1, num_routes, num_capsules, out_channels, in_channels))

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor

    def forward(self, x):
        batch_size = x.size(0)
        x = torch.stack([x] * self.num_capsules, dim=2).unsqueeze(4)

        W = torch.cat([self.W] * batch_size, dim=0)
        u_hat = torch.matmul(W, x)

        b_ij = _Variable(torch.zeros(1, self.num_routes, self.num_capsules, 1))
        if self.use_cuda:
            b_ij = b_ij.cuda()

        num_iterations = 3
        for iteration in range(num_iterations):
            c_ij = F.softmax(b_ij)
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)

            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            v_j = self.squash(s_j)

            if iteration < num_iterations - 1:
                a_ij = torch.matmul(u_hat.transpose(3, 4), torch.cat([v_j] * self.num_routes, dim=1))
                b_ij = b_ij + a_ij.squeeze(4).mean(dim=0, keepdim=True)

        return v_j.squeeze(1)


class ImageProcessor(nn.Module):
    """Processes an agent's image, with or without attention"""

    def __init__(self, num_capsules_l1, num_capsules_l2, cuda):
        super(ImageProcessor, self).__init__()
        self.use_cuda = cuda
        self.reset_parameters()
        self.capsConvLayer = CapsConvLayer()
        self.capsPrimaryLayer = CapsPrimaryLayer(num_capsules=num_capsules_l1)
        self.capsShapeLayer = CapsShapeLayer(num_capsules=num_capsules_l2, cuda=True)

    def reset_parameters(self):
        reset_parameters_util_h(self)

    def forward(self, x):
        debuglogger.debug(f'Inside image processing...')
        x = self.capsConvLayer(x)
        x = self.capsPrimaryLayer(x)
        x = self.capsShapeLayer(x)
        return x


class TextProcessor(nn.Module):
    """Processes sentence representations to the correct hidden dimension"""

    def __init__(self, desc_dim, hid_dim):
        super(TextProcessor, self).__init__()
        self.desc_dim = desc_dim
        self.hid_dim = hid_dim
        self.transform = nn.Linear(desc_dim, hid_dim)
        self.reset_parameters()

    def reset_parameters(self):
        reset_parameters_util_x(self)

    def forward(self, desc):
        bs, num_classes, desc_dim = desc.size()
        desc = desc.view(-1, desc_dim)
        out = self.transform(desc)
        out = out.view(bs, num_classes, -1)
        return F.relu(out)


class MessageProcessor(nn.Module):
    """Processes a received message from an agent"""

    def __init__(self, m_dim, hid_dim, cuda):
        super(MessageProcessor, self).__init__()
        self.m_dim = m_dim
        self.hid_dim = hid_dim
        self.use_cuda = cuda
        self.rnn = nn.GRUCell(self.m_dim, self.hid_dim)
        self.reset_parameters()

    def reset_parameters(self):
        reset_parameters_util_x(self)

    def forward(self, m, h, use_message):
        if use_message:
            debuglogger.debug(f'Using message')
            return self.rnn(m, h)
        else:
            debuglogger.debug(f'Ignoring message, using blank instead...')
            blank_msg = _Variable(torch.zeros_like(m.data))
            if self.use_cuda:
                blank_msg = blank_msg.cuda()
            return self.rnn(blank_msg, h)


class MessageGenerator(nn.Module):
    """Generates a message for an agent
    TODO MAKE RECURRENT? - later"""

    def __init__(self, m_dim, hid_dim, use_binary):
        super(MessageGenerator, self).__init__()
        self.m_dim = m_dim
        self.hid_dim = hid_dim
        self.use_binary = use_binary
        # Why different biases?
        self.w_h = nn.Linear(self.hid_dim, self.hid_dim, bias=True)
        self.w_d = nn.Linear(self.hid_dim, self.hid_dim, bias=False)
        self.w = nn.Linear(self.hid_dim, self.m_dim)
        self.reset_parameters()

    def reset_parameters(self):
        reset_parameters_util_x(self)

    def forward(self, y_scores, h_c, desc, training):
        """
            desc = \sum_i y_scores desc_i
            w_hat = tanh(W_h h_c + W_d desc)
            w = bernoulli(sig(w_hat)) or round(sig(w_hat))
        """
        # y_scores: batch_size x num_classes
        # desc: batch_size x num_classes x hid_dim
        # h_c: batch_size x hid_dim
        batch_size, num_classes = y_scores.size()
        y_broadcast = y_scores.unsqueeze(2).expand(
            batch_size, num_classes, self.hid_dim)
        debuglogger.debug(f'y_broadcast: {y_broadcast.size()}')
        # debuglogger.debug(f'y_broadcast: {y_broadcast}')
        debuglogger.debug(f'desc: {desc.size()}')
        # Weight descriptions based on current predictions
        desc = torch.mul(y_broadcast, desc).sum(1).squeeze(1)
        debuglogger.debug(f'desc: {desc.size()}')
        # desc: batch_size x hid_dim
        h_w = F.tanh(self.w_h(h_c) + self.w_d(desc))
        w_scores = self.w(h_w)
        if self.use_binary:
            w_probs = F.sigmoid(w_scores)
            if training:
                # debuglogger.info(f"Training...")
                probs_ = w_probs.data.cpu().numpy()
                rand_num = np.random.rand(*probs_.shape)
                # debuglogger.debug(f'rand_num: {rand_num}')
                # debuglogger.info(f'probs: {probs_}')
                w_binary = _Variable(torch.from_numpy(
                    (rand_num < probs_).astype('float32')))
            else:
                # debuglogger.info(f"Eval mode, rounding...")
                w_binary = torch.round(w_probs).detach()
            if w_probs.is_cuda:
                w_binary = w_binary.cuda()
            w_feats = w_binary
            # debuglogger.debug(f'w_binary: {w_binary}')
        else:
            w_feats = w_scores
            w_probs = None
        # debuglogger.info(f'Message : {w_feats}')
        return w_feats, w_probs


class RewardEstimator(nn.Module):
    """Estimates the reward the agent will receive. Value used as a baseline in REINFORCE loss"""

    def __init__(self, hid_dim):
        super(RewardEstimator, self).__init__()
        self.hid_dim = hid_dim
        self.v1 = nn.Linear(hid_dim, math.ceil(hid_dim / 2))
        self.v2 = nn.Linear(math.ceil(hid_dim / 2), 1)
        self.reset_parameters()

    def reset_parameters(self):
        reset_parameters_util_x(self)

    def forward(self, x):
        # Detach input from rest of graph - only want gradients to flow through the RewardEstimator and no further
        x = x.detach()
        x = F.relu(self.v1(x))
        x = self.v2(x)
        return x


class Agent(nn.Module):
    def __init__(self,
                 im_feature_type,
                 im_feat_dim,
                 h_dim,
                 m_dim,
                 desc_dim,
                 num_classes,
                 s_dim,
                 use_binary,
                 use_MLP,
                 cuda,
                 num_capsules_l1,
                 num_capsules_l2
                 ):
        super(Agent, self).__init__()
        self.im_feature_type = im_feature_type
        self.im_feat_dim = im_feat_dim
        self.h_dim = h_dim
        self.m_dim = m_dim
        self.desc_dim = desc_dim
        self.num_classes = num_classes
        self.s_dim = s_dim
        self.use_binary = use_binary
        self.use_MLP = use_MLP
        self.use_cuda = cuda
        self.num_capsules_l1 = num_capsules_l1
        self.num_capsules_l2 = num_capsules_l2
        self.image_processor = ImageProcessor(num_capsules_l1, num_capsules_l2, cuda)
        self.text_processor = TextProcessor(desc_dim, h_dim)
        self.message_processor = MessageProcessor(m_dim, h_dim, cuda)
        self.message_generator = MessageGenerator(m_dim, h_dim, use_binary)
        self.reward_estimator = RewardEstimator(h_dim)
        # Network for combining processed image and message representations
        self.text_im_combine = nn.Linear(h_dim * 2, h_dim)
        # Network for making predicitons
        self.y1 = nn.Linear(self.h_dim * 2, self.h_dim)
        self.y2 = nn.Linear(self.h_dim, 1)
        # Network for making stop decision decisions
        self.s = nn.Linear(self.h_dim, self.s_dim)
        self.h_z = None
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, 1)
                if m.bias is not None:
                    m.bias.data.zero_()
        self.image_processor.reset_parameters()
        self.message_processor.reset_parameters()
        self.text_processor.reset_parameters()
        self.message_generator.reset_parameters()
        self.reward_estimator.reset_parameters()

    def reset_state(self):
        """Initialize state for the agent.

        The agent is stateful, keeping tracking of previous messages it
        has sent and received.

        """
        self.h_z = None
        self.image_processor.reset_state()

    def initial_state(self, batch_size):
        h = _Variable(torch.zeros(batch_size, self.h_dim))
        if self.use_cuda:
            h = h.cuda()
        return h

    def predict_classes(self, h_c, desc_proc, batch_size):
        """
        Scores each class using an MLP or simple dot product
        desc_proc:     bs x num_classes x hid_dim
        h_c:           bs x hid_dim
        h_c:           bs x hid_dim x 1
        h_c_expand:    bs x num_classes x hid_dim
        hid_cat_desc:  (bs x num_classes) x (hid_dim * 2)
        y:             bs x num_classes
        """
        if self.use_MLP:
            h_c_expand = torch.unsqueeze(
                h_c, dim=1).expand(-1, self.num_classes, -1)
            debuglogger.debug(f'h_c_expand: {h_c_expand.size()}')
            # debuglogger.debug(f'h_c: {h_c}')
            # debuglogger.debug(f'h_c_expand: {h_c_expand}')
            hid_cat_desc = torch.cat([h_c_expand, desc_proc], dim=2)
            debuglogger.debug(f'hid_cat_desc: {hid_cat_desc.size()}')
            hid_cat_desc = hid_cat_desc.view(-1, self.h_dim * 2)
            debuglogger.debug(f'hid_cat_desc: {hid_cat_desc.size()}')
            y = F.relu(self.y1(hid_cat_desc))
            debuglogger.debug(f'y: {y.size()}')
            y = self.y2(y).view(batch_size, -1)
        else:
            h_c_unsqueezed = h_c.unsqueeze(dim=2)
            y = torch.bmm(desc_proc, h_c_unsqueezed).squeeze(dim=2)
        debuglogger.debug(f'y: {y.size()}')
        return y

    def forward(self, x, m, t, desc, use_message, batch_size, training):
        """
        Update State:
            h_z = message_processor(m, h_z)

        Image processing
            h_i = image_processor(x, h_z)
            Image Attention (https://arxiv.org/pdf/1502.03044.pdf):
                \beta_i = U tanh(W_r h_z + W_x x_i)
                \alpha = 1 / |x|        if t == 0
                \alpha = softmax(\beta) otherwise
                x = \sum_i \alpha x_i

        Combine Image and Message information
            h_c = text_im_combine(h_z, h_i)

        Text processing
            desc_proc = text_processor(desc)

        STOP Bit:
            s_hat = W_s h_c
            s = bernoulli(sig(s_hat)) or round(sig(s_hat))

        Predictions:
            y_i = f_y(h_c, desc_proc_i)

        Generate message:
            m_out = message_generator(y, h_c, desc_proc)
            Communication:
                desc = \sum_i y_i t_i
                w_hat = tanh(W_h h_c + W_d t)
                w = bernoulli(sig(w_hat)) or round(sig(w_hat))

        Args:
            x: Image features.
            m: communication from other agent
            t: (attention) Timestep. Used to change attention equation in first iteration.
            desc: List of description vectors used in communication and predictions.
            batch_size: size of batch
            training: whether agent is training or not
        Output:
            s, s_probs: A STOP bit and its associated probability, indicating whether the agent has decided to
                make a selection. The conversation will continue until both agents have selected STOP.
            w, w_probs: A binary message and the probability of each bit in the message being ``1``.
            y: A prediction for each class described in the descriptions.
            r: An estimate of the reward the agent will receive
        """
        debuglogger.debug(f'Input sizes...')
        debuglogger.debug(f'x: {x.size()}')
        debuglogger.debug(f'm: {m.size()}')
        debuglogger.debug(f'm: {m}')
        debuglogger.debug(f'desc: {desc.size()}')

        # Initialize hidden state if necessary
        if self.h_z is None:
            self.h_z = self.initial_state(batch_size)

        # Process message sent from the other agent
        self.h_z = self.message_processor(m, self.h_z, use_message)
        debuglogger.debug(f'h_z: {self.h_z.size()}')

        # Process the image
        h_i = self.image_processor(x, self.h_z, t)
        debuglogger.debug(f'h_i: {h_i.size()}')

        # Combine the image and message info to a single vector
        h_c = self.text_im_combine(torch.cat([self.h_z, h_i], dim=1))
        debuglogger.debug(f'h_c: {h_c.size()}')

        # Process the texts
        # desc: bs x num_classes x desc_dim
        # desc_proc:    bs x num_classes x hid_dim
        desc_proc = self.text_processor(desc)
        debuglogger.debug(f'desc_proc: {desc_proc.size()}')

        # Estimate the reward
        r = self.reward_estimator(h_c)
        debuglogger.debug(f'r: {r.size()}')

        # Calculate stop bits
        s_score = self.s(h_c)
        s_prob = F.sigmoid(s_score)
        debuglogger.debug(f's_score: {s_score.size()}')
        debuglogger.debug(f's_prob: {s_prob.size()}')
        if training:
            # Sample decisions
            prob_ = s_prob.data.cpu().numpy()
            rand_num = np.random.rand(*prob_.shape)
            # debuglogger.debug(f'rand_num: {rand_num}')
            # debuglogger.debug(f'prob: {prob_}')
            s_binary = _Variable(torch.from_numpy(
                (rand_num < prob_).astype('float32')))
            if self.use_cuda:
                s_binary = s_binary.cuda()
        else:
            # Infer decisions
            s_binary = torch.round(s_prob).detach()
        debuglogger.debug(f'stop decisions: {s_binary.size()}')
        # debuglogger.debug(f'stop decisions: {s_binary}')

        # Predict classes
        # y: batch_size * num_classes
        y = self.predict_classes(h_c, desc_proc, batch_size)
        y_scores = F.softmax(y, dim=1).detach()
        debuglogger.debug(f'y_scores: {y_scores.size()}')
        # debuglogger.debug(f'y_scores: {y_scores}')

        # Generate message
        w, w_probs = self.message_generator(y_scores, h_c, desc_proc, training)
        debuglogger.debug(f'w: {w.size()}')
        debuglogger.debug(f'w_probs: {w_probs.size()}')

        return (s_binary, s_prob), (w, w_probs), y, r


if __name__ == "__main__":
    print("Testing agent init and forward pass...")
    im_feature_type = ""
    im_feat_dim = 128
    h_dim = 64
    m_dim = 6
    num_capsules_l1 = 8
    num_capsules_l2 = 10
    desc_dim = 100
    num_classes = 3
    s_dim = 1
    use_binary = True
    use_message = True
    batch_size = 8
    training = True
    dropout = 0.3
    use_MLP = False
    cuda = False
    agent = Agent(im_feature_type,
                  im_feat_dim,
                  h_dim,
                  m_dim,
                  desc_dim,
                  num_classes,
                  s_dim,
                  use_binary,
                  use_MLP,
                  cuda,
                  num_capsules_l1,
                  num_capsules_l2
                  )

    summary(agent, input_size=(batch_size, 3, 128, 128))

    x = _Variable(torch.ones(batch_size, 3, im_feat_dim, im_feat_dim))
    m = _Variable(torch.ones(batch_size, m_dim))
    desc = _Variable(torch.ones(batch_size, num_classes, desc_dim))

    for i in range(2):
        s, w, y, r = agent(x, m, i, desc, use_message, batch_size, training)
        # print(f's_binary: {s[0]}')
        # print(f's_probs: {s[1]}')
        # print(f'w_binary: {w[0]}')
        # print(f'w_probs: {w[1]}')
        # print(f's_binary: {s[0]}')
        # print(f's_probs: {s[1]}')
        # print(f'y: {y}')
        # print(f'r: {r}')
