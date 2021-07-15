import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as _Variable
import math
import numpy as np
import logging

from torchinfo import summary

from misc import conv_output_shape, reset_parameters_util_x, reset_parameters_util_h

FORMAT = '[%(asctime)s %(levelname)s] %(message)s'
logging.basicConfig(format=FORMAT)
debuglogger = logging.getLogger('main_logger')
debuglogger.setLevel('DEBUG')


def squash(input_tensor, dim=-1, epsilon=1e-7):
    squared_norm = (input_tensor ** 2).sum(dim=dim, keepdim=True)
    safe_norm = torch.sqrt(squared_norm + epsilon)
    scale = squared_norm / (1 + squared_norm)
    unit_vector = input_tensor / safe_norm
    return scale * unit_vector


class CapsPrimaryLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, cap_dim, num_cap_map):
        super(CapsPrimaryLayer, self).__init__()

        self.capsule_dim = cap_dim
        self.num_cap_map = num_cap_map
        self.conv_out = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0)

    def forward(self, x):
        debuglogger.debug(f'Now in CapsPrimaryLayer')
        batch_size = x.size(0)
        outputs = self.conv_out(x)
        map_dim = outputs.size(-1)
        outputs = outputs.view(batch_size, self.capsule_dim, self.num_cap_map, map_dim, map_dim)
        # [bs, 8 (or 10), 32, 6, 6]
        outputs = outputs.view(batch_size, self.capsule_dim, self.num_cap_map, -1).transpose(1, 2).transpose(2, 3)
        # [bs, 32, 36, 8]
        outputs = squash(outputs)
        return outputs


class CapsShapeLayer(nn.Module):
    def __init__(self, num_digit_cap, num_prim_cap, num_prim_map, in_cap_dim, out_cap_dim, num_iterations, use_cuda):
        super(CapsShapeLayer, self).__init__()
        self.num_prim_cap = num_prim_cap
        self.num_prim_map = num_prim_map
        self.num_digit_cap = num_digit_cap
        self.num_iterations = num_iterations
        self.out_cap_dim = out_cap_dim
        self.use_cuda = use_cuda

        self.W = nn.Parameter(0.01 * torch.randn(1, num_digit_cap, num_prim_map, 1, out_cap_dim, in_cap_dim))

    def forward(self, x):
        batch_size = x.size(0)  # [bs, 32, 36, 8]
        u = x[:, None, :, :, :, None]  # [bs, 1, 32, 36, 8, 1]
        u_hat = torch.matmul(self.W, u).squeeze(-1)  # [bs, 10, 32, 36, 16]

        # detach u_hat during routing iterations to prevent gradients from flowing
        temp_u_hat = u_hat.detach()

        b = torch.zeros(batch_size, self.num_digit_cap, u_hat.size(2), u_hat.size(3), 1)
        if self.use_cuda:
            b = b.cuda()
        # [bs, 10, 32, 36, 1]
        for i in range(self.num_iterations - 1):
            c = F.softmax(b, dim=1)
            s = (c * temp_u_hat).sum(dim=2).sum(dim=2)  # [bs, 10, 16]
            v = squash(s)
            # [bs, 10, 1152, 16] . [batch_size, 10, 16, 1]
            uv = torch.matmul(temp_u_hat.view(batch_size, self.num_digit_cap, -1, self.out_cap_dim),
                              v.unsqueeze(-1))  # [batch_size, 10, 1152, 1]
            b += uv.view(b.shape)

        c = F.softmax(b, dim=3)
        s = (c * u_hat).sum(dim=2).sum(dim=2)
        v = squash(s)
        return v


class ImageProcessor(nn.Module):
    def __init__(self, img_c=3, f_conv1=256, k_conv1=9, s_conv1=1, f_prim=256, primary_cap_dim=8,
                 k_prim=9, s_prim=2, img_h=128, shape_cap_dim=16, num_iterations=3, cuda=False, hid_dim=256):
        super(ImageProcessor, self).__init__()

        # convolution layer
        self.conv1 = nn.Conv2d(in_channels=img_c, out_channels=f_conv1,
                               kernel_size=(k_conv1, k_conv1), stride=(s_conv1, s_conv1))

        # primary capsule layer
        assert f_prim % primary_cap_dim == 0
        self.num_prim_map = int(f_prim / primary_cap_dim)
        self.primary_capsules = CapsPrimaryLayer(in_channels=f_conv1, out_channels=f_prim,
                                                 kernel_size=k_prim, stride=s_prim,
                                                 cap_dim=primary_cap_dim,
                                                 num_cap_map=self.num_prim_map)
        num_prim_cap = int((img_h - 2 * (k_prim - 1)) * (img_h - 2 * (k_prim - 1))
                           / (s_prim * s_prim))

        self.digit_capsules = CapsShapeLayer(num_digit_cap=num_classes,
                                             num_prim_cap=num_prim_cap,
                                             num_prim_map=self.num_prim_map,
                                             in_cap_dim=primary_cap_dim,
                                             out_cap_dim=shape_cap_dim,
                                             num_iterations=num_iterations,
                                             use_cuda=cuda)

        self.im_transform = nn.Linear(img_c * shape_cap_dim, hid_dim)

    def forward(self, imgs):
        x = F.relu(self.conv1(imgs), inplace=True)
        x = self.primary_capsules(x)
        x = self.digit_capsules(x)
        x = x.view(x.shape[0], -1)
        x = self.im_transform(x)
        return x

    def reset_parameters(self):
        reset_parameters_util_h(self)


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
        debuglogger.debug(f'Now in Message Processor')
        debuglogger.debug(f'm shape {m.shape}')
        debuglogger.debug(f'h shape {h.shape}')
        debuglogger.debug(f'use_message {use_message}')
        if self.use_cuda:
            h = h.cuda()
        if use_message:
            debuglogger.debug(f'Using message')
            if self.use_cuda:
                m = m.cuda()
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
        h_w = torch.tanh(self.w_h(h_c) + self.w_d(desc))
        w_scores = self.w(h_w)
        if self.use_binary:
            w_probs = torch.sigmoid(w_scores)
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
                 im_dim,
                 h_dim,
                 m_dim,
                 desc_dim,
                 num_classes,
                 s_dim,
                 use_binary,
                 use_mlp,
                 cuda,
                 pcd,
                 scd
                 ):
        super(Agent, self).__init__()
        self.im_feat_dim = im_dim
        self.h_dim = h_dim
        self.m_dim = m_dim
        self.desc_dim = desc_dim
        self.num_classes = num_classes
        self.s_dim = s_dim
        self.use_binary = use_binary
        self.use_MLP = use_mlp
        self.use_cuda = cuda
        self.image_processor = ImageProcessor(img_h=im_dim, cuda=cuda, hid_dim=h_dim, primary_cap_dim=pcd,
                                              shape_cap_dim=scd)
        self.text_processor = TextProcessor(desc_dim=desc_dim, hid_dim=h_dim)
        self.message_processor = MessageProcessor(m_dim=m_dim, hid_dim=h_dim, cuda=cuda)
        self.message_generator = MessageGenerator(m_dim=m_dim, hid_dim=h_dim, use_binary=use_binary)
        self.reward_estimator = RewardEstimator(hid_dim=h_dim)
        # Network for combining processed image and message representations
        self.text_im_combine = nn.Linear(self.h_dim * 2, self.h_dim)
        # Network for making predictions
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

    def forward(self, x, m, desc, use_message, batch_size, training):
        """
        Update State:
            h_z = message_processor(m, h_z)

        Image processing
            h_i = image_processor(x, h_z)

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
        h_i = self.image_processor(x)
        debuglogger.debug(f'h_i: {h_i.size()}')

        # Combine the image and message info to a single vector
        single_vector = torch.cat([self.h_z, h_i], dim=1)
        h_c = self.text_im_combine(single_vector)
        debuglogger.debug(f'h_c: {h_c.size()}')

        # Process the texts
        # desc: bs x num_classes x desc_dim
        # desc_proc:    bs x num_classes x hid_dim
        desc_proc = self.text_processor(desc)
        debuglogger.debug(f'desc_proc: {desc_proc.size()}')

        # Estimate the reward
        reward_estimator = self.reward_estimator(h_c)
        debuglogger.debug(f'r: {reward_estimator.size()}')

        # Calculate stop bits
        s_score = self.s(h_c)
        s_prob = torch.sigmoid(s_score)
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
        class_prediction = self.predict_classes(h_c, desc_proc, batch_size)
        y_scores = F.softmax(class_prediction, dim=1).detach()
        debuglogger.debug(f'y_scores: {y_scores.size()}')
        # debuglogger.debug(f'y_scores: {y_scores}')

        # Generate message
        w, w_probs = self.message_generator(y_scores, h_c, desc_proc, training)
        debuglogger.debug(f'w: {w.size()}')
        debuglogger.debug(f'w_probs: {w_probs.size()}')

        return (s_binary, s_prob), (w, w_probs), class_prediction, reward_estimator


if __name__ == "__main__":
    print("Testing agent init and forward pass...")
    im_dim = 64
    h_dim = 256
    m_dim = 6
    pcd = 8
    scd = 16
    desc_dim = 100
    num_classes = 3
    s_dim = 1
    use_binary = True
    use_message = True
    batch_size = 32
    training = True
    use_MLP = False
    cuda = True
    agent = Agent(im_dim,
                  h_dim,
                  m_dim,
                  desc_dim,
                  num_classes,
                  s_dim,
                  use_binary,
                  use_MLP,
                  cuda,
                  pcd,
                  scd
                  )

    x = _Variable(torch.ones(batch_size, 3, im_dim, im_dim))
    m = _Variable(torch.ones(batch_size, m_dim))
    desc = _Variable(torch.ones(batch_size, num_classes, desc_dim))

    summary(agent, input_data=(x, m, desc, use_message, batch_size, training))

    for i in range(2):
        if cuda:
            x = x.cuda()
            m = m.cuda()
            desc = desc.cuda()
        s, w, y, r = agent(x, m, desc, use_message, batch_size, training)
        # print(f's_binary: {s[0]}')
        # print(f's_probs: {s[1]}')
        # print(f'w_binary: {w[0]}')
        # print(f'w_probs: {w[1]}')
        # print(f's_binary: {s[0]}')
        # print(f's_probs: {s[1]}')
        # print(f'y: {y}')
        # print(f'r: {r}')
