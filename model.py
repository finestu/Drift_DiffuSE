import time
from math import ceil
import warnings
import pdb
import torch
import pytorch_lightning as pl
from torch_ema import ExponentialMovingAverage
import copy
from sgmse import sampling
from sgmse.sdes import SDERegistry
from sgmse.backbones import BackboneRegistry
from sgmse.util.inference import evaluate_model
from sgmse.util.other import pad_spec
from sgmse.util.other import si_sdr
import torch
import numpy as np


class ScoreModel(pl.LightningModule):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--lr", type=float, default=1e-4, help="The learning rate (1e-4 by default)")
        parser.add_argument("--ema_decay", type=float, default=0.999,
                            help="The parameter EMA decay constant (0.999 by default)")
        parser.add_argument("--t_eps", type=float, default=0.03, help="The minimum time (3e-2 by default)")
        parser.add_argument("--num_eval_files", type=int, default=20,
                            help="Number of files for speech enhancement performance evaluation during training. Pass 0 to turn off (no checkpoints based on evaluation metrics will be generated).")
        parser.add_argument("--loss_type", type=str, default="mse", choices=("mse", "mae"),
                            help="The type of loss function to use.")
        return parser

    def __init__(
            self, score,drift, sde, lr=1e-4, ema_decay=0.999, t_eps=3e-2,
            num_eval_files=20, loss_type='mse', data_module_cls=None, **kwargs
    ):
        """
        Create a new ScoreModel.

        Args:
            backbone: Backbone DNN that serves as a score-based model.
            sde: The SDE that defines the diffusion process.
            lr: The learning rate of the optimizer. (1e-4 by default).
            ema_decay: The decay constant of the parameter EMA (0.999 by default).
            t_eps: The minimum time to practically run for to avoid issues very close to zero (1e-5 by default).
            loss_type: The type of loss to use (wrt. noise z/std). Options are 'mse' (default), 'mae'
        """
        #pdb.set_trace()
        super().__init__()
        
        score = BackboneRegistry.get_by_name('ncsnpp_score')
        self.score = score(**kwargs)
        drift = BackboneRegistry.get_by_name('ncsnpp_drift')
        
        self.drift = drift(**kwargs)
        
        # Initialize SDE
        sde_cls = SDERegistry.get_by_name(sde)
        self.sde = sde_cls(**kwargs)
        # Store hyperparams and save them
        self.lr = lr
        self.ema_decay = ema_decay
        self.ema = ExponentialMovingAverage(self.parameters(), decay=self.ema_decay)
        self._error_loading_ema = False
        self.t_eps = t_eps
        self.loss_type = loss_type
        self.num_eval_files = num_eval_files
        self.lsx=0

        self.save_hyperparameters(ignore=['no_wandb'])
        self.data_module = data_module_cls(**kwargs, gpu=kwargs.get('gpus', 0) > 0)

    

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        # Method overridden so that the EMA params are updated after each optimizer step
        super().optimizer_step(*args, **kwargs)
        self.ema.update(self.parameters())

    # on_load_checkpoint / on_save_checkpoint needed for EMA storing/loading
    def on_load_checkpoint(self, checkpoint):
        ema = checkpoint.get('ema', None)
        if ema is not None:
            self.ema.load_state_dict(checkpoint['ema'])
        else:
            self._error_loading_ema = True
            warnings.warn("EMA state_dict not found in checkpoint!")

    def on_save_checkpoint(self, checkpoint):
        checkpoint['ema'] = self.ema.state_dict()

    def train(self, mode, no_ema=False):
        res = super().train(mode)  # call the standard `train` method with the given mode
        if not self._error_loading_ema:
            if mode == False and not no_ema:
                # eval
                self.ema.store(self.parameters())  # store current params in EMA
                self.ema.copy_to(self.parameters())  # copy EMA parameters over current params for evaluation
            else:
                # train
                if self.ema.collected_params is not None:
                    self.ema.restore(self.parameters())  # restore the EMA weights (if stored)
        return res

    def eval(self, no_ema=False):
        return self.train(False, no_ema=no_ema)
    
    def get_t(self,y,x,mean):
        

        return (((mean-x)/(y-x))[:,0,0,0]).abs()

    def _loss(self, err):
        if self.loss_type == 'mse':
            losses = torch.square(err.abs())
        elif self.loss_type == 'mae':
            losses = err.abs()
        # taken from reduce_op function: sum over channels and position and mean over batch dim
        # presumably only important for absolute loss number, not for gradients
        loss = torch.mean(0.5 * torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))
        return loss
    
    def _sisdr(self,s,s_hat):
        #pdb.set_trace()
        alpha=torch.div(torch.sum(torch.mul(s_hat,s),dim=-0),torch.sum(torch.mul(s,s),dim=0))
        #alpha = torch.dot(s_hat, s) / torch.norm(s)**2
        #sdr=10* torch.log10(torch.norm(alpha * s)**2 / torch.norm(alpha * s - s_hat)**2)
        sdr = 10 * torch.log10(torch.div(torch.sum(torch.mul(torch.mul(alpha,s),torch.mul(alpha,s)),dim=0) ,torch.sum(torch.mul(torch.mul(alpha,s)-s_hat,torch.mul(alpha,s)-s_hat),dim=0)))
        # 将最终的SDR值转换为NumPy数组
        return sdr
    
    def si_sdr(self,s, s_hat):
        alpha = np.dot(s_hat, s)/np.linalg.norm(s)**2   
        sdr = 10*np.log10(np.linalg.norm(alpha*s)**2/np.linalg.norm(alpha*s - s_hat)**2)
        return sdr



    def _step(self, batch, batch_idx):
        
        x, y,len1,norm = batch
        t = torch.rand(x.shape[0], device=x.device)*(self.sde.T-0.01)+0.01
        mean, std = self.sde.marginal_prob(x, t, y)
        z = torch.randn_like(x)  # i.i.d. normal distributed with var=0.5
        sigmas = std[:, None, None, None]
        perturbed_data = mean + sigmas * z
        target_snr = 0.5
        step_size = (target_snr * sigmas) ** 2 * 2
        #pdb.set_trace()
        
        random_number = torch.rand(1)*0.1+0.01 
        mask=~(step_size>1)
        step_size1=copy.deepcopy(step_size)
        step_size[mask]=random_number
        #pdb.set_trace()
        t1=t[:,None,None,None]-step_size
        mask=~(t1>0)
        step_size[mask]=t[:,None,None,None][mask]
        t1[mask]=0

        t1 = torch.exp(-self.sde.theta* t1)
        t2=torch.exp(-self.sde.theta* t[:,None,None,None])
        dnn_input = torch.cat([perturbed_data, y], dim=1)
        score = -self.score(dnn_input, t)
        x_mean = perturbed_data + step_size1 * score
        #dnn_input = torch.cat([x_mean, y], dim=1)
        drift = self.drift(dnn_input, t,torch.squeeze(torch.squeeze(torch.squeeze(step_size,1),1),1))
        x1 = x_mean - drift*step_size
        loss5 = self._loss(y-x-(y-x1)/(t1))
       
        
        err = (score)*sigmas  + z
        loss1 = self._loss(err)
        loss=loss5+loss1
       
        self.log('loss1',loss1,on_step=True,on_epoch=True)
        self.log('loss5',loss5,on_step=True,on_epoch=True)
        return loss
     



    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log('valid_loss', loss, on_step=False, on_epoch=True)

        # Evaluate speech enhancement performance
        if batch_idx == 0 and self.num_eval_files != 0 :
            pesq, si_sdr, estoi = evaluate_model(self, self.num_eval_files)
            #pesq1,sisdr_1,estoi_1=evaluate_model()
            self.log('pesq', pesq, on_step=False, on_epoch=True)
            self.log('si_sdr', si_sdr, on_step=False, on_epoch=True)
            self.log('estoi', estoi, on_step=False, on_epoch=True)
            print('############################################')

        return loss
    
    def forward1(self,x,t,y):
        dnn_input = torch.cat([x, y], dim=1)

        # the minus is most likely unimportant here - taken from Song's repo
        score = -self.score(dnn_input, t)
        return score
    
    def forward2(self, x, t,step_size, y):
        # Concatenate y as a,sn extra channel
        dnn_input = torch.cat([x, y], dim=1)

        # the minus is most likely unimportant here - taken from Song's repo
        
        drift = self.drift(dnn_input, t,step_size)
        return drift

    

    def forward(self, x, t,step_size,y,step=0.03):
        # Concatenate y as a,sn extra channel

        # the minus is most likely unimportant here - taken from Song's repo
        score = self.forward1(x,t,y)
        x1=x+score*step_size
        #if step<step_size:d
        #    step=step_sizedd
        #step=torch.tensor([0.03]).to(x.device)
        drift = self.forward2(x1,t,step,y)
        x=x1-drift*step #+ step_size[:, None, None, None] * score
        return x

    def to(self, *args, **kwargs):
        """Override PyTorch .to() to also transfer the EMA of the model weights"""
        self.ema.to(*args, **kwargs)

        return super().to(*args, **kwargs)

    def get_pc_sampler(self, predictor_name, corrector_name, y, N=None, minibatch=None, **kwargs):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs}
        if minibatch is None:
            return sampling.get_pc_sampler(predictor_name, corrector_name, sde=sde, score_fn=self, y=y, **kwargs)
        else:
            M = y.shape[0]

            def batched_sampling_fn():
                samples, ns = [], []
                for i in range(int(ceil(M / minibatch))):
                    y_mini = y[i * minibatch:(i + 1) * minibatch]
                    sampler = sampling.get_pc_sampler(predictor_name, corrector_name, sde=sde, score_fn=self, y=y_mini,
                                                      **kwargs)
                    sample, n = sampler()
                    samples.append(sample)
                    ns.append(n)
                samples = torch.cat(samples, dim=0)
                return samples, ns

            return batched_sampling_fn

    def get_ode_sampler(self, y, N=None, minibatch=None, **kwargs):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs}
        if minibatch is None:
            return sampling.get_ode_sampler(sde, self, y=y, **kwargs)
        else:
            M = y.shape[0]

            def batched_sampling_fn():
                samples, ns = [], []
                for i in range(int(ceil(M / minibatch))):
                    y_mini = y[i * minibatch:(i + 1) * minibatch]
                    sampler = sampling.get_ode_sampler(sde, self, y=y_mini, **kwargs)
                    sample, n = sampler()
                    samples.append(sample)
                    ns.append(n)
                samples = torch.cat(samples, dim=0)
                return sample, ns

            return batched_sampling_fn

    def train_dataloader(self):
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        return self.data_module.val_dataloader()

    def test_dataloader(self):
        return self.data_module.test_dataloader()

    def setup(self, stage=None):
        return self.data_module.setup(stage=stage)

    def to_audio(self, spec, length=None):
        return self._istft(self._backward_transform(spec), length)

    def _forward_transform(self, spec):
        return self.data_module.spec_fwd(spec)

    def _backward_transform(self, spec):
        return self.data_module.spec_back(spec)

    def _stft(self, sig):
        return self.data_module.stft(sig)

    def _istft(self, spec, length=None):
        return self.data_module.istft(spec, length)

    def enhance(self, y, sampler_type="pc", predictor="reverse_diffusion",
                corrector="ald", N=30, corrector_steps=1, snr=0.5, timeit=False,
                **kwargs
                ):
        """
        One-call speech enhancement of noisy speech `y`, for convenience.
        """
        sr = 16000
        start = time.time()
        T_orig = y.size(1)
        norm_factor = y.abs().max().item()
        y = y / norm_factor
        Y = torch.unsqueeze(self._forward_transform(self._stft(y.cuda())), 0)
        Y = pad_spec(Y)
        if sampler_type == "pc":
            sampler = self.get_pc_sampler(predictor, corrector, Y.cuda(), N=N,
                                          corrector_steps=corrector_steps, snr=snr, intermediate=False,
                                          **kwargs)
        elif sampler_type == "ode":
            sampler = self.get_ode_sampler(Y.cuda(), N=N, **kwargs)
        else:
            print("{} is not a valid sampler type!".format(sampler_type))
        sample, nfe = sampler()
        x_hat = self.to_audio(sample.squeeze(), T_orig)
        x_hat = x_hat * norm_factor
        x_hat = x_hat.squeeze().cpu().numpy()
        end = time.time()
        if timeit:
            rtf = (end - start) / (len(x_hat) / sr)
            return x_hat, nfe, rtf
        else:
            return x_hat
