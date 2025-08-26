import torch
import numpy as np

def get_grad_log_ratio(discriminator, x, t_hat, img_resolution, S_clip_min, S_clip_max, labels, logit=False):
    mean_vp_tau, tau = vpsde().transform_unnormalized_wve_to_normalized_vp(t_hat) ## VP pretrained classifier
    if t_hat > S_clip_max or t_hat < S_clip_min:
        if logit:
          return torch.zeros_like(x), torch.ones(x.shape[0], device=x.device)
        return torch.zeros_like(x)
    x_ = mean_vp_tau * x

    with torch.enable_grad():
        x_ = x.float().clone().detach().requires_grad_()

        # sigma_ = torch.ones(x.shape[0], device=x.device) * t_hat
        sigma_ = torch.ones(x.shape[0], device=x.device) * tau

        log_ratio, pred = get_log_ratio(discriminator, x_, sigma_, labels)
        discriminator_guidance_score = torch.autograd.grad(outputs=log_ratio.sum(), inputs=x_, retain_graph=False)[0]
        # print(mean_vp_tau.shape)
        # print(std_wve_t.shape)
        # print(discriminator_guidance_score.shape)
        # discriminator_guidance_score *= - (t_hat ** 2)
        discriminator_guidance_score *= - ((t_hat ** 2) * mean_vp_tau)
    # print(discriminator_guidance_score)
    if logit:
      return discriminator_guidance_score, pred
    return discriminator_guidance_score


def get_log_ratio(discriminator, input, sigma, labels):
    if discriminator == None:
        return torch.zeros(input.shape[0], device=input.device), None
    else:
        logits = discriminator(input, sigma, labels)
        prediction = torch.clip(logits, 1e-5, 1. - 1e-5)
        log_ratio = torch.log(prediction / (1. - prediction))
        return log_ratio, prediction


class vpsde():
    def __init__(self):
        self.beta_0 = 0.1
        self.beta_1 = 20.
        self.s = 0.008
        self.f_0 = np.cos(self.s / (1. + self.s) * np.pi / 2.) ** 2

    @property
    def T(self):
        return 1

    def compute_tau(self, std_wve_t):
        tau = -self.beta_0 + torch.sqrt(self.beta_0 ** 2 + 2. * (self.beta_1 - self.beta_0) * torch.log(1. + std_wve_t ** 2))
        tau /= self.beta_1 - self.beta_0
        return tau

    def marginal_prob(self, t):
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        mean = torch.exp(log_mean_coeff)
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        return mean, std

    def transform_unnormalized_wve_to_normalized_vp(self, t, std_out=False):
        tau = self.compute_tau(t)
        mean_vp_tau, std_vp_tau = self.marginal_prob(tau)
        if std_out:
            return mean_vp_tau, std_vp_tau, tau
        return mean_vp_tau, tau

    def compute_t_cos_from_t_lin(self, t_lin):
        sqrt_alpha_t_bar = torch.exp(-0.25 * t_lin ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t_lin * self.beta_0)
        time = torch.arccos(np.sqrt(self.f_0) * sqrt_alpha_t_bar)
        t_cos = self.T * ((1. + self.s) * 2. / np.pi * time - self.s)
        return t_cos

    def get_diffusion_time(self, batch_size, batch_device, t_min=1e-5, importance_sampling=True):
        if importance_sampling:
            Z = self.normalizing_constant(t_min)
            u = torch.rand(batch_size, device=batch_device)
            return (-self.beta_0 + torch.sqrt(self.beta_0 ** 2 + 2 * (self.beta_1 - self.beta_0) *
                    torch.log(1. + torch.exp(Z * u + self.antiderivative(t_min))))) / (self.beta_1 - self.beta_0), Z.detach()
        else:
            return torch.rand(batch_size, device=batch_device) * (self.T - t_min) + t_min, 1

    def antiderivative(self, t, stabilizing_constant=0.):
        if isinstance(t, float) or isinstance(t, int):
            t = torch.tensor(t).float()
        return torch.log(1. - torch.exp(- self.integral_beta(t)) + stabilizing_constant) + self.integral_beta(t)

    def normalizing_constant(self, t_min):
        return self.antiderivative(self.T) - self.antiderivative(t_min)

    def integral_beta(self, t):
        return 0.5 * t ** 2 * (self.beta_1 - self.beta_0) + t * self.beta_0