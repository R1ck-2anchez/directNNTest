import torch
from torchvision import transforms


class SDTransf:
    def __init__(self, infty=True, light=True):
        self.infty = infty
        self.light = light

    def random_trans(self, X, radius):
        #This function generates a random perturbation within the lp ball of radius, with shape the same as X
        if self.light:
           return self.random_trans_light(X, radius)
        x_shape = X.shape
        flattened_X = torch.flatten(X, start_dim=1)
        delta = torch.rand(flattened_X.shape)-0.5
        if self.infty:
            norms = torch.norm(delta, p=torch.inf, dim=1)
        else:
            norms = torch.norm(delta, p=2, dim=1)
        ratios = 1/norms
        clipped_ratios = torch.max(1/radius*torch.ones(ratios.shape), ratios)
        normalizer = clipped_ratios*norms
        delta = delta/(normalizer[:, None])
        delta = torch.reshape(delta, x_shape)
        return delta
    
    def random_trans_light(self, X, radius):
      #start = time.time()
      X_shape = X.shape
      x = X[0]
      x_shape = x.shape
      flattened_X = torch.flatten(x, start_dim=0)
      delta = torch.rand(flattened_X.shape)-0.5     
      if self.infty:
        norms = torch.norm(delta, p=torch.inf)
      else:
        norms = torch.norm(delta, p=2)
      ratios = 1/norms
      clipped_ratios = torch.max(1/radius*torch.ones(ratios.shape), ratios)
      normalizer = clipped_ratios*norms
      delta = delta/(normalizer)
      delta = torch.reshape(delta, x_shape)
      delta = torch.unsqueeze(delta, 0)
      delta = torch.cat(X_shape[0]*[delta], 0)
      return delta


class NaturalTransf:
    # full list of transforms: https://pytorch.org/vision/stable/transforms.html
    def __init__(self, seed=123):
        torch.manual_seed(seed)

    def shear(self, X, degree=45):
        return transforms.RandomAffine(degrees=0, shear=degree)(X)

    def shift(self, X, shift=0.2):
        return transforms.RandomAffine(degrees=0, translate=(shift, shift))(X)

    def zoom(self, X, ratio=0.8):
        input_dim = X.shape[-2]
        return transforms.RandomResizedCrop(size=(input_dim, input_dim), ratio=(ratio, ratio))(X)

    def rotate(self, X, degree=20):
        return transforms.RandomRotation(degrees=degree)(X)

    def brightness(self, X, bright=.5):
        return transforms.ColorJitter(brightness=bright)(X)

    def blur(self, X, sig=0.1):
        return transforms.GaussianBlur(kernel_size=(5, 9), sigma=(sig, sig+0.1))(X)
    
    def contrast(self, X, cf=0.9):
        return transforms.functional.adjust_contrast(X, contrast_factor=cf)
    
    def gen_natural(self, X):
        deltas = []
        degs = [(1+i) for i in range(4)]
        for deg in degs:
          X_trans = self.shear(X, deg)
          delta = X_trans-X
          deltas.append(delta)

        shifts = [0.02*(1+i) for i in range(4)]
        for shift in shifts:
          X_trans = self.shift(X, shift)
          delta = X_trans-X
          deltas.append(delta)

        ratios = [0.9+0.04*i for i in range(4)]
        for ratio in ratios:
          X_trans = self.zoom(X, ratio)
          delta = X_trans-X
          deltas.append(delta)

        degs = [2+2*i for i in range(4)]
        for deg in degs:
          X_trans = self.rotate(X, deg)
          delta = X_trans-X
          deltas.append(delta)

        brights = [0.8+0.1*i for i in range(4)]
        for bright in brights:
          X_trans = self.brightness(X, bright)
          delta = X_trans-X
          deltas.append(delta)

        sigmas = [0.05*(1+i) for i in range(4)]
        for sigma in sigmas:
          X_trans = self.blur(X, sigma)
          delta = X_trans-X
          deltas.append(delta)

        cfs = [0.87+0.07*i for i in range(4)]
        for cf in cfs:
          X_trans = self.contrast(X, cf)
          delta = X_trans-X
          deltas.append(delta)
        
        return deltas


class DataGen:
  def __init__(self, l2_nums, linf_nums, l2_rad, linf_rad, l2_light=True, linf_light=True):
    self.linf_trans = SDTransf(light=linf_light)
    self.l2_trans = SDTransf(infty=False, light=l2_light)
    self.natural_trans = NaturalTransf()
    self.l2_nums = l2_nums
    self.linf_nums = linf_nums
    self.l2_rad = l2_rad
    self.linf_rad = linf_rad

  def lp_gen(self, x):
    deltas = []
    for _ in range(self.linf_nums):
      delta = self.linf_trans.random_trans(x, self.linf_rad)
      deltas.append(delta)

    for _ in range(self.l2_nums):
      delta = self.l2_trans.random_trans(x, self.l2_rad)
      deltas.append(delta)
    print("Size of lp deltas:", len(deltas))
    return deltas
  
  def natural_gen(self, x):
     return self.natural_trans.gen_natural(x)
  
  def mixed_gen(self, x):
     sd_deltas = self.lp_gen(x)
     natural_deltas = self.natural_gen(x)
     return sd_deltas + natural_deltas