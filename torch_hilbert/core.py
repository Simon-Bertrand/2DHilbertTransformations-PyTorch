from abc import ABC, abstractmethod
import torch

def sign(X, n):
  #Because the intersection of both conditions is empty, we can use the (+) pixel-wise with 1.0 or 0.0 values
  return ((0<X)&(X<=n/2-1)).float() + -((n/2+1<=X)&(X<=n-1)).float()

class Hilbert2DTransformationMethod(ABC):
  @abstractmethod
  def maskHilbertTransform(self, X, Y, img, axis=0):pass
  @abstractmethod
  def maskAnalyticSignal(self, X,Y, img, axis=0):pass
  def _compute(self, target, img, axis=0):
    X,Y = torch.meshgrid(torch.fft.fftfreq(img.shape[0]), torch.fft.fftfreq(img.shape[1]), indexing="ij")
    return torch.fft.ifft2(
        (self.maskHilbertTransform if target=="HT" else self.maskAnalyticSignal)(X,Y, img, axis=axis)
        *
        torch.fft.fft2(img)
    )

  def computeHT(self, img, axis=0): return self._compute("HT", img, axis=axis)
  def computeAS(self, img, axis=0):return self._compute("AS", img, axis=axis)


class CombinedCls(Hilbert2DTransformationMethod):
  def maskHilbertTransform(self, X,Y, img, axis=0):
    return -0.5j*((1. if axis==0 else -1.)*sign(X, img.shape[0]) + sign(Y, img.shape[1]))
  def maskAnalyticSignal(self, X,Y, img, axis=0):
    return 1+0.5*((1. if axis==0 else -1.)*sign(X, img.shape[0]) + sign(Y, img.shape[1]))

class SingleOrthantCls(Hilbert2DTransformationMethod):
  def maskHilbertTransform(self, X,Y, img, **kwargs):
    return sign(X, img.shape[0]) + sign(Y, img.shape[1]) + sign(X, img.shape[0]) * sign(Y, img.shape[1])
  def maskAnalyticSignal(self, X,Y, img, **kwargs):
    return (1+sign(X, img.shape[0]))*(1+sign(Y, img.shape[1]))

class BasicCls(Hilbert2DTransformationMethod):
  def maskHilbertTransform(self, X,Y, img,**kwargs):
    return -sign(X, img.shape[0]) * sign(Y, img.shape[1])
  def maskAnalyticSignal(self, X,Y, img,**kwargs):
    return (1 - 1j* sign(X, img.shape[0])*sign(Y, img.shape[1]))

class DirectionalCls(Hilbert2DTransformationMethod):
  def maskHilbertTransform(self, X,Y, img, axis=0):
    return -1j*(sign(X, img.shape[0]) if axis==0 else sign(Y, img.shape[1]))
  def maskAnalyticSignal(self, X,Y, img, axis=0):
    return (1 + (sign(X, img.shape[0]) if axis==0 else sign(Y, img.shape[1])))