from typing import Literal
import torch

__all__ = ["Hilbert2DTransformationMethod"]


class HilbertTransformations(torch.nn.Module):
    def __init__(
        self,
        method: Literal["combined", "single_orthant", "basic", "directional"],
        mode: Literal["HT", "AS"],
    ):
        super().__init__()

        self.method = self._chooseMethod(method, mode)

    def _sign(self, X, n):
        # Because the intersection of both conditions is empty, we can use the
        # (+) pixel-wise with 1.0 or 0.0 values
        return ((0 < X) & (X <= n / 2 - 1)).float() + -(
            (n / 2 + 1 <= X) & (X <= n - 1)
        ).float()

    def _chooseMethod(self, method, mode):
        match method:
            case "combined":

                def maskHilbertTransform(X, Y, img, axis=0):
                    return -0.5j * (
                        (1.0 if axis == 0 else -1.0)
                        * self._sign(X, img.size(-2))
                        + self._sign(Y, img.size(-1))
                    )

                def maskAnalyticSignal(X, Y, img, axis=0):
                    return 1 + 0.5 * (
                        (1.0 if axis == 0 else -1.0)
                        * self._sign(X, img.size(-2))
                        + self._sign(Y, img.size(-1))
                    )

            case "single_orthant":

                def maskHilbertTransform(X, Y, img, axis=None):
                    return (
                        self._sign(X, img.size(-2))
                        + self._sign(Y, img.size(-1))
                        + self._sign(X, img.size(-2))
                        * self._sign(Y, img.size(-1))
                    )

                def maskAnalyticSignal(X, Y, img, axis=None):
                    return (1 + self._sign(X, img.size(-2))) * (
                        1 + self._sign(Y, img.size(-1))
                    )

            case "basic":

                def maskHilbertTransform(X, Y, img, axis=None):
                    return -self._sign(X, img.size(-2)) * self._sign(
                        Y, img.size(-1)
                    )

                def maskAnalyticSignal(X, Y, img, axis=None):
                    return 1 - 1j * self._sign(X, img.size(-2)) * self._sign(
                        Y, img.size(-1)
                    )

            case "directional":

                def maskHilbertTransform(X, Y, img, axis=0):
                    return -1j * (
                        self._sign(X, img.size(-2))
                        if axis == 0
                        else self._sign(Y, img.size(-1))
                    )

                def maskAnalyticSignal(X, Y, img, axis=0):
                    return 1 + (
                        self._sign(X, img.size(-2))
                        if axis == 0
                        else self._sign(Y, img.size(-1))
                    )

            case _:
                raise ValueError(f"Invalid method: {method}")

        return maskHilbertTransform if mode == "HT" else maskAnalyticSignal

    def forward(self, img, axis=0):
        X, Y = torch.meshgrid(
            torch.fft.fftfreq(img.size(-2)),
            torch.fft.fftfreq(img.size(-1)),
            indexing="ij",
        )
        return torch.fft.ifft2(
            self.method(X, Y, img, axis=axis) * torch.fft.fft2(img)
        )


# def _compute(self, target, img, axis=0):
#   X,Y = torch.meshgrid(torch.fft.fftfreq(img.size(-2)), torch.fft.fftfreq
# (img.size(-1)), indexing="ij")
#   return torch.fft.ifft2(
#       (self.maskHilbertTransform if target=="HT" else s
# elf.maskAnalyticSignal)(X,Y, img, axis=axis)
#       *
#       torch.fft.fft2(img)
#   )

# def computeHT(self, img, axis=0): return self._compute("HT", img, axis=axis)
# def computeAS(self, img, axis=0):return self._compute("AS", img, axis=axis)


# class _Hilbert2DTransformationMethod(ABC):

#     @abstractmethod
#     def maskHilbertTransform(self, X, Y, img, axis=0):
#         pass

#     @abstractmethod
#     def maskAnalyticSignal(self, X, Y, img, axis=0):
#         pass


# class _CombinedCls(_Hilbert2DTransformationMethod):
#     def maskHilbertTransform(self, X, Y, img, axis=0):
#         return -0.5j * (
#             (1.0 if axis == 0 else -1.0) * _sign(X, img.size(-2))
#             + _sign(Y, img.size(-1))
#         )

#     def maskAnalyticSignal(self, X, Y, img, axis=0):
#         return 1 + 0.5 * (
#             (1.0 if axis == 0 else -1.0) * _sign(X, img.size(-2))
#             + _sign(Y, img.size(-1))
#         )


# class _SingleOrthantCls(_Hilbert2DTransformationMethod):
#     def maskHilbertTransform(self, X, Y, img, **kwargs):
#         return (
#             _sign(X, img.size(-2))
#             + _sign(Y, img.size(-1))
#             + _sign(X, img.size(-2)) * _sign(Y, img.size(-1))
#         )

#     def maskAnalyticSignal(self, X, Y, img, **kwargs):
#         return (1 + _sign(X, img.size(-2))) * (1 + _sign(Y, img.size(-1)))


# class _BasicCls(_Hilbert2DTransformationMethod):
#     def maskHilbertTransform(self, X, Y, img, **kwargs):
#         return -_sign(X, img.size(-2)) * _sign(Y, img.size(-1))

#     def maskAnalyticSignal(self, X, Y, img, **kwargs):
#         return 1 - 1j * _sign(X, img.size(-2)) * _sign(Y, img.size(-1))


# class _DirectionalCls(_Hilbert2DTransformationMethod):
#     def maskHilbertTransform(self, X, Y, img, axis=0):
#         return -1j * (
#             _sign(X, img.size(-2)) if axis == 0 else _sign(Y, img.size(-1))
#         )

#     def maskAnalyticSignal(self, X, Y, img, axis=0):
#         return 1 + (
#             _sign(X, img.size(-2)) if axis == 0 else _sign(Y, img.size(-1))
#         )
